"""Conservative arrangement-role hints, not a trained or calibrated classifier.

AudioSet supplies instrument-family evidence. Harmonic grouping and temporal
features supply role-form evidence. Neither stem names nor embeddings are labels.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, resample_poly, stft

METHOD = "instrument_gated_acoustic_roles_v1"
ELIGIBLE_VIEWS = frozenset({"mix", "bass", "guitar", "piano", "other"})
SYNTH_INDICES = (158,)  # Sampler/keyboard alone does not prove a synthesizer.
GUITAR_INDICES = (140, 141, 143, 144, 145, 146)  # Exclude bass and plucked parent.
FAMILY_THRESHOLD = 0.20
SAMPLE_RATE = 16_000
FRAME_HOP = 320
FRAME_SIZE = 4096
FRAME_SECONDS = FRAME_HOP / SAMPLE_RATE


@dataclass(frozen=True)
class RoleSpec:
    role_id: str
    label: str
    family: str


ROLE_SPECS = (
    RoleSpec("synth_bass", "Synth Bass", "synthesizer"),
    RoleSpec("synth_pad", "Synth Pad", "synthesizer"),
    RoleSpec("synth_lead", "Synth Lead", "synthesizer"),
    RoleSpec("synth_chord", "Synth Chord", "synthesizer"),
    RoleSpec("lead_guitar", "Lead Guitar", "guitar"),
    RoleSpec("rhythm_guitar", "Rhythm Guitar", "guitar"),
)


def _check_cancel(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("编曲角色分析已取消")


def family_scores(scores: np.ndarray) -> dict[str, float]:
    specific_guitar = float(np.max(scores[list(GUITAR_INDICES[1:])]))
    generic_guitar = float(scores[140])
    # AudioSet's Guitar parent may fire for Bass guitar. Do not turn bass-only
    # parent evidence into Lead/Rhythm Guitar.
    bass_guitar = float(scores[142])
    if bass_guitar >= 0.20 and bass_guitar >= generic_guitar * 0.65:
        generic_guitar = 0.0
    return {
        "synthesizer": float(np.max(scores[list(SYNTH_INDICES)])),
        "guitar": max(specific_guitar, generic_guitar),
    }


def needs_role_features(scores: np.ndarray, view: str) -> bool:
    return view in ELIGIBLE_VIEWS and max(family_scores(scores).values()) >= FAMILY_THRESHOLD


def _activity_intervals(active: np.ndarray, duration: float) -> list[tuple[float, float]]:
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    intervals = []
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        # STFT uses centered frames. Preserve rests rather than unioning windows.
        left = max(0.0, (start - 0.5) * FRAME_SECONDS)
        right = min(duration, (end - 0.5) * FRAME_SECONDS)
        if right - left >= 0.12:
            intervals.append((left, right))
    return intervals


def _frame_roots(spectrum: np.ndarray, frequencies: np.ndarray) -> list[float]:
    """Group observed spectral peaks by integer harmonics before counting voices.

    Prefer an observed fundamental. A missing fundamental requires a near-complete
    2/3/4 harmonic series; a major chord's shared 4/5/6 subharmonic is insufficient.
    """
    maximum = float(np.max(spectrum))
    if maximum <= 1e-10:
        return []
    peaks, _ = find_peaks(spectrum, height=maximum * 0.10, prominence=maximum * 0.035)
    peaks = peaks[(frequencies[peaks] >= 38) & (frequencies[peaks] <= 5000)]
    if not len(peaks):
        return []
    peaks = peaks[np.argsort(spectrum[peaks])[-36:]]
    logmag = np.log(np.maximum(spectrum, 1e-12))
    offset = 0.5 * (logmag[peaks - 1] - logmag[peaks + 1]) / (
        logmag[peaks - 1] - 2 * logmag[peaks] + logmag[peaks + 1] - 1e-12
    )
    pitch_hz = (peaks + np.clip(offset, -0.5, 0.5)) * SAMPLE_RATE / FRAME_SIZE
    energy = spectrum[peaks].astype(np.float64) ** 2
    total = float(np.sum(energy))
    remaining = energy.copy()
    roots = []
    for _ in range(4):
        best = None
        for index, root in enumerate(pitch_hz):
            if remaining[index] < total * 0.025 or root > 1800:
                continue
            multiples = np.rint(pitch_hz / root)
            tolerance = np.maximum(5.0, pitch_hz * 0.018)
            matches = (
                (multiples >= 1) & (multiples <= 20)
                & (np.abs(pitch_hz - multiples * root) <= tolerance)
            )
            explained = float(np.sum(remaining[matches]))
            if best is None or explained > best[0]:
                best = (explained, float(root), matches)
        if not roots:
            for root in pitch_hz / 2:
                if root < 38:
                    continue
                multiples = np.rint(pitch_hz / root)
                matches = (
                    (multiples >= 2) & (multiples <= 20)
                    & (np.abs(pitch_hz - multiples * root) <= np.maximum(5.0, pitch_hz * 0.018))
                )
                explained = float(np.sum(energy[matches]))
                if (
                    {2, 3, 4} <= set(multiples[matches])
                    and explained >= total * 0.90
                    and (best is None or explained > best[0] * 1.1)
                ):
                    best = (explained, float(root), matches)
        if best is None or best[0] < total * 0.13:
            break
        roots.append(best[1])
        remaining[best[2]] = 0
    return roots


def extract_role_features(
    audio: np.ndarray,
    sample_rate: int,
    *,
    cancel_event: threading.Event | None = None,
) -> dict:
    """Summarize a real, unpadded window; output only small finite values."""
    _check_cancel(cancel_event)
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0, posinf=0, neginf=0)
    duration = len(audio) / sample_rate
    empty = {"duration": duration, "activity": [], "active_fraction": 0.0}
    if duration < 0.5 or not len(audio) or float(np.max(np.abs(audio))) < 1e-6:
        return empty
    divisor = math.gcd(sample_rate, SAMPLE_RATE)
    audio = resample_poly(audio, SAMPLE_RATE // divisor, sample_rate // divisor)
    frequencies, _, spectrum = stft(
        audio, fs=SAMPLE_RATE, nperseg=FRAME_SIZE, noverlap=FRAME_SIZE - FRAME_HOP,
        boundary="zeros", padded=False,
    )
    magnitude = np.abs(spectrum)
    power = magnitude ** 2
    envelope = np.sqrt(np.sum(power, axis=0))
    reference = float(np.quantile(envelope, 0.9))
    active = envelope > max(1e-6, reference * 0.10)
    activity = _activity_intervals(active, duration)
    if not activity:
        return empty
    band = magnitude[(frequencies >= 40) & (frequencies <= 5000)]
    flatness = np.exp(np.mean(np.log(band + 1e-10), axis=0)) / (np.mean(band, axis=0) + 1e-10)
    root_counts = np.zeros(len(envelope), dtype=int)
    pitches = np.full(len(envelope), np.nan)
    for frame in np.flatnonzero(active):
        if frame % 25 == 0:
            _check_cancel(cancel_event)
        if flatness[frame] > 0.30:
            continue
        roots = _frame_roots(magnitude[:, frame], frequencies)
        # Octave doubling is one musical line, not chord evidence.
        classes: list[float] = []
        for root in roots:
            midi = 69 + 12 * math.log2(root / 440)
            if not any(abs((midi - previous + 6) % 12 - 6) < 0.7 for previous in classes):
                classes.append(midi)
        root_counts[frame] = len(classes)
        if classes:
            pitches[frame] = classes[0]
    valid = active & np.isfinite(pitches)
    mono = valid & (root_counts == 1)
    active_count = max(1, int(np.sum(active)))
    mono_pitches = pitches[mono]
    # Stable note runs, with unvoiced/polyphonic gaps left as gaps.
    rounded = median_filter(np.where(mono, np.rint(pitches), -100), size=5)
    edges = np.r_[0, np.flatnonzero(np.diff(rounded) != 0) + 1, len(rounded)]
    notes = [
        int(rounded[start])
        for start, end in zip(edges[:-1], edges[1:], strict=True)
        if rounded[start] > -100 and (end - start) * FRAME_SECONDS >= 0.12
    ]
    note_changes = [note for i, note in enumerate(notes) if not i or note != notes[i - 1]]
    repetition = 0.0
    for lag in range(2, min(9, len(note_changes) // 2 + 1)):
        repetition = max(
            repetition,
            float(np.mean(np.abs(np.subtract(note_changes[lag:], note_changes[:-lag])) <= 1)),
        )
    # Attack peaks use positive spectral flux, normalized by window amplitude.
    flux = np.r_[0, np.sum(np.maximum(np.diff(magnitude, axis=1), 0), axis=0)]
    scale = float(np.quantile(np.sum(magnitude, axis=0)[active], 0.75))
    flux /= max(scale, 1e-9)
    attacks, _ = find_peaks(flux, height=0.13, prominence=0.08, distance=8)
    attacks = attacks[active[attacks]]
    onsets = int(len(attacks))
    if active[0] and (not len(attacks) or attacks[0] * FRAME_SECONDS > 0.20):
        onsets += 1
    active_seconds = min(duration, active_count * FRAME_SECONDS)
    return {
        "duration": float(duration),
        "activity": activity,
        "active_fraction": float(np.mean(active)),
        "tonal_fraction": float(np.sum(valid) / active_count),
        "mono_fraction": float(np.sum(mono) / active_count),
        "poly_fraction": float(np.sum(active & (root_counts >= 2)) / active_count),
        "pitch_median": float(np.median(mono_pitches)) if len(mono_pitches) else 0.0,
        "pitch_range": float(np.ptp(np.quantile(mono_pitches, [0.1, 0.9]))) if len(mono_pitches) else 0.0,
        "low_fraction": float(np.mean(mono_pitches < 53)) if len(mono_pitches) else 0.0,
        "note_changes": len(note_changes),
        "unique_notes": len(set(note_changes)),
        "repetition": repetition,
        "onset_rate": onsets / max(duration, 0.5),
        "onsets": onsets,
        "envelope_variation": float(np.std(envelope[active]) / max(np.mean(envelope[active]), 1e-9)),
        "active_seconds": float(active_seconds),
    }


def infer_role_forms(features: dict) -> dict[str, tuple[float, list[str]]]:
    """Independent gates allow overlapping roles and conservative abstention."""
    f = features
    if f.get("duration", 0) < 1.5 or f.get("active_seconds", 0) < 1.0 or f.get("tonal_fraction", 0) < 0.65:
        return {}
    result = {}
    mono = f["mono_fraction"] >= 0.72
    poly = f["poly_fraction"] >= 0.48
    low = f["low_fraction"] >= 0.72
    melodic = (
        mono and not low and f["pitch_median"] >= 55
        and f["pitch_range"] >= 3 and f["note_changes"] >= 4 and f["unique_notes"] >= 3
    )
    riff = (
        mono and f["pitch_median"] < 64 and f["repetition"] >= 0.80
        and f["note_changes"] >= 8 and f["onset_rate"] >= 1.0
    )
    if mono and low:
        result["synth_bass"] = (0.82, ["低音区占比高", "以单音线条为主"])
    if (
        f["active_fraction"] >= 0.88 and f["onset_rate"] <= 0.65
        and f["envelope_variation"] <= 0.30
        and (poly or (mono and f["note_changes"] <= 3))
        and not low
    ):
        result["synth_pad"] = (0.76, ["持续性高、起音稀疏", "音量变化平缓"])
    if poly:
        result["synth_chord"] = (0.78, ["多个独立音高同时发声", "已按泛音关系合并同一音的频谱峰"])
    if melodic:
        result["synth_lead"] = (0.78, ["中高音区单音旋律", "存在多个稳定音高及旋律运动"])
        if not riff:
            result["lead_guitar"] = (0.74, ["单音旋律及音高运动", "未发现明显低音重复节奏型"])
    if (poly and f["onset_rate"] >= 0.8 and f["onsets"] >= 4) or riff:
        result["rhythm_guitar"] = (
            0.78,
            ["和弦型重复起音"] if poly else ["中低音区反复音型", "起音密集、节奏性强"],
        )
    return result


def _ownership(windows: list[dict], index: int) -> tuple[float, float]:
    centers = [(w["start_sec"] + w["end_sec"]) / 2 for w in windows]
    return (
        windows[0]["start_sec"] if index == 0 else (centers[index - 1] + centers[index]) / 2,
        windows[-1]["end_sec"] if index == len(windows) - 1 else (centers[index] + centers[index + 1]) / 2,
    )


def fuse_role_predictions(
    analyses: dict[str, dict],
    *,
    cancel_event: threading.Event | None = None,
) -> dict:
    _check_cancel(cancel_event)
    mix = analyses["mix"]
    windows = mix["windows"]
    segments: dict[str, list[dict]] = {spec.role_id: [] for spec in ROLE_SPECS}
    for index, window in enumerate(windows):
        _check_cancel(cancel_event)
        views = []
        mix_rms = max(float(mix.get("window_rms", [mix["rms"]] * len(windows))[index]), 1e-9)
        for view, analysis in analyses.items():
            if view not in ELIGIBLE_VIEWS:
                continue
            features = analysis.get("role_features", [])
            if index >= len(features) or not features[index]:
                continue
            rms = float(analysis.get("window_rms", [analysis["rms"]] * len(windows))[index])
            if rms < 1e-6 or (view != "mix" and rms / mix_rms < 10 ** (-32 / 20)):
                continue
            families = family_scores(analysis["scores"][index])
            forms = infer_role_forms(features[index])
            views.append((view, families, forms, features[index]))
        left, right = _ownership(windows, index)
        for spec in ROLE_SPECS:
            eligible = [v for v in views if v[1][spec.family] >= FAMILY_THRESHOLD]
            # A family-bearing stem is preferable to unrelated notes in a mix.
            stems = [v for v in eligible if v[0] != "mix"]
            pool = stems or [v for v in eligible if v[1][spec.family] >= 0.35]
            matches = [v for v in pool if spec.role_id in v[2]]
            if not matches:
                continue
            selected = max(matches, key=lambda v: v[1][spec.family])
            view, families, forms, features = selected
            form_score, reasons = forms[spec.role_id]
            score = round(min(families[spec.family], form_score), 3)
            evidence = [view]
            if view != "mix" and any(v[0] == "mix" and v[1][spec.family] >= FAMILY_THRESHOLD for v in eligible):
                evidence.append("mix")
            for start, end in features["activity"]:
                start = max(left, window["start_sec"] + start)
                end = min(right, window["start_sec"] + end)
                if end - start < 0.12:
                    continue
                segments[spec.role_id].append({
                    "start_sec": round(start, 3), "end_sec": round(end, 3),
                    "score": score, "evidence": evidence,
                    "reasons": [f"检测到{'合成器' if spec.family == 'synthesizer' else '吉他'}类别证据", *reasons],
                })
    roles = []
    for spec in ROLE_SPECS:
        merged: list[dict] = []
        for segment in segments[spec.role_id]:
            if merged and segment["start_sec"] <= merged[-1]["end_sec"] + 0.025:
                previous = merged[-1]
                previous["end_sec"] = max(previous["end_sec"], segment["end_sec"])
                previous["score"] = max(previous["score"], segment["score"])
                previous["evidence"] = sorted(set(previous["evidence"] + segment["evidence"]))
                previous["reasons"] = list(dict.fromkeys(previous["reasons"] + segment["reasons"]))
            else:
                merged.append(dict(segment))
        if merged:
            roles.append({
                "role_id": spec.role_id, "label": spec.label,
                "instrument_family": spec.family,
                "confidence": "possible", "confidence_label": "辅助推断",
                "score": max(s["score"] for s in merged),
                "evidence": sorted({v for s in merged for v in s["evidence"]}),
                "reasons": list(dict.fromkeys(r for s in merged for r in s["reasons"])),
                "segments": merged,
            })
    return {
        "schema_version": 1, "method": METHOD,
        "score_semantics": "heuristic_evidence_not_probability",
        "window_sec": 10.0, "hop_sec": 5.0,
        "status": "candidates" if roles else "undetermined",
        "roles": roles,
        "summary": (
            "编曲角色为辅助推断，可同时出现；请结合分轨试听核对。"
            if roles else "当前证据不足以判断编曲角色，不代表这些角色不存在。"
        ),
        "limitations": "基于乐器类别与声学形态，未经角色标注训练；约 5 秒上下文分辨率，时间段非逐音符标注。",
    }
