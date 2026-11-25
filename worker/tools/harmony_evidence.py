"""Conservative signal evidence for rejecting sparse, non-harmonic inputs.

These rules are diagnostics, not calibrated probabilities. They deliberately
abstain only on sustained bass/percussion dominance or near-monophonic spectra.
"""

from __future__ import annotations


def measure_harmony(cqt, active):
    import numpy as np
    from scipy.ndimage import median_filter

    cqt = np.asarray(cqt, dtype=np.float32)
    active = np.asarray(active, dtype=bool)
    if not active.any():
        return {"insufficient": True, "reason": "silence", "active_frames": 0}

    # CQT geometry is shared with the neural model: F#0, 36 bins/octave.
    horizontal = median_filter(cqt, size=(1, 17))
    vertical = median_filter(cqt, size=(17, 1))
    harmonic = cqt * horizontal**2 / np.maximum(
        horizontal**2 + vertical**2, 1e-12
    )
    frequencies = 23.12465141947715 * 2 ** (np.arange(cqt.shape[0]) / 36)
    harmonic_sum = harmonic.sum(axis=0)
    bass_share = harmonic[frequencies < 220].sum(axis=0) / np.maximum(
        harmonic_sum, 1e-12
    )
    median_bass_share = float(np.median(bass_share[active]))
    harmonic_share = float(harmonic[:, active].sum() / max(cqt[:, active].sum(), 1e-12))

    # A pure pitched source and its overtones are not three independent notes.
    peak = np.argmax(harmonic, axis=0)
    bins = np.arange(cqt.shape[0])[:, None]
    explained = np.zeros_like(cqt, dtype=bool)
    for multiple in range(1, 9):
        target = peak + 36 * np.log2(multiple)
        explained |= np.abs(bins - target[None, :]) <= 3
    comb_share = (harmonic * explained).sum(axis=0) / np.maximum(harmonic_sum, 1e-12)
    monophonic_share = float(np.mean(comb_share[active] >= 0.90))
    power = np.maximum(cqt[:, active] ** 2, 1e-20)
    flatness = float(np.median(
        np.exp(np.mean(np.log(power), axis=0)) / np.mean(power, axis=0)
    ))
    bass_percussion = median_bass_share >= 0.90 and harmonic_share < 0.65
    # A median tolerates short attack/release transients on brief clips.
    median_comb_share = float(np.median(comb_share[active]))
    monophonic = median_comb_share >= 0.90
    # Percussive attacks leave residual energy outside a single pitched comb.
    # Require bass dominance of the peak as well as 3/4 explained harmonic energy.
    pitched_bass_with_drums = (
        median_comb_share >= 0.75
        and harmonic_share < 0.90
        and float(np.median(frequencies[peak[active]])) < 220
    )
    noise_like = flatness >= 0.35
    insufficient = bass_percussion or monophonic or noise_like or pitched_bass_with_drums
    return {
        "insufficient": bool(insufficient),
        "reason": (
            "bass_percussion_dominant" if bass_percussion
            else "single_pitch_dominant" if monophonic
            else "single_bass_with_percussion" if pitched_bass_with_drums
            else "noise_like_spectrum" if noise_like
            else "no_conclusive_rejection"
        ),
        "active_frames": int(active.sum()),
        "median_bass_share": round(median_bass_share, 5),
        "harmonic_share": round(harmonic_share, 5),
        "monophonic_frame_share": round(monophonic_share, 5),
        "median_harmonic_comb_share": round(median_comb_share, 5),
        "spectral_flatness": round(flatness, 5),
        "method": "conservative_spectral_evidence_v1",
    }


def smooth_display_gaps(progression):
    """Return a display-only timeline; exports and metrics retain raw rests.

    Only bridge <100 ms internal gaps with identical reliable neighbours.
    Different chords, weak decisions, and real rests remain untouched.
    """
    result = [dict(item) for item in progression]
    for index in range(1, len(result) - 1):
        left, gap, right = result[index - 1:index + 2]
        if (
            gap["chord"] == "N"
            and gap["duration_sec"] < 0.1
            and left["chord"] == right["chord"] != "N"
            and min(left["confidence"], right["confidence"]) >= 0.5
            and abs(left["start_time"] + left["duration_sec"] - gap["start_time"]) < 0.002
            and abs(gap["start_time"] + gap["duration_sec"] - right["start_time"]) < 0.002
        ):
            gap["chord"] = left["chord"]
            gap["confidence"] = min(left["confidence"], right["confidence"])
            gap["display_smoothed"] = True
    merged = []
    for item in result:
        if (
            merged
            and item.get("event_boundary") != "reattack"
            and merged[-1]["chord"] == item["chord"]
            and abs(
                merged[-1]["start_time"]
                + merged[-1]["duration_sec"]
                - item["start_time"]
            )
            < 0.002
        ):
            previous = merged[-1]
            duration = previous["duration_sec"] + item["duration_sec"]
            previous["confidence"] = (
                previous["confidence"] * previous["duration_sec"]
                + item["confidence"] * item["duration_sec"]
            ) / duration
            previous["duration_sec"] = duration
        else:
            merged.append(dict(item))
    return [
        item for index, item in enumerate(merged)
        if not (index in {0, len(merged) - 1} and item["chord"] == "N"
                and item["duration_sec"] < 0.1)
    ]
