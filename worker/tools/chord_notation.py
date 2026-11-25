"""Convert stable chord regions into a measure-oriented notation timeline."""

from __future__ import annotations

import math

import numpy as np

SAMPLE_RATE = 22050
HOP_LENGTH = 512
BEATS_PER_MEASURE = 4
MIN_MEASURE_SHARE = 0.22


def _check_cancelled(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("和弦分析任务已取消")


def select_tempo_divisor(
    harmonic_regions: list[dict],
    raw_tempo_bpm: float,
) -> tuple[int, dict]:
    """Choose a half-time notation grid only with strong harmonic evidence."""
    raw_tempo_bpm = float(raw_tempo_bpm)
    beat_period = 60.0 / max(raw_tempo_bpm, 1e-9)
    durations_in_beats = np.asarray(
        [
            float(item.get("duration_sec", 0.0)) / beat_period
            for item in harmonic_regions
            if item.get("chord") not in {"", "N", "N/A", None}
            and float(item.get("duration_sec", 0.0)) > 0
        ],
        dtype=np.float32,
    )
    median_beats = (
        float(np.median(durations_in_beats))
        if durations_in_beats.size
        else 0.0
    )
    eight_beat_share = (
        float(np.mean(
            (durations_in_beats >= 6.5)
            & (durations_in_beats <= 9.5)
        ))
        if durations_in_beats.size
        else 0.0
    )
    divisor = (
        2
        if raw_tempo_bpm > 130.0
        and median_beats >= 6.5
        and eight_beat_share >= 0.35
        else 1
    )
    return divisor, {
        "median_harmonic_duration_raw_beats": round(median_beats, 3),
        "eight_beat_region_share": round(eight_beat_share, 3),
        "tempo_octave_corrected": divisor == 2,
    }


def _measure_phase(
    beat_frames,
    onset,
    harmonic_regions,
    frame_duration: float,
) -> tuple[int, list[float]]:
    region_frames = [
        int(round(float(item.get("start_time", 0.0)) / frame_duration))
        for item in harmonic_regions[1:]
    ]
    onset = np.asarray(onset, dtype=np.float32)
    onset_scale = (
        float(np.percentile(onset, 90))
        if onset.size
        else 0.0
    )
    scores = []
    for phase in range(BEATS_PER_MEASURE):
        boundaries = beat_frames[phase::BEATS_PER_MEASURE]
        score = 0.0
        for frame in boundaries:
            frame = int(frame)
            local = onset[max(0, frame - 2):min(len(onset), frame + 3)]
            if local.size:
                score += float(local.max())
            score += onset_scale * 0.2 * sum(
                max(0.0, 1.0 - abs(region_frame - frame) / 8.0)
                for region_frame in region_frames
                if abs(region_frame - frame) < 8
            )
        scores.append(score / max(1, len(boundaries)))
    return int(np.argmax(scores)), [round(value, 5) for value in scores]


def _measure_boundaries(
    beat_frames,
    phase: int,
    frame_count: int,
) -> np.ndarray:
    boundaries = np.asarray(
        beat_frames[phase::BEATS_PER_MEASURE],
        dtype=np.float64,
    )
    if boundaries.size < 2:
        return np.asarray([0.0, float(frame_count)])
    measure_frames = float(np.median(np.diff(boundaries)))
    values = boundaries.tolist()
    while values[0] > 0:
        values.insert(0, values[0] - measure_frames)
    while values[-1] < frame_count:
        values.append(values[-1] + measure_frames)
    result = np.asarray(values, dtype=np.float64)
    result = result[
        (result >= -0.25 * measure_frames)
        & (result <= frame_count + 0.25 * measure_frames)
    ]
    if result[0] > 0:
        result = np.r_[0.0, result]
    else:
        result[0] = 0.0
    if result[-1] < frame_count:
        result = np.r_[result, float(frame_count)]
    else:
        result[-1] = float(frame_count)
    return np.unique(result)


def build_notation_from_beats(
    harmonic_regions: list[dict],
    onset,
    beat_frames,
    *,
    raw_tempo_bpm: float,
    frame_duration: float,
    duration: float,
) -> dict:
    """Build one notation event per measure plus supported within-bar changes."""
    beat_frames = np.asarray(beat_frames, dtype=int)
    onset = np.asarray(onset, dtype=np.float32)
    duration = max(0.0, float(duration))
    divisor, tempo_diagnostics = select_tempo_divisor(
        harmonic_regions,
        raw_tempo_bpm,
    )
    notation_beats = beat_frames[::divisor]
    notation_tempo = float(raw_tempo_bpm) / divisor
    if notation_beats.size < BEATS_PER_MEASURE + 1:
        return {
            "notation_events": [],
            "measures": [],
            "notation": {
                "status": "insufficient_beats",
                "meter": "4/4",
                "raw_tempo_bpm": round(float(raw_tempo_bpm), 2),
                "tempo_bpm": round(notation_tempo, 2),
                "tempo_divisor": divisor,
                "beat_count": int(len(notation_beats)),
                "measure_count": 0,
                "event_count": 0,
                **tempo_diagnostics,
            },
        }

    phase, phase_scores = _measure_phase(
        notation_beats,
        onset,
        harmonic_regions,
        frame_duration,
    )
    boundaries = _measure_boundaries(
        notation_beats,
        phase,
        len(onset),
    )
    measures = []
    notation_events = []
    for measure_number, (left, right) in enumerate(
        zip(boundaries[:-1], boundaries[1:]),
    ):
        start = max(0.0, float(left) * frame_duration)
        end = min(duration, float(right) * frame_duration)
        measure_duration = end - start
        if measure_duration <= 0.1:
            continue

        contributions = []
        for region in harmonic_regions:
            region_start = float(region.get("start_time", 0.0))
            region_end = region_start + float(region.get("duration_sec", 0.0))
            overlap_start = max(start, region_start)
            overlap_end = min(end, region_end)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > 0:
                contributions.append(
                    (overlap_start, overlap_end, region, overlap)
                )
        minimum_overlap = max(
            MIN_MEASURE_SHARE * measure_duration,
            0.12,
        )
        eligible = [
            item for item in contributions
            if item[3] >= minimum_overlap
        ]
        if not eligible and contributions:
            eligible = [max(contributions, key=lambda item: item[3])]

        merged = []
        has_chord = any(
            item[2].get("chord") not in {"", "N", "N/A", None}
            for item in eligible
        )
        for event_start, event_end, region, overlap in eligible:
            if has_chord and region.get("chord") in {"", "N", "N/A", None}:
                continue
            if merged and merged[-1][2].get("chord") == region.get("chord"):
                previous = merged[-1]
                merged[-1] = (
                    previous[0],
                    event_end,
                    region,
                    previous[3] + overlap,
                )
            else:
                merged.append((event_start, event_end, region, overlap))

        measure_labels = []
        for position, (event_start, event_end, region, overlap) in enumerate(
            merged,
        ):
            label = str(region.get("chord") or "N")
            measure_labels.append(label)
            item = dict(region)
            item.update(
                {
                    "chord": label,
                    "start_time": round(event_start, 6),
                    "duration_sec": round(event_end - event_start, 6),
                    "measure_number": measure_number,
                    "beat_offset": round(
                        (event_start - start)
                        / max(measure_duration, 1e-9)
                        * BEATS_PER_MEASURE,
                        3,
                    ),
                    "measure_support": round(
                        overlap / max(measure_duration, 1e-9),
                        3,
                    ),
                    "event_boundary": (
                        "measure_start" if position == 0
                        else "within_measure"
                    ),
                }
            )
            notation_events.append(item)
        measures.append(
            {
                "measure_number": measure_number,
                "start_time": round(start, 6),
                "duration_sec": round(measure_duration, 6),
                "chords": measure_labels or ["N"],
            }
        )

    return {
        "notation_events": notation_events,
        "measures": measures,
        "notation": {
            "status": "recognized",
            "meter": "4/4",
            "meter_assumption": True,
            "raw_tempo_bpm": round(float(raw_tempo_bpm), 2),
            "tempo_bpm": round(notation_tempo, 2),
            "tempo_divisor": divisor,
            "raw_beat_count": int(len(beat_frames)),
            "beat_count": int(len(notation_beats)),
            "measure_count": len(measures),
            "event_count": len(notation_events),
            "measure_phase": phase,
            "measure_phase_scores": phase_scores,
            "minimum_measure_share": MIN_MEASURE_SHARE,
            **tempo_diagnostics,
        },
    }


def analyze_notation_timeline(
    input_path,
    harmonic_regions: list[dict],
    *,
    cancel_event=None,
) -> dict:
    """Analyze a full audio file after segmented chord inference has completed."""
    import librosa

    _check_cancelled(cancel_event)
    audio, sample_rate = librosa.load(
        input_path,
        sr=SAMPLE_RATE,
        mono=True,
    )
    if audio.size == 0:
        raise ValueError("音频为空")
    onset = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
        aggregate=np.median,
    )
    raw_tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
        trim=False,
    )
    raw_tempo_value = (
        float(np.asarray(raw_tempo).reshape(-1)[0])
        if np.size(raw_tempo)
        else 0.0
    )
    if not math.isfinite(raw_tempo_value) or raw_tempo_value <= 0:
        raw_tempo_value = 120.0
    _check_cancelled(cancel_event)
    return build_notation_from_beats(
        harmonic_regions,
        onset,
        beat_frames,
        raw_tempo_bpm=raw_tempo_value,
        frame_duration=HOP_LENGTH / sample_rate,
        duration=len(audio) / sample_rate,
    )
