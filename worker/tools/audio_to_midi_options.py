"""Validation helpers shared by audio-to-MIDI HTTP and worker entry points."""

from __future__ import annotations

import math
from collections.abc import Mapping


_NUMBER_FIELDS = {
    "time_window_ms": ("帧间隔", 1.0, 500.0),
    "activation_level": ("激活阈值", 0.0, 1.0),
    "fmin": ("最低频率", 8.18, 12543.85),
    "fmax": ("最高频率", 8.18, 12543.85),
    "min_note_duration_ms": ("最短音符时长", 1.0, 60000.0),
    "bpm": ("BPM", 1.0, 999.0),
    "target_sample_rate": ("目标采样率", 1000.0, 192000.0),
    "onset_semitone": ("音符边界阈值", 0.01, 24.0),
}

_ALIASES = {
    "time_window_ms": "hop_ms",
    "activation_level": "periodicity_threshold",
}

_CHOICES = {
    "transcription_profile": {
        "balanced",
        "vocal_melisma",
        "fast_guitar",
    },
    "transcription_mode": {"mono", "poly"},
    "quality_mode": {"balanced", "high"},
}

_CHOICE_MESSAGES = {
    "transcription_profile": (
        "转录模式仅支持 balanced、vocal_melisma 或 fast_guitar"
    ),
    "transcription_mode": "识别模式仅支持 mono 或 poly",
    "quality_mode": "quality_mode 仅支持 balanced 或 high",
}


def _raw_value(values: Mapping, key: str):
    value = values.get(key)
    if value not in (None, ""):
        return value
    alias = _ALIASES.get(key)
    if alias:
        value = values.get(alias)
        if value not in (None, ""):
            return value
    return None


def _finite_number(value, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label}必须是数字")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是数字") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label}必须是有限数")
    if not minimum <= number <= maximum:
        raise ValueError(f"{label}必须在 {minimum:g}~{maximum:g} 之间")
    return number


def validate_audio_to_midi_options(values: Mapping) -> dict:
    """Return normalized, finite audio-to-MIDI options or raise ``ValueError``."""
    if not isinstance(values, Mapping):
        raise ValueError("音频转 MIDI 参数必须是对象")

    options = {}
    for key, (label, minimum, maximum) in _NUMBER_FIELDS.items():
        raw = _raw_value(values, key)
        if raw is not None:
            options[key] = _finite_number(raw, label, minimum, maximum)

    for key, choices in _CHOICES.items():
        raw = values.get(key)
        if raw in (None, ""):
            continue
        if not isinstance(raw, str) or raw not in choices:
            raise ValueError(_CHOICE_MESSAGES[key])
        options[key] = raw

    fmin = options.get("fmin", 50.0)
    fmax = options.get("fmax", 1000.0)
    if fmin >= fmax:
        raise ValueError("音域必须满足最低频率小于最高频率")

    return options
