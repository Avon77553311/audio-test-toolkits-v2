"""Validation shared by chord-analysis HTTP and worker entry points."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping


_NUMBER_FIELDS = {
    "time_window_ms": ("时间窗口", 100.0, 5000.0),
    "min_frequency": ("最低频率", 20.0, 11025.0),
    "max_frequency": ("最高频率", 40.0, 11025.0),
    "confidence_threshold": ("置信度阈值", 0.0, 1.0),
}

_CHOICES = {
    "analysis_mode": {"harmony", "bass", "mix"},
    "quality_mode": {"balanced", "high"},
}

_CHOICE_MESSAGES = {
    "analysis_mode": "analysis_mode 仅支持 harmony、bass 或 mix",
    "quality_mode": "quality_mode 仅支持 balanced 或 high",
}

_INTEGER_PATTERN = re.compile(r"[+-]?\d+")


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


def _integer(value, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label}必须是整数")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and _INTEGER_PATTERN.fullmatch(value.strip()):
        number = int(value)
    else:
        raise ValueError(f"{label}必须是整数")
    if not minimum <= number <= maximum:
        raise ValueError(f"{label}必须在 {minimum}~{maximum} 之间")
    return number


def validate_chord_analysis_options(values: Mapping) -> dict:
    """Return normalized finite chord-analysis options or raise ``ValueError``."""
    if not isinstance(values, Mapping):
        raise ValueError("和弦分析参数必须是对象")

    options = {}
    for key, choices in _CHOICES.items():
        raw = values.get(key)
        if raw in (None, ""):
            continue
        if not isinstance(raw, str) or raw not in choices:
            raise ValueError(_CHOICE_MESSAGES[key])
        options[key] = raw

    analysis_mode = options.get("analysis_mode", "harmony")
    for key, (label, minimum, maximum) in _NUMBER_FIELDS.items():
        raw = values.get(key)
        if raw not in (None, ""):
            options[key] = _finite_number(raw, label, minimum, maximum)

    raw_duration = values.get("min_chord_duration_ms")
    if raw_duration not in (None, ""):
        minimum = 500.0 if analysis_mode == "harmony" else 50.0
        options["min_chord_duration_ms"] = _finite_number(
            raw_duration, "最短分段时长", minimum, 10000.0
        )

    raw_count = values.get("expected_chord_count")
    if raw_count not in (None, ""):
        options["expected_chord_count"] = _integer(
            raw_count, "预计和弦数量", 0, 128
        )

    minimum_frequency = options.get("min_frequency", 32.70319566257483)
    maximum_frequency = options.get("max_frequency", 2093.0)
    if minimum_frequency >= maximum_frequency:
        raise ValueError("音域必须满足最低频率小于最高频率")

    for key in ("experimental", "debug"):
        raw = values.get(key)
        if raw is not None:
            if not isinstance(raw, bool):
                raise ValueError(f"{key} 必须是布尔值")
            options[key] = raw

    return options
