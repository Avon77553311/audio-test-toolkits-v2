"""长音频和弦分析的低内存分段、时间轴拼接与边界消重。"""

import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from tools.runtime_resources import (
    executable_path,
    run_cancellable,
    runtime_directory,
)

DEFAULT_SEGMENT_DURATION_SEC = 180.0
DEFAULT_OVERLAP_SEC = 2.0
_TIME_START_KEYS = ("start_time", "start", "onset_sec")
_TIME_END_KEYS = ("end_time", "end", "offset_sec")
_TIMELINE_KEYS = (
    "chord_progression",
    "chord_events",
    "harmonic_regions",
    "raw_chord_progression",
    "chords",
    "notes",
    "segments",
)


@dataclass(frozen=True)
class AudioSegment:
    index: int
    start: float
    duration: float
    keep_start: float
    keep_end: float


def probe_audio_duration(input_path, timeout=30, cancel_event=None):
    """用 ffprobe 获取容器时长，拒绝缺失、非有限或非正值。"""
    try:
        ffprobe = executable_path("ffprobe")
    except RuntimeError as exc:
        raise RuntimeError("离线媒体组件 ffprobe 不可用") from exc
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        os.fspath(input_path),
    ]
    try:
        completed = (
            run_cancellable(
                command,
                timeout=timeout,
                cancel_event=cancel_event,
                check=True,
                cancel_message="和弦分析任务已取消",
            )
            if cancel_event is not None
            else subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
        )
    except FileNotFoundError as exc:
        raise RuntimeError("离线媒体组件 ffprobe 不可用") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffprobe 读取音频时长超时") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()[-500:]
        raise RuntimeError(f"ffprobe 读取音频时长失败: {detail or '未知错误'}") from exc

    try:
        duration = float(json.loads(completed.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("ffprobe 未返回有效音频时长") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("ffprobe 未返回有效音频时长")
    return duration


def build_segment_plan(
    duration, segment_duration=DEFAULT_SEGMENT_DURATION_SEC, overlap=DEFAULT_OVERLAP_SEC
):
    """创建带少量重叠的窗口，并以重叠中点划分各窗口的保留区。"""
    duration = float(duration)
    segment_duration = float(segment_duration)
    overlap = float(overlap)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("音频时长必须为正数")
    if not math.isfinite(segment_duration) or segment_duration <= 0:
        raise ValueError("分段时长必须为正数")
    if not math.isfinite(overlap) or overlap < 0 or overlap >= segment_duration:
        raise ValueError("重叠时长必须不小于 0 且小于分段时长")
    if duration <= segment_duration:
        return [AudioSegment(0, 0.0, duration, 0.0, duration)]

    stride = segment_duration - overlap
    starts = []
    start = 0.0
    while start < duration:
        starts.append(start)
        if start + segment_duration >= duration:
            break
        start += stride

    windows = [(start, min(duration, start + segment_duration)) for start in starts]
    boundaries = [0.0]
    for (_, previous_end), (next_start, _) in zip(windows[:-1], windows[1:]):
        boundaries.append((previous_end + next_start) / 2.0)
    boundaries.append(duration)
    return [
        AudioSegment(
            index, start, end - start, boundaries[index], boundaries[index + 1]
        )
        for index, (start, end) in enumerate(windows)
    ]


def extract_pcm_segment(
    input_path,
    output_path,
    start,
    duration,
    timeout=180,
    cancel_event=None,
):
    """通过 ffmpeg 将一个窗口解码为无压缩 PCM WAV。"""
    try:
        ffmpeg = executable_path("ffmpeg")
    except RuntimeError as exc:
        raise RuntimeError("离线媒体组件 ffmpeg 不可用") from exc
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        os.fspath(input_path),
        "-map",
        "0:a:0",
        "-vn",
        "-c:a",
        "pcm_f32le",
        os.fspath(output_path),
    ]
    try:
        (
            run_cancellable(
                command,
                timeout=timeout,
                cancel_event=cancel_event,
                check=True,
                cancel_message="和弦分析任务已取消",
            )
            if cancel_event is not None
            else subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
        )
    except FileNotFoundError as exc:
        raise RuntimeError("离线媒体组件 ffmpeg 不可用") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffmpeg 裁剪音频分段超时") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()[-500:]
        raise RuntimeError(f"ffmpeg 裁剪音频分段失败: {detail or '未知错误'}") from exc


def _number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _bounds(item):
    start_key = next((key for key in _TIME_START_KEYS if _number(item.get(key))), None)
    if start_key is None:
        return None
    start = float(item[start_key])
    end_key = next((key for key in _TIME_END_KEYS if _number(item.get(key))), None)
    if end_key is not None:
        end = float(item[end_key])
    elif _number(item.get("duration_sec")):
        end = start + float(item["duration_sec"])
    else:
        end = start
    return start_key, start, end_key, max(start, end)


def _offset_and_clip(item, offset, keep_start, keep_end):
    copied = dict(item)
    bounds = _bounds(copied)
    if bounds is None:
        return copied
    start_key, start, end_key, end = bounds
    start += offset
    end += offset
    if end < keep_start or start > keep_end:
        return None
    clipped_start = max(start, keep_start)
    clipped_end = min(end, keep_end)
    if end > start and clipped_end <= clipped_start:
        return None
    copied[start_key] = round(clipped_start, 6)
    if start < keep_start:
        copied["_continued_from_previous_window"] = True
    if end > keep_end:
        copied["_continues_to_next_window"] = True
    if end_key is not None:
        copied[end_key] = round(clipped_end, 6)
    if _number(copied.get("duration_sec")):
        copied["duration_sec"] = round(max(0.0, clipped_end - clipped_start), 6)
    return copied


def _identity(item):
    for key in ("chord", "label", "note", "midi", "pitch", "text"):
        if key in item:
            return key, item[key]
    return None


def _merge_timeline(items, allow_touch=True):
    """按时间排序，消除 overlap 重复，并拼接连续同标签片段。"""
    ordered = sorted(
        items, key=lambda item: (_bounds(item) or ("", float("inf"), None, 0))[1]
    )
    merged = []
    tolerance = 0.02 if allow_touch else -1e-9
    for item in ordered:
        current = dict(item)
        current_bounds = _bounds(current)
        if not merged or current_bounds is None:
            merged.append(current)
            continue
        previous = merged[-1]
        previous_bounds = _bounds(previous)
        if previous_bounds is None or _identity(previous) != _identity(current):
            merged.append(current)
            continue
        is_window_continuation = False
        if (
            not allow_touch
            or previous.get("event_boundary")
            or current.get("event_boundary")
        ):
            is_window_continuation = bool(
                previous.get("_continues_to_next_window")
                and current.get("_continued_from_previous_window")
            )
            if not is_window_continuation:
                merged.append(current)
                continue
        _, previous_start, previous_end_key, previous_end = previous_bounds
        current_start_key, _, _, current_end = current_bounds
        merge_tolerance = 0.02 if is_window_continuation else tolerance
        if float(current[current_start_key]) > previous_end + merge_tolerance:
            merged.append(current)
            continue

        new_end = max(previous_end, current_end)
        previous_duration = max(0.0, previous_end - previous_start)
        added_duration = max(0.0, new_end - previous_end)
        if previous_end_key is not None:
            previous[previous_end_key] = round(new_end, 6)
        if _number(previous.get("duration_sec")):
            previous["duration_sec"] = round(max(0.0, new_end - previous_start), 6)
        if _number(current.get("confidence")):
            previous["confidence"] = round(
                (
                    float(previous.get("confidence", 0.0)) * previous_duration
                    + float(current["confidence"]) * added_duration
                ) / max(previous_duration + added_duration, 1e-9),
                3,
            )
        previous.pop("_continues_to_next_window", None)
        current.pop("_continued_from_previous_window", None)
    for item in merged:
        item.pop("_continued_from_previous_window", None)
        item.pop("_continues_to_next_window", None)
    return merged


def _stats_for_progression(base_stats, progression, duration):
    stats = dict(base_stats or {})
    counts = {}
    confidences = []
    for item in progression:
        chord = item.get("chord")
        if chord is not None:
            counts[chord] = counts.get(chord, 0) + 1
        if _number(item.get("confidence")):
            confidences.append(float(item["confidence"]))
    stats.update(
        {
            "duration_sec": round(duration, 2),
            "total_segments": len(progression),
            "unique_chords": len(counts),
            "most_common_chords": [
                {"chord": chord, "count": count}
                for chord, count in sorted(
                    counts.items(), key=lambda pair: pair[1], reverse=True
                )[:5]
            ],
            "avg_confidence": round(sum(confidences) / len(confidences), 3)
            if confidences
            else 0.0,
        }
    )
    active = [item for item in progression if item.get("chord") not in {None, "", "N", "N/A"}]
    active_duration = sum(float(item.get("duration_sec", 0)) for item in active)
    stats["chord_duration_coverage"] = round(active_duration / max(duration, 1e-9), 5)
    stats["duration_weighted_chord_confidence"] = round(
        sum(float(item.get("confidence", 0)) * float(item.get("duration_sec", 0))
            for item in active) / max(active_duration, 1e-9), 5
    )
    return stats


def _check_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("和弦分析任务已取消")


def _aggregate_keys(candidates):
    if not candidates:
        return None
    grouped = {}
    for key_data, weight in candidates:
        identity = (
            str(key_data.get("key") or "Unknown"),
            str(key_data.get("scale") or "Unknown"),
        )
        entry = grouped.setdefault(
            identity,
            {
                "weight": 0.0,
                "confidence": 0.0,
                "value": dict(key_data),
            },
        )
        entry["weight"] += weight
        if _number(key_data.get("confidence")):
            entry["confidence"] += float(key_data["confidence"]) * weight
    selected = max(
        grouped.values(),
        key=lambda entry: (entry["weight"], entry["confidence"]),
    )
    value = selected["value"]
    if selected["weight"] > 0:
        value["confidence"] = round(
            selected["confidence"] / selected["weight"],
            3,
        )
    return value


def analyze_with_segmentation(
    input_path,
    analyze_segment,
    cancel_event=None,
    segment_duration=DEFAULT_SEGMENT_DURATION_SEC,
    overlap=DEFAULT_OVERLAP_SEC,
    context=None,
):
    """短音频走原分析路径；长音频逐段解码、分析、偏移并合并。"""
    duration = (
        probe_audio_duration(input_path, cancel_event=cancel_event)
        if cancel_event is not None
        else probe_audio_duration(input_path)
    )
    plan = build_segment_plan(duration, segment_duration, overlap)
    segmented = len(plan) > 1

    if not segmented:
        _check_cancelled(cancel_event)
        result = analyze_segment(input_path, cancel_event=cancel_event)
        result = dict(result)
    else:
        collected = {key: [] for key in _TIMELINE_KEYS}
        present_timeline_keys = set()
        result = None
        beat_count = 0.0
        saw_beat_count = False
        tempos = []
        keys = []
        diagnostics = []
        temp_parent = runtime_directory()
        with tempfile.TemporaryDirectory(
            prefix="chord-segments-", dir=os.fspath(temp_parent)
        ) as temp_dir:
            for spec in plan:
                _check_cancelled(cancel_event)
                if context is not None:
                    context.set_segment(f"segment-{spec.index:04d}")
                segment_status = "failed"
                try:
                    segment_path = Path(temp_dir) / f"segment-{spec.index:04d}.wav"
                    if cancel_event is None:
                        extract_pcm_segment(
                            input_path,
                            segment_path,
                            spec.start,
                            spec.duration,
                        )
                    else:
                        extract_pcm_segment(
                            input_path,
                            segment_path,
                            spec.start,
                            spec.duration,
                            cancel_event=cancel_event,
                        )
                    _check_cancelled(cancel_event)
                    chunk = dict(
                        analyze_segment(
                            os.fspath(segment_path), cancel_event=cancel_event
                        )
                    )
                    if result is None:
                        result = dict(chunk)
                        for timeline_key in _TIMELINE_KEYS:
                            result.pop(timeline_key, None)
                    keep_duration = max(0.0, spec.keep_end - spec.keep_start)
                    diagnostics.append({
                        "start": spec.keep_start,
                        "end": spec.keep_end,
                        "recognition_status": chunk.get("recognition_status"),
                        "neural": dict(chunk.get("neural") or {}),
                        "ensemble": dict(chunk.get("ensemble") or {}),
                    })
                    if _number(chunk.get("beat_count")):
                        saw_beat_count = True
                        beat_count += (
                            float(chunk["beat_count"])
                            * keep_duration
                            / max(spec.duration, 1e-9)
                        )
                    if _number(chunk.get("tempo_bpm")):
                        tempos.append((float(chunk["tempo_bpm"]), keep_duration))
                    if isinstance(chunk.get("key"), dict):
                        keys.append((chunk["key"], keep_duration))
                    for key in _TIMELINE_KEYS:
                        values = chunk.get(key)
                        if not isinstance(values, list):
                            continue
                        present_timeline_keys.add(key)
                        for value in values:
                            if isinstance(value, dict):
                                shifted = _offset_and_clip(
                                    value, spec.start, spec.keep_start, spec.keep_end
                                )
                                if shifted is not None:
                                    collected[key].append(shifted)
                    # 在下一段开始前释放底层结果及临时 PCM，保持峰值内存受单段约束。
                    segment_path.unlink(missing_ok=True)
                    del chunk
                    segment_status = "completed"
                finally:
                    if context is not None:
                        context.finish_segment(segment_status)
                        context.set_segment(None)

        result = result or {}
        for key in present_timeline_keys:
            result[key] = (
                collected[key] if key == "raw_chord_progression"
                else _merge_timeline(
                    collected[key],
                    allow_touch=key not in {"chord_events", "notes"},
                )
            )
        result["segment_diagnostics"] = diagnostics
        if any(item["recognition_status"] for item in diagnostics):
            has_chords = any(
                item.get("chord") not in {"", "N", "N/A", None}
                for item in result.get("chord_progression", [])
            )
            result["recognition_status"] = (
                "recognized" if has_chords
                else "insufficient_harmony"
                if all(item["recognition_status"] == "insufficient_harmony" for item in diagnostics)
                else "no_confident_chords"
            )
        if result.get("neural"):
            # Do not present the first window's decision as a whole-file gate.
            result["neural"] = {
                "model": result["neural"].get("model"),
                "quality_gate_passed": all(
                    item["neural"].get("quality_gate_passed", False) for item in diagnostics
                ),
                "quality_gate_reason": "per_segment_decisions",
                "diagnostics_location": "segment_diagnostics",
            }
        if any(item["ensemble"] for item in diagnostics):
            devices = {item["ensemble"].get("device", "cpu") for item in diagnostics}
            result["ensemble"] = {
                "quality_mode": (result.get("ensemble") or {}).get("quality_mode", "balanced"),
                "chosen": "per_segment",
                "device": next(iter(devices)) if len(devices) == 1 else "mixed",
                "fallback": any(item["ensemble"].get("fallback") for item in diagnostics),
                "mps_fallback": any(item["ensemble"].get("mps_fallback") for item in diagnostics),
                "execution": "local",
                "diagnostics_location": "segment_diagnostics",
            }
        if result.get("neural"):
            result["neural"]["weak_same_root_fragments_absorbed"] = sum(
                int(
                    item["neural"].get(
                        "weak_same_root_fragments_absorbed",
                        0,
                    )
                )
                for item in diagnostics
            )
            result["neural"]["passing_chords_restored"] = sum(
                int(item["neural"].get("passing_chords_restored", 0))
                for item in diagnostics
            )
        if saw_beat_count:
            result["beat_count"] = int(round(beat_count))
        if tempos:
            total_weight = sum(weight for _tempo, weight in tempos)
            result["tempo_bpm"] = round(
                sum(tempo * weight for tempo, weight in tempos)
                / total_weight,
                2,
            )
        aggregated_key = _aggregate_keys(keys)
        if aggregated_key is not None:
            result["key"] = aggregated_key

    progression = result.get("chord_progression")
    if isinstance(progression, list):
        result["stats"] = _stats_for_progression(
            result.get("stats"), progression, duration
        )
    else:
        stats = dict(result.get("stats") or {})
        stats["duration_sec"] = round(duration, 2)
        result["stats"] = stats
    chord_events = result.get("chord_events")
    if isinstance(chord_events, list):
        result["stats"]["event_count"] = len(chord_events)
        result["stats"]["reattack_count"] = sum(
            item.get("event_boundary") == "reattack"
            for item in chord_events
        )
    harmonic_regions = result.get("harmonic_regions")
    if isinstance(harmonic_regions, list):
        result["stats"]["harmonic_region_count"] = len(harmonic_regions)

    metadata = {
        "segmented": segmented,
        "segment_count": len(plan),
        "segment_duration_sec": float(segment_duration),
    }
    result.update(metadata)
    result["stats"].update(metadata)
    return result
