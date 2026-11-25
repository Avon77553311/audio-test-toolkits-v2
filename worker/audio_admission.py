"""生产音频上传的统一准入限制与轻量媒体探测。"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import BinaryIO


@dataclass(frozen=True)
class AudioLimit:
    name_zh: str
    max_mb: int
    max_seconds: int

    @property
    def max_bytes(self) -> int:
        return self.max_mb * 1024 * 1024


# 唯一限制矩阵。HTTP、持久化适配器与直接 manager/tool 入口均引用这里。
AUDIO_LIMITS = {
    "stems_cpu": AudioLimit("音轨分离（CPU）", 20, 5 * 60),
    "stems_gpu": AudioLimit("音轨分离（GPU）", 25, 10 * 60),
    "rvc_gpu": AudioLimit("RVC 音色转换（GPU）", 25, 10 * 60),
    "seed_vc_source": AudioLimit("Seed-VC 源音频", 15, 5 * 60),
    "seed_vc_reference": AudioLimit("Seed-VC 参考音频", 15, 60),
    "song_structure_gpu": AudioLimit("歌曲结构分析（GPU）", 50, 10 * 60),
    "song_info_cpu": AudioLimit("歌曲信息分析（CPU）", 100, 15 * 60),
    "chord_cpu": AudioLimit("和弦分析（CPU）", 80, 10 * 60),
    "chord_gpu": AudioLimit("和弦分析（GPU）", 50, 10 * 60),
    "midi_cpu": AudioLimit("音频转 MIDI（CPU）", 25, 10 * 60),
    "midi_gpu": AudioLimit("音频转 MIDI（GPU）", 25, 10 * 60),
    "speech_free": AudioLimit("音频转文字（免费版）", 25, 10 * 60),
    "speech_paid": AudioLimit("音频转文字（付费版）", 25, 10 * 60),
    "audio_trimmer_cpu": AudioLimit("音频裁剪（CPU）", 100, 15 * 60),
    "format_converter_cpu": AudioLimit("格式转换（CPU）", 100, 30 * 60),
    "pitch_speed_cpu": AudioLimit("升降调变速（CPU）", 100, 15 * 60),
}


class AudioAdmissionError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def limit_for(mode: str) -> AudioLimit:
    try:
        return AUDIO_LIMITS[mode]
    except KeyError as exc:
        raise ValueError(f"未知音频准入模式: {mode}") from exc


def _stream_size(stream: BinaryIO) -> int | None:
    """不读取/分配文件内容地测量可 seek 的上传流，并恢复游标。"""
    try:
        position = stream.tell()
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(position)
        return size
    except (AttributeError, OSError):
        return None


def validate_upload_size(uploaded, mode: str) -> None:
    limit = limit_for(mode)
    declared = getattr(uploaded, "content_length", None)
    size = declared if declared is not None else _stream_size(uploaded.stream)
    if size is not None and size > limit.max_bytes:
        raise AudioAdmissionError(f"文件过大，最大支持 {limit.max_mb} MB", 413)


def probe_audio_duration(path: str, timeout: int = 30) -> float:
    command = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "json", os.fspath(path)]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True,
                                   timeout=timeout)
        duration = float((json.loads(completed.stdout).get("format") or {}).get("duration"))
        if duration <= 0:
            raise ValueError
        return duration
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired,
            json.JSONDecodeError, TypeError, ValueError, OSError) as exc:
        raise AudioAdmissionError("无法读取音频时长，请确认文件有效") from exc


def validate_audio_file(path: str, mode: str) -> float:
    limit = limit_for(mode)
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise AudioAdmissionError("无法读取音频时长，请确认文件有效") from exc
    if size > limit.max_bytes:
        raise AudioAdmissionError(f"文件过大，最大支持 {limit.max_mb} MB", 413)
    duration = probe_audio_duration(path)
    if duration > limit.max_seconds:
        minutes = limit.max_seconds // 60
        raise AudioAdmissionError(f"音频过长，最大支持 {minutes} 分钟")
    return duration


def task_mode(task_type: str, values: dict, *, reference: bool = False) -> str:
    if task_type == "stems":
        return "stems_gpu" if values.get("resource_type") == "gpu" else "stems_cpu"
    if task_type == "vocal-transform":
        engine = values.get("conversion_engine", values.get("conversionEngine", "seed-vc"))
        if engine == "seed-vc":
            return "seed_vc_reference" if reference else "seed_vc_source"
        return "rvc_gpu"
    if task_type == "song-info":
        return "song_structure_gpu" if values.get("use_structure_gpu") else "song_info_cpu"
    if task_type == "chord-analysis":
        return "chord_gpu" if values.get("use_runpod") else "chord_cpu"
    if task_type == "audio-to-midi":
        return "midi_gpu" if values.get("use_runpod") else "midi_cpu"
    if task_type == "speech-to-text":
        return (
            "speech_paid"
            if values.get("transcription_mode") == "paid"
            else "speech_free"
        )
    if task_type == "audio-trimmer":
        return "audio_trimmer_cpu"
    if task_type == "format-converter":
        return "format_converter_cpu"
    if task_type == "pitch-speed":
        return "pitch_speed_cpu"
    raise ValueError(f"任务没有配置音频准入模式: {task_type}")
