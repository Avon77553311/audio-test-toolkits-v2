"""共享的安全无损音频预压缩。"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


class AudioPrecompressionError(ValueError):
    """输入无法安全转换为分析用 FLAC。"""


def precompress_audio(source: str, *, sample_rate: int | None, channels: int | None,
                      only_if_smaller: bool = False, remove_source: bool = True,
                      timeout: int = 300) -> str:
    """在源文件同目录生成 FLAC，并在完整成功后原子发布。

    分析输入总是采用指定采样率/声道；成品处理可用 ``only_if_smaller``，
    此时 FLAC 没有实际节省空间就保留原文件。失败不会改动源文件。
    """
    source_path = Path(source)
    if only_if_smaller:
        import soundfile as sf

        try:
            subtype = sf.info(source_path).subtype
        except (RuntimeError, OSError):
            # 无法证明整数 FLAC 能无损表示源文件时，保留原文件。
            return os.fspath(source_path)
        if subtype not in {"PCM_16", "PCM_24", "PCM_S8", "PCM_U8"}:
            return os.fspath(source_path)
    directory = source_path.parent
    fd, temporary = tempfile.mkstemp(
        prefix=f".{source_path.stem}.precompress-", suffix=".part", dir=directory)
    os.close(fd)
    temporary_path = Path(temporary)
    target_path = temporary_path.with_suffix(".flac")
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", os.fspath(source_path), "-map", "0:a:0", "-vn",
    ]
    if channels is not None:
        command.extend(["-ac", str(channels)])
    if sample_rate is not None:
        command.extend(["-ar", str(sample_rate)])
    command.extend(["-c:a", "flac", "-f", "flac", os.fspath(temporary_path)])
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False)
        if completed.returncode or not temporary_path.is_file():
            raise AudioPrecompressionError("音频无损预压缩失败，请确认文件有效")
        if only_if_smaller and temporary_path.stat().st_size >= source_path.stat().st_size:
            temporary_path.unlink(missing_ok=True)
            return os.fspath(source_path)
        os.replace(temporary_path, target_path)
        if remove_source and target_path != source_path:
            source_path.unlink()
        return os.fspath(target_path)
    except FileNotFoundError as exc:
        raise AudioPrecompressionError("FFmpeg 不可用，无法进行音频无损预压缩") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioPrecompressionError("音频无损预压缩超时") from exc
    except OSError as exc:
        raise AudioPrecompressionError("音频无损预压缩失败") from exc
    finally:
        temporary_path.unlink(missing_ok=True)
        # os.replace 前后任一异常都不能遗留发布文件。
        if target_path.exists() and source_path.exists():
            target_path.unlink(missing_ok=True)


def precompress_for_mode(source: str, mode: str) -> str:
    """按业务模式选择唯一预压缩规格；结构 GPU 在专用客户端中处理。"""
    if mode == "song_info_cpu":
        return precompress_audio(source, sample_rate=44_100, channels=1)
    if mode in {"chord_cpu", "chord_gpu"}:
        return precompress_audio(source, sample_rate=22_050, channels=1)
    if mode == "pitch_speed_cpu":
        return precompress_audio(
            source, sample_rate=None, channels=None, only_if_smaller=True)
    return source
