"""RunPod client for the structured neural chord analyzer."""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests

from audio_admission import probe_audio_duration
from tools.config import (
    RUNPOD_API_KEY,
    RUNPOD_MAX_AUDIO_BYTES,
    RUNPOD_MAX_AUDIO_SECONDS,
    RUNPOD_MAX_POLL,
    RUNPOD_POLL_INTERVAL,
    RUNPOD_STEM_API_URL,
    RUNPOD_STEM_ENABLED,
    RUNPOD_STEM_PROTOCOL,
    RUNPOD_TIMEOUT,
    STEM_CALLBACK_PUBLIC_BASE_URL,
    STEM_CALLBACK_TTL_SECONDS,
)
from tools.stem_callback import stem_callback_store


class CloudChordClient:
    def __init__(self):
        if not STEM_CALLBACK_PUBLIC_BASE_URL:
            from capabilities import capabilities
            if not capabilities()['features']['cloud_chords']:
                raise ValueError('云端神经和弦未启用')
            return
        if not RUNPOD_STEM_ENABLED or RUNPOD_STEM_PROTOCOL != "demucs-v1":
            raise ValueError("RunPod 音频推理未启用")
        if not RUNPOD_STEM_API_URL or not RUNPOD_API_KEY:
            raise ValueError("RunPod Endpoint 或 API Key 未配置")

    def analyze(
        self,
        source,
        options,
        *,
        cancel_event=None,
        progress=None,
    ):
        if not STEM_CALLBACK_PUBLIC_BASE_URL:
            from cloud_inline import analyze_chords
            return analyze_chords(source, options, progress)
        source = Path(source).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError("和弦分析源音频不存在")
        if source.stat().st_size > RUNPOD_MAX_AUDIO_BYTES:
            raise ValueError("和弦分析音频超过云端大小上限")
        duration = probe_audio_duration(os.fspath(source))
        if duration <= 0 or duration > RUNPOD_MAX_AUDIO_SECONDS:
            raise ValueError("和弦分析音频超过云端时长上限")

        configured_root = stem_callback_store._output_root
        if configured_root is None or not source.is_relative_to(configured_root):
            raise ValueError("和弦分析输入不在受控输出目录内")
        callback_root = configured_root
        callback_id, _callback_url, callback = stem_callback_store.register(
            callback_root,
            STEM_CALLBACK_PUBLIC_BASE_URL,
            STEM_CALLBACK_TTL_SECONDS,
            source_path=source,
        )
        input_url = stem_callback_store.input_url(
            callback_id,
            STEM_CALLBACK_PUBLIC_BASE_URL,
            callback,
        )
        headers = {
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": {
                "task": "chord_analysis",
                "protocol": "structured-chord-v1",
                "filename": source.name,
                "audio_url": input_url,
                "analysis_mode": options.get("analysis_mode", "mix"),
                "confidence_threshold": options.get("confidence_threshold", 0.18),
                "min_chord_duration_ms": options.get(
                    "min_chord_duration_ms",
                    500,
                ),
                "enhance_with_stems": True,
            }
        }
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("和弦分析任务已取消")
            if progress:
                progress(12, "正在提交神经和弦分析")
            from task_scheduler import ensure_not_cancelled

            ensure_not_cancelled()
            response = requests.post(
                f"{RUNPOD_STEM_API_URL}/run",
                headers=headers,
                json=payload,
                timeout=RUNPOD_TIMEOUT,
            )
            response.raise_for_status()
            job_id = response.json().get("id")
            if not job_id:
                raise RuntimeError("RunPod 和弦任务响应缺少 id")
            from task_scheduler import notify_runpod_started

            notify_runpod_started(job_id)
            for attempt in range(RUNPOD_MAX_POLL):
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("和弦分析任务已取消")
                time.sleep(RUNPOD_POLL_INTERVAL)
                status_response = requests.get(
                    f"{RUNPOD_STEM_API_URL}/status/{job_id}",
                    headers=headers,
                    timeout=RUNPOD_TIMEOUT,
                )
                status_response.raise_for_status()
                data = status_response.json()
                status = data.get("status")
                if status == "COMPLETED":
                    output = data.get("output") or {}
                    if (
                        output.get("protocol") != "structured-chord-v1"
                        or not isinstance(output.get("metadata"), dict)
                    ):
                        raise RuntimeError("RunPod 和弦分析结果格式无效")
                    if progress:
                        progress(90, "神经和弦分析完成")
                    return output["metadata"]
                if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
                    raise RuntimeError(
                        f"RunPod 和弦分析失败: {data.get('error') or status}"
                    )
                if progress:
                    span = max(1, RUNPOD_MAX_POLL - 1)
                    progress(
                        16 + int(70 * min(attempt, span) / span),
                        "RunPod 正在分析和弦",
                    )
            raise TimeoutError("RunPod 和弦分析超时")
        finally:
            stem_callback_store.discard(callback_id)
