"""与仓库 runpod_handler.py 的 Demucs v1 协议配套的 RunPod 客户端。"""

import base64
import io
import os
import time
import zipfile
from pathlib import Path

import requests

from audio_admission import probe_audio_duration
from tools.config import (
    RUNPOD_API_KEY,
    RUNPOD_STEM_API_URL,
    RUNPOD_STEM_ENABLED,
    RUNPOD_STEM_PROTOCOL,
    RUNPOD_TIMEOUT,
    RUNPOD_POLL_INTERVAL,
    RUNPOD_MAX_POLL,
    RUNPOD_MAX_AUDIO_BYTES,
    RUNPOD_MAX_AUDIO_SECONDS,
    STEM_CALLBACK_PUBLIC_BASE_URL,
    STEM_CALLBACK_TTL_SECONDS,
    STEM_CALLBACK_WAIT_SECONDS,
)
from tools.stem_callback import stem_callback_store


class CloudStemClient:
    def __init__(self):
        if not RUNPOD_STEM_ENABLED:
            raise ValueError(
                "云端 Stem 未启用；如确认接受 GPU 费用，请显式设置 RUNPOD_STEM_ENABLED=true"
            )
        if RUNPOD_STEM_PROTOCOL != "demucs-v1":
            raise ValueError(
                "未确认 endpoint 的 Demucs 能力；请部署本仓库 runpod_handler.py 后设置 RUNPOD_STEM_PROTOCOL=demucs-v1"
            )
        if not RUNPOD_STEM_API_URL:
            raise ValueError("RUNPOD_STEM_ENDPOINT_ID 未设置")
        if not RUNPOD_API_KEY:
            raise ValueError("RUNPOD_API_KEY 未设置；请仅通过环境变量或 Secret 配置")
        if not STEM_CALLBACK_PUBLIC_BASE_URL:
            raise ValueError(
                "STEM_CALLBACK_PUBLIC_BASE_URL 未设置，不能从不可信 Host 头构造回调 URL"
            )

    def separate(
        self,
        source,
        output_root,
        model,
        cancel_event=None,
        progress=None,
        instrument_analysis=False,
    ):
        source = Path(source)
        output_root = Path(output_root)
        if model not in {"htdemucs_ft", "htdemucs_6s"}:
            raise ValueError(f"不支持的 Demucs 模型：{model}")
        size = os.path.getsize(source)
        if size > RUNPOD_MAX_AUDIO_BYTES:
            raise ValueError(
                f"云端 Stem 音频超过成本保护上限：{RUNPOD_MAX_AUDIO_BYTES} bytes"
            )
        duration = probe_audio_duration(os.fspath(source))
        if duration <= 0 or duration > RUNPOD_MAX_AUDIO_SECONDS:
            raise ValueError(
                f"云端 Stem 音频时长必须在 0 到 {RUNPOD_MAX_AUDIO_SECONDS:g} 秒之间"
            )

        source = source.resolve()
        output_root = output_root.resolve()
        configured_root = stem_callback_store._output_root
        if configured_root is not None and source.is_relative_to(configured_root):
            callback_root = configured_root
        elif output_root.is_relative_to(source.parent):
            callback_root = source.parent
        else:
            raise ValueError("Stem 输入与输出目录不在同一受控根目录内")
        callback_id, callback_url, callback = stem_callback_store.register(
            callback_root,
            STEM_CALLBACK_PUBLIC_BASE_URL,
            STEM_CALLBACK_TTL_SECONDS,
            source_path=source,
        )
        input_url = stem_callback_store.input_url(
            callback_id, STEM_CALLBACK_PUBLIC_BASE_URL, callback
        )

        try:
            headers = {
                "Authorization": f"Bearer {RUNPOD_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "input": {
                    "task": "separate",
                    "model": model,
                    "filename": os.path.basename(source),
                    "audio_url": input_url,
                    "callback_url": callback_url,
                    "callback_task_id": callback_id,
                    "instrument_analysis": bool(instrument_analysis),
                }
            }
            from task_scheduler import ensure_not_cancelled

            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Stem 任务已取消，未提交 RunPod")
            ensure_not_cancelled()
            response = requests.post(
                f"{RUNPOD_STEM_API_URL}/run",
                headers=headers,
                json=payload,
                timeout=RUNPOD_TIMEOUT,
            )
            if response.status_code >= 400:
                detail = (response.text or "").strip()[:500]
                try:
                    data = response.json()
                    if isinstance(data, dict) and data.get("error"):
                        detail = data["error"]
                except Exception:
                    pass
                raise RuntimeError(
                    f"RunPod Stem 任务提交失败（HTTP {response.status_code}）：{detail or '无错误详情'}"
                )

            job_data = response.json()
            job_id = job_data.get("id")
            if not job_id:
                raise RuntimeError("RunPod Stem 任务提交失败：响应缺少 id")
            from task_scheduler import notify_runpod_started

            notify_runpod_started(job_id)
            if progress:
                progress(18, "RunPod 已接受任务，等待 GPU 分离")
            result = self._poll(job_id, headers, progress=progress)
            archive = result.get("archive_base64")
            if archive:  # 兼容尚未更新的 worker。
                archive_bytes = base64.b64decode(archive, validate=True)
            else:
                if progress:
                    progress(82, "RunPod 分离完成，等待轨道文件回调")

                # 跨进程轮询回调状态
                deadline = time.time() + STEM_CALLBACK_WAIT_SECONDS
                received = False
                callback_error = None
                archive_path = None

                while time.time() < deadline:
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError("Stem 任务已取消")

                    received, callback_error, archive_path = (
                        stem_callback_store.check_received(callback_id)
                    )
                    if received or callback_error:
                        break
                    time.sleep(2.0)

                if callback_error:
                    raise RuntimeError(callback_error)
                if not received:
                    raise TimeoutError(
                        f"RunPod Stem 已完成但二进制回调在预定时间内未到达（job_id={job_id}）"
                    )
                if not archive_path or not archive_path.is_file():
                    raise RuntimeError("RunPod Stem 回调已通知但结果文件不存在")
                archive_bytes = archive_path.read_bytes()
                if progress:
                    progress(86, "已收到 RunPod 轨道文件，正在校验")

            target = output_root / model / source.stem
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as bundle:
                stems = result.get("stems") or self._model_stems(model)
                expected = {f"{name}.wav" for name in stems}
                allowed = expected | {"instrument_analysis.json"}
                names = set(bundle.namelist())
                if not expected <= names or not names <= allowed:
                    raise RuntimeError("RunPod Demucs 返回的轨道清单与压缩包不一致")
                bundle.extractall(target)
            return target
        finally:
            stem_callback_store.discard(callback_id)

    @staticmethod
    def _model_stems(model):
        return {
            "htdemucs_ft": ["vocals", "drums", "bass", "other"],
            "htdemucs_6s": ["drums", "guitar", "piano", "bass", "vocals", "other"],
        }.get(model, [])

    def _poll(self, job_id, headers, progress=None):
        url = f"{RUNPOD_STEM_API_URL}/status/{job_id}"
        last_error = None
        for attempt in range(RUNPOD_MAX_POLL):
            time.sleep(RUNPOD_POLL_INTERVAL)
            try:
                response = requests.get(url, headers=headers, timeout=RUNPOD_TIMEOUT)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("RunPod 状态响应格式无效")
                last_error = None
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if progress:
                    progress(20, "RunPod 状态暂时无法读取，正在自动重试")
                continue
            status = data.get("status")
            if status == "COMPLETED":
                if progress:
                    progress(80, "RunPod GPU 分离完成，正在接收结果")
                return data.get("output") or {}
            if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
                raise RuntimeError(
                    f"RunPod Stem 任务失败：{data.get('error') or status}"
                )
            if progress:
                span = max(1, RUNPOD_MAX_POLL - 1)
                percent = 20 + int(56 * min(attempt, span) / span)
                progress(percent, f"RunPod GPU 分离中（{status or 'IN_QUEUE'}）")
        detail = (
            f"；最后一次状态读取错误：{type(last_error).__name__}"
            if last_error is not None
            else ""
        )
        raise TimeoutError(f"RunPod Stem 任务轮询超时（job_id={job_id}）{detail}")


def run_cloud_demucs(
    source,
    output_root,
    model,
    progress,
    cancel_event=None,
    instrument_analysis=True,
):
    progress(12, "正在提交显式启用的 RunPod Demucs 任务")
    target = CloudStemClient().separate(
        source,
        output_root,
        model,
        cancel_event=cancel_event,
        progress=progress,
        instrument_analysis=instrument_analysis,
    )
    progress(88, "RunPod Demucs 已回调，正在校验轨道")
    return target
