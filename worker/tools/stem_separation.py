"""CPU Demucs 异步多模型 Stem 分离和“人声消词后重混”任务。"""

import contextvars
import inspect
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.audio_editing import edit_audio

LOGGER = logging.getLogger(__name__)

MODEL_STEMS = {
    "htdemucs_ft": ("vocals", "drums", "bass", "other"),
    "htdemucs_6s": ("vocals", "drums", "bass", "guitar", "piano", "other"),
}
STEM_LABELS = {
    "vocals": "人声 / Vocals", "drums": "鼓 / Drums", "bass": "贝斯 / Bass",
    "guitar": "吉他 / Guitar", "piano": "钢琴 / Piano", "other": "其他 / Other",
}
STAGE_LABELS = {
    "queued": "排队", "preparing": "准备 / 下载模型", "separating": "分离处理中",
    "organizing": "整理轨道", "completed": "完成", "failed": "失败", "cancelled": "已取消",
}
DEFAULT_MODEL = "htdemucs_6s"
# 保留旧导入名，语义为默认模型及其输出。
MODEL = DEFAULT_MODEL
STEMS = MODEL_STEMS[DEFAULT_MODEL]


class StemTaskManager:
    def __init__(self, output_root, max_workers=1, runner=None):
        self.output_root = Path(output_root).resolve()
        self.task_root = self.output_root / "stem_tasks"
        self.task_root.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)), thread_name_prefix="demucs-cpu")
        self.runner = runner or self._run_demucs
        self.tasks = {}
        self.cancel_events = {}
        self.lock = threading.RLock()

    def submit(
        self,
        source_path,
        combined=False,
        ranges=None,
        operation="mute",
        fade_ms=12,
        model=DEFAULT_MODEL,
        instrument_analysis=True,
        release_semaphore=None,
    ):
        if model not in MODEL_STEMS:
            raise ValueError(f"不支持的 Demucs 模型: {model}")
        task_id = uuid.uuid4().hex
        folder = self.task_root / task_id
        try:
            folder.mkdir(parents=True, exist_ok=False)
            ext = Path(source_path).suffix.lower() or ".wav"
            source = folder / f"input{ext}"
            shutil.copy2(os.fspath(source_path), source)
            task = {
                "id": task_id, "status": "queued", "stage": "queued", "stage_label": STAGE_LABELS["queued"],
                "progress": 0, "message": "等待 CPU 分离队列", "created": time.time(), "outputs": [],
                "tracks": [], "model": model, "stems": list(MODEL_STEMS[model]),
                "instrument_analysis_requested": bool(instrument_analysis),
                "instrument_analysis": None,
                "completed_tasks": 0, "pending_tasks": 1,
                "completed_tracks": [], "pending_tracks": [],
            }
            cancel_event = threading.Event()
            with self.lock:
                self.tasks[task_id] = task
                self.cancel_events[task_id] = cancel_event
                self._persist_locked(task_id)
            context = contextvars.copy_context()
            self.executor.submit(
                context.run, self._execute, task_id, source, folder, combined, ranges or [],
                operation, fade_ms, model, bool(instrument_analysis), cancel_event,
                release_semaphore
            )
            return task_id
        except Exception:
            with self.lock:
                self.tasks.pop(task_id, None)
                self.cancel_events.pop(task_id, None)
            shutil.rmtree(folder, ignore_errors=True)
            if release_semaphore:
                try:
                    release_semaphore.release()
                except ValueError:
                    pass
            raise

    def get(self, task_id):
        if not isinstance(task_id, str) or not re.fullmatch(r"[0-9a-f]{32}", task_id):
            return None
        with self.lock:
            # 每次优先读取共享磁盘状态，避免多 Worker 轮询命中不同进程时
            # 把仍在运行的任务误判为“不存在”或长期返回旧进度。
            task = self._load_from_disk_locked(task_id) or self.tasks.get(task_id)
            if not task:
                return None
            self.tasks[task_id] = task
            return dict(task)

    def cancel(self, task_id):
        """请求停止 queued/running Stem；实际子进程由执行线程终止。"""
        with self.lock:
            task = self.tasks.get(task_id) or self._load_from_disk_locked(task_id)
            event = self.cancel_events.get(task_id)
            if not task or task.get("status") not in {"queued", "running"} or event is None:
                return False
            self.tasks[task_id] = task
            event.set()
            task.update(status="cancelled", stage="cancelled", stage_label="已取消",
                        message="任务已取消", pending_tasks=0)
            self._persist_locked(task_id)
            return True

    @staticmethod
    def _raise_if_cancelled(cancel_event):
        if cancel_event.is_set():
            raise RuntimeError("Stem 任务已取消")

    def _persist_locked(self, task_id):
        task = self.tasks.get(task_id)
        folder = self.task_root / task_id
        if not task or not folder.is_dir():
            return
        status_file = folder / "status.json"
        tmp_file = folder / "status.json.tmp"
        try:
            tmp_file.write_text(json.dumps(task, ensure_ascii=False), encoding="utf-8")
            tmp_file.replace(status_file)
        except Exception:
            tmp_file.unlink(missing_ok=True)

    def _load_from_disk_locked(self, task_id):
        status_file = self.task_root / task_id / "status.json"
        if not status_file.is_file():
            return None
        try:
            return json.loads(status_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _update(self, task_id, **values):
        """原子更新任务；总进度只允许前进，避免 bag 子模型百分比归零造成回退。"""
        with self.lock:
            task = self.tasks[task_id]
            if task.get("status") == "cancelled" and values.get("status") != "cancelled":
                return
            if "progress" in values:
                values["progress"] = max(task.get("progress", 0), min(100, int(values["progress"])))
            if "stage" in values:
                values.setdefault("stage_label", STAGE_LABELS[values["stage"]])
            task.update(values)
            self._persist_locked(task_id)

    def _run_demucs(self, source, demucs_root, model, progress, cancel_event=None):
        cancel_event = cancel_event or threading.Event()
        command = [sys.executable, "-m", "tools.demucs_worker", "-n", model, "--device", "cpu", "--out", os.fspath(demucs_root), os.fspath(source)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        recent_output = deque(maxlen=20)
        output_queue = queue.Queue()

        def read_output():
            try:
                for line in process.stdout or ():
                    output_queue.put(line)
            finally:
                output_queue.put(None)

        threading.Thread(target=read_output, daemon=True).start()
        output_finished = False
        while process.poll() is None or not output_finished:
            if cancel_event.is_set():
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise RuntimeError("Stem 任务已取消，Demucs 子进程已终止")
            try:
                line = output_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                output_finished = True
                continue
            text = line.strip()
            if text:
                recent_output.append(text)
            match = re.search(r"AUDIO_TOOLKIT_PROGRESS (\d{1,3})", line) or re.search(r"(\d{1,3})%", line)
            if match:
                progress(min(88, 10 + int(match.group(1)) * 0.78), "Demucs 正在分离（多模型进度可能分轮显示）")
        code = process.wait()
        if code:
            detail = "\n".join(recent_output)
            if "No module named demucs" in detail:
                detail = "当前 Python 环境未安装 Demucs，请执行：python3 -m pip install -r requirements-local.txt"
            elif not detail:
                detail = "Demucs 未返回错误详情，请在启动 app_local.py 的同一 Python 环境中检查安装。"
            raise RuntimeError(f"Demucs 处理失败（退出码 {code}）：{detail}")

    def _execute(
        self,
        task_id,
        source,
        folder,
        combined,
        ranges,
        operation,
        fade_ms,
        model,
        instrument_analysis,
        cancel_event,
        release_semaphore=None,
    ):
        import numpy as np
        import soundfile as sf
        stems = MODEL_STEMS[model]
        demucs_root = folder / "demucs"
        try:
            self._raise_if_cancelled(cancel_event)
            self._update(task_id, status="running", stage="preparing", progress=3,
                         message=f"正在准备 {model}；首次运行可能需要下载模型")
            self._update(task_id, stage="separating", progress=10, message="Demucs 正在分离")
            runner_parameters = inspect.signature(self.runner).parameters
            runner_kwargs = ({"cancel_event": cancel_event}
                             if "cancel_event" in runner_parameters else {})
            if "instrument_analysis" in runner_parameters:
                runner_kwargs["instrument_analysis"] = instrument_analysis
            self.runner(source, demucs_root, model, lambda value, text="": self._update(
                task_id, stage="separating", progress=value, message=text or "Demucs 正在分离"),
                **runner_kwargs)
            self._raise_if_cancelled(cancel_event)
            self._update(task_id, stage="organizing", progress=90, message="正在校验并整理轨道")
            stem_dir = demucs_root / model / source.stem
            paths = {name: stem_dir / f"{name}.wav" for name in stems}
            missing = [name for name, path in paths.items() if not path.is_file()]
            if missing:
                raise RuntimeError(f"Demucs 未生成 {model} 的完整 {len(stems)} 轨: " + ", ".join(missing))
            from tools.audio_result_validation import validate_stem_outputs

            validate_stem_outputs(paths.values(), source)
            final_dir = folder / "results"
            final_dir.mkdir()
            final_stems = []
            for name, path in paths.items():
                self._raise_if_cancelled(cancel_event)
                target = final_dir / f"{name}.wav"
                shutil.copy2(path, target)
                final_stems.append(target)
            instrument_result = None
            instrument_report = stem_dir / "instrument_analysis.json"
            if instrument_report.is_file():
                try:
                    instrument_result = json.loads(
                        instrument_report.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    raise RuntimeError("乐器识别结果损坏") from exc
                if (
                    not isinstance(instrument_result, dict)
                    or not isinstance(instrument_result.get("instruments"), list)
                ):
                    raise RuntimeError("乐器识别结果格式无效")
                shutil.copy2(
                    instrument_report,
                    final_dir / "instrument_analysis.json",
                )
            outputs = list(final_stems)
            if combined:
                self._raise_if_cancelled(cancel_event)
                edited_vocal = final_dir / "vocals_edited.wav"
                edit_audio(final_dir / "vocals.wav", edited_vocal, ranges, operation, fade_ms)
                arrays, sr = [], None
                for name in stems:
                    source_path = edited_vocal if name == "vocals" else final_dir / f"{name}.wav"
                    data, current_sr = sf.read(source_path, always_2d=True, dtype="float32")
                    if sr is not None and current_sr != sr:
                        raise RuntimeError("Stem 采样率不一致")
                    sr = current_sr
                    arrays.append(data)
                length = min(map(len, arrays))
                mixed = np.clip(sum(item[:length] for item in arrays), -1, 1)
                remix = final_dir / "remix_edited.wav"
                sf.write(remix, mixed, sr, subtype="PCM_16")
                outputs.extend([edited_vocal, remix])
            relative = [os.fspath(path.relative_to(self.output_root)) for path in outputs]
            tracks = []
            for path, rel in zip(outputs, relative):
                stem_id = path.stem
                label = STEM_LABELS.get(stem_id, "编辑后人声" if stem_id == "vocals_edited" else "重混结果")
                tracks.append({"stem_id": stem_id, "label": label, "model": model, "path": rel})
            self._raise_if_cancelled(cancel_event)
            instrument_count = len(
                (instrument_result or {}).get("instruments", [])
            )
            message = "处理完成"
            if instrument_result is not None:
                message += f"，识别到 {instrument_count} 类乐器"
            self._update(task_id, status="completed", stage="completed", progress=100,
                         message=message, outputs=relative, tracks=tracks,
                         instrument_analysis=instrument_result,
                         completed_tasks=1, pending_tasks=0,
                         completed_tracks=[track["stem_id"] for track in tracks], pending_tracks=[])
        except Exception as exc:
            if cancel_event.is_set():
                self._update(task_id, status="cancelled", stage="cancelled",
                             stage_label="已取消", message="任务已取消", pending_tasks=0)
            else:
                LOGGER.exception(
                    "Stem execution failed task_id=%s model=%s", task_id, model
                )
                self._update(task_id, status="failed", stage="failed", message=str(exc))
        finally:
            with self.lock:
                self.cancel_events.pop(task_id, None)
            shutil.rmtree(demucs_root, ignore_errors=True)
            Path(source).unlink(missing_ok=True)
            if release_semaphore:
                try:
                    release_semaphore.release()
                except ValueError:
                    pass
