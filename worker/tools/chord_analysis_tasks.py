"""线程安全的和弦分析后台任务管理器。"""
import contextvars
import json
import os
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TASK_ID_PATTERN = re.compile(r"[0-9a-f]{32}")

class ChordAnalysisTaskManager:
    """在进程内执行和弦分析，状态查询支持内存读取与轻量磁盘持久化。"""

    def __init__(self, output_root, max_workers=2, task_ttl_seconds=24 * 60 * 60,
                 task_timeout_seconds=30 * 60, task_namespace="chord_tasks",
                 task_label="和弦分析"):
        self.output_root = Path(output_root).resolve()
        self.task_label = str(task_label)
        self.task_root = self.output_root / task_namespace
        self.task_root.mkdir(parents=True, exist_ok=True)
        self.task_ttl_seconds = max(1, int(task_ttl_seconds))
        self.task_timeout_seconds = max(1, int(task_timeout_seconds))
        self.executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)),
                                           thread_name_prefix="chord-analysis")
        self.tasks = {}
        self.cancel_events = {}
        self.lock = threading.RLock()

    def submit(self, source_path, tool, kwargs):
        task_id = uuid.uuid4().hex
        folder = (self.task_root / task_id).resolve()
        try:
            if folder.parent != self.task_root:
                raise ValueError("无效任务目录")
            folder.mkdir(parents=True, exist_ok=False)
            source = folder / ("input" + (Path(source_path).suffix.lower() or ".wav"))
            shutil.copy2(os.fspath(source_path), source)

            task = {
                "task_id": task_id, "status": "queued", "success": True,
                "message": "任务已提交", "created_at": time.time(),
                "started_monotonic": time.monotonic(),
                "stage": "queued", "stage_label": "排队", "progress": 0,
            }
            cancel_event = threading.Event()
            with self.lock:
                self._cleanup_locked()
                self.tasks[task_id] = task
                self.cancel_events[task_id] = cancel_event
                self._persist_locked(task_id)

            context = contextvars.copy_context()
            self.executor.submit(context.run, self._execute, task_id, tool, source, folder,
                                 dict(kwargs))
            return task_id
        except Exception:
            with self.lock:
                self.tasks.pop(task_id, None)
                self.cancel_events.pop(task_id, None)
            shutil.rmtree(folder, ignore_errors=True)
            raise

    def get(self, task_id):
        if not isinstance(task_id, str) or not TASK_ID_PATTERN.fullmatch(task_id):
            return None
        with self.lock:
            self._cleanup_locked()
            task = self.tasks.get(task_id)
            if not task:
                task = self._load_from_disk_locked(task_id)
                if not task:
                    return None
                self.tasks[task_id] = task

            if (task["status"] in {"queued", "running"}
                    and time.monotonic() - task.get("started_monotonic", time.monotonic()) > self.task_timeout_seconds):
                event = self.cancel_events.get(task_id)
                if event:
                    event.set()
                task.update(status="failed", success=False, message=f"{self.task_label}任务超时")
                self._persist_locked(task_id)
            return dict(task)

    def _update(self, task_id, **values):
        with self.lock:
            task = self.tasks.get(task_id)
            if task and task.get("status") not in {"failed", "cancelled"}:
                if "progress" in values:
                    values["progress"] = max(
                        int(task.get("progress", 0)),
                        min(100, max(0, int(values["progress"]))),
                    )
                task.update(values)
                self._persist_locked(task_id)

    def _persist_locked(self, task_id):
        task = self.tasks.get(task_id)
        if not task:
            return
        folder = self.task_root / task_id
        if not folder.exists():
            return
        status_file = folder / "status.json"
        tmp_file = folder / "status.json.tmp"
        try:
            tmp_file.write_text(json.dumps(task, ensure_ascii=False), encoding="utf-8")
            tmp_file.replace(status_file)
        except Exception:
            if tmp_file.exists():
                tmp_file.unlink()

    def _load_from_disk_locked(self, task_id):
        status_file = self.task_root / task_id / "status.json"
        if not status_file.exists():
            return None
        try:
            task = json.loads(status_file.read_text(encoding="utf-8"))
            if task.get("status") in {"queued", "running"}:
                task.update(status="failed", success=False, message="任务因服务器重启中断")
            return task
        except (Exception, json.JSONDecodeError):
            return None

    def _cleanup_locked(self):
        cutoff = time.time() - self.task_ttl_seconds
        expired = [task_id for task_id, task in self.tasks.items()
                   if task.get("created_at", 0) < cutoff and task.get("status") not in {"queued", "running"}]
        for task_id in expired:
            self.tasks.pop(task_id, None)
            self.cancel_events.pop(task_id, None)
            shutil.rmtree(self.task_root / task_id, ignore_errors=True)

        # 清理进程重启后未恢复进内存的过期任务目录。
        for folder in self.task_root.iterdir():
            if (not folder.is_dir() or not TASK_ID_PATTERN.fullmatch(folder.name)
                    or folder.name in self.tasks):
                continue
            status_file = folder / "status.json"
            try:
                modified = status_file.stat().st_mtime if status_file.exists() else folder.stat().st_mtime
            except OSError:
                continue
            if modified < cutoff:
                shutil.rmtree(folder, ignore_errors=True)

    def cancel(self, task_id):
        if not isinstance(task_id, str) or not TASK_ID_PATTERN.fullmatch(task_id):
            return False
        with self.lock:
            task = self.tasks.get(task_id)
            if not task or task.get("status") not in {"queued", "running"}:
                return False
            self.cancel_events.setdefault(task_id, threading.Event()).set()
            task.update(status="cancelled", success=False, message="任务已取消")
            self._persist_locked(task_id)
            return True

    def _execute(self, task_id, tool, source, folder, kwargs):
        try:
            with self.lock:
                cancel_event = self.cancel_events.setdefault(task_id, threading.Event())
            if cancel_event.is_set():
                return
            self._update(
                task_id,
                status="running",
                stage="preparing",
                stage_label="准备",
                progress=2,
                message=f"正在准备{self.task_label}",
            )
            output_dir = folder / "results"
            output_dir.mkdir()

            started = time.time()
            process_kwargs = dict(kwargs)
            process_kwargs["_cancel_event"] = cancel_event
            if self.task_label == "和弦分析":
                process_kwargs["_progress_callback"] = (
                    lambda value, message: self._update(
                        task_id,
                        status="running",
                        stage="analyzing",
                        stage_label="分析中",
                        progress=value,
                        message=message,
                    )
                )
            result = tool.process(os.fspath(source), os.fspath(output_dir), **process_kwargs)
            if cancel_event.is_set():
                shutil.rmtree(output_dir, ignore_errors=True)
                return
            if not result.get("success"):
                raise RuntimeError(result.get("message") or f"{self.task_label}失败")

            public_result = dict(result)
            public_outputs = []
            for output in result.get("output_files", []):
                candidate = Path(output)
                if not candidate.is_absolute():
                    candidate = output_dir / candidate
                candidate = candidate.resolve()

                try:
                    safe = candidate.is_file() and os.path.commonpath([self.output_root, candidate]) == os.fspath(self.output_root)
                except (AttributeError, ValueError):
                    safe = False

                if not safe:
                    raise RuntimeError("分析结果包含无效输出路径")
                public_outputs.append(os.fspath(candidate.relative_to(self.output_root)))

            public_result["output_files"] = public_outputs
            public_result["elapsed_sec"] = round(time.time() - started, 2)
            self._update(
                task_id,
                status="completed",
                success=True,
                stage="completed",
                stage_label="完成",
                progress=100,
                message=public_result.get("message", "分析完成"),
                result=public_result,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._update(task_id, status="failed", success=False, message=f"处理失败: {exc}")
        finally:
            source.unlink(missing_ok=True)
            with self.lock:
                self.cancel_events.pop(task_id, None)
