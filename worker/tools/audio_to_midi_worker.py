"""Short-lived subprocess boundary for audio-to-MIDI conversion.

The web process calls :func:`run_audio_to_midi_subprocess`; librosa, NumPy and
pYIN are imported only by the child process and their native memory is returned
to the operating system when that process exits.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping


RESULT_PREFIX = "AUDIO_TOOLKIT_MIDI_RESULT="
PROGRESS_PREFIX = "AUDIO_TOOLKIT_PROGRESS="
DEFAULT_TIMEOUT_SEC = 30 * 60


def _result_from_stdout(stdout: str) -> dict[str, Any] | None:
    """Read the last structured result while tolerating converter diagnostics."""
    for line in reversed((stdout or "").splitlines()):
        if not line.startswith(RESULT_PREFIX):
            continue
        try:
            payload = json.loads(line[len(RESULT_PREFIX) :])
        except json.JSONDecodeError as exc:
            raise RuntimeError("音频转 MIDI 子进程返回了无效 JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("音频转 MIDI 子进程返回结果格式无效")
        return payload
    return None


def _read_worker_events(
    event_file: Path, offset: int, on_runpod_started, on_progress=None
):
    """读取子进程追加的完整 JSON lines；返回新的安全偏移。"""
    if not event_file.is_file():
        return offset
    with event_file.open("r", encoding="utf-8") as stream:
        stream.seek(offset)
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line:
                return position
            if not line.endswith("\n"):
                return position
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("event") == "runpod_started" and event.get("job_id"):
                on_runpod_started(str(event["job_id"]))
            elif event.get("event") == "progress" and on_progress:
                on_progress(event.get("progress", 0), event.get("message"))


def _terminate_process_group(
    process: subprocess.Popen, grace_seconds: float = 2
) -> None:
    """Terminate the whole converter tree so ffmpeg/RunPod helpers cannot leak."""
    poll = getattr(process, "poll", None)
    if poll is not None and poll() is not None:
        return
    if os.name == "posix" and getattr(process, "pid", None) is not None:
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.communicate(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        if os.name == "posix" and getattr(process, "pid", None) is not None:
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.communicate()


def run_audio_to_midi_subprocess(
    input_path: str,
    output_dir: str,
    kwargs: Mapping[str, Any] | None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT_SEC,
    cancel_event=None,
    on_runpod_started=None,
    on_progress=None,
) -> dict[str, Any]:
    """Run ``AudioToMidiTool`` in a disposable Python process.

    Only JSON is sent over stdin/stdout. No user-controlled value is included in
    the command line, and ``shell`` is never enabled.
    """
    request = {
        "input_path": os.fspath(input_path),
        "output_dir": os.fspath(output_dir),
        "kwargs": dict(kwargs or {}),
    }
    try:
        request_json = json.dumps(request, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"音频转 MIDI 参数无法序列化为 JSON：{exc}") from exc

    command = [sys.executable, "-m", "tools.audio_to_midi_worker"]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    event_file = Path(output_dir) / f".midi_worker_{uuid.uuid4().hex}.events"
    cancel_file = Path(output_dir) / f".midi_worker_{uuid.uuid4().hex}.cancel"
    event_offset = 0
    try:
        if cancel_event is None and on_runpod_started is None:
            completed = subprocess.run(
                command,
                input=request_json,
                text=True,
                capture_output=True,
                cwd=project_root,
                timeout=timeout,
                check=False,
                start_new_session=(os.name == "posix"),
            )
        else:
            child_env = os.environ.copy()
            child_env["AUDIO_TOOLKIT_TASK_EVENT_FILE"] = os.fspath(event_file)
            child_env["AUDIO_TOOLKIT_TASK_CANCEL_FILE"] = os.fspath(cancel_file)
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=project_root,
                env=child_env,
                start_new_session=(os.name == "posix"),
            )
            deadline = time.monotonic() + timeout
            first_communicate = True
            event_callback = on_runpod_started or (lambda _job_id: None)
            while True:
                event_offset = _read_worker_events(
                    event_file, event_offset, event_callback, on_progress
                )
                if cancel_event is not None and cancel_event.is_set():
                    cancel_file.touch(mode=0o600, exist_ok=True)
                    _terminate_process_group(process)
                    raise RuntimeError("音频转 MIDI 任务已取消，子进程已终止")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _terminate_process_group(process)
                    raise RuntimeError(f"音频转 MIDI 子进程超时（{timeout:g} 秒）")
                try:
                    stdout, stderr = process.communicate(
                        input=request_json if first_communicate else None,
                        timeout=min(0.1, remaining),
                    )
                    completed = subprocess.CompletedProcess(
                        command, process.returncode, stdout, stderr
                    )
                    # 处理 stdout 中的进度行
                    for line in (stdout or "").splitlines():
                        if line.startswith(PROGRESS_PREFIX) and on_progress:
                            try:
                                p_data = json.loads(line[len(PROGRESS_PREFIX) :])
                                on_progress(
                                    p_data.get("progress", 0), p_data.get("message")
                                )
                            except:
                                pass
                    event_offset = _read_worker_events(
                        event_file, event_offset, event_callback, on_progress
                    )
                    break
                except subprocess.TimeoutExpired:
                    first_communicate = False
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"音频转 MIDI 子进程超时（{timeout:g} 秒）") from exc
    except OSError as exc:
        raise RuntimeError(f"无法启动音频转 MIDI 子进程：{exc}") from exc
    finally:
        event_file.unlink(missing_ok=True)
        cancel_file.unlink(missing_ok=True)

    payload = _result_from_stdout(completed.stdout)
    if payload is None:
        detail = (completed.stderr or completed.stdout or "无错误详情").strip()
        if len(detail) > 1000:
            detail = detail[-1000:]
        raise RuntimeError(
            f"音频转 MIDI 子进程异常退出（退出码 {completed.returncode}）：{detail}"
        )

    if completed.returncode != 0 or payload.get("success") is not True:
        message = payload.get("message") or f"退出码 {completed.returncode}"
        error_type = payload.get("error_type")
        suffix = f"（{error_type}）" if error_type else ""
        raise RuntimeError(f"音频转 MIDI 子进程失败{suffix}：{message}")

    result = payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("音频转 MIDI 子进程成功结果格式无效")
    return result


def _execute_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one worker request and execute the tool without re-spawning."""
    input_path = request.get("input_path")
    output_dir = request.get("output_dir")
    kwargs = request.get("kwargs", {})
    if not isinstance(input_path, str) or not input_path:
        raise ValueError("input_path 必须是非空字符串")
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("output_dir 必须是非空字符串")
    if not isinstance(kwargs, dict):
        raise ValueError("kwargs 必须是 JSON 对象")

    # Deliberately import the heavy conversion module only inside the child.
    from tools.audio_to_midi import AudioToMidiTool

    return AudioToMidiTool()._process_direct(input_path, output_dir, **kwargs)


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        if not isinstance(request, dict):
            raise ValueError("请求必须是 JSON 对象")
        result = _execute_request(request)
        payload = {"success": True, "result": result}
        code = 0
    except Exception as exc:  # Always give the parent a structured failure.
        payload = {
            "success": False,
            "error_type": type(exc).__name__,
            "message": str(exc) or type(exc).__name__,
        }
        code = 1

    try:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        code = 1
        encoded = json.dumps(
            {
                "success": False,
                "error_type": type(exc).__name__,
                "message": f"音频转 MIDI 结果无法序列化：{exc}",
            },
            ensure_ascii=False,
        )
    print(RESULT_PREFIX + encoded, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
