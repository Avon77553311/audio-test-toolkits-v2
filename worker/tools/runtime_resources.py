"""Runtime resource helpers shared by Render and RunPod workers."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def runtime_directory() -> Path:
    configured = os.environ.get("AUDIO_TOOLKIT_RUNTIME_DIR")
    root = (
        Path(configured).expanduser()
        if configured
        else Path(tempfile.gettempdir()) / "audio-toolkit-runtime"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def executable_path(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"媒体组件 {name} 不可用")
    return name


def run_cancellable(
    command,
    *,
    timeout: float,
    cancel_event=None,
    check: bool = False,
    cancel_message: str = "任务已取消",
):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + float(timeout)
    while process.poll() is None:
        if cancel_event is not None and cancel_event.is_set():
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise RuntimeError(cancel_message)
        if time.monotonic() >= deadline:
            process.kill()
            process.wait()
            raise subprocess.TimeoutExpired(command, timeout)
        time.sleep(0.05)
    stdout, stderr = process.communicate()
    completed = subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout,
        stderr,
    )
    if check and process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode,
            command,
            output=stdout,
            stderr=stderr,
        )
    return completed


def chord_model_files() -> tuple[Path, ...]:
    from tools.chord_models import model_paths,complete
    paths=model_paths('pt')
    if not complete(paths):raise RuntimeError('完整且校验通过的五折 TorchScript 和弦模型未安装')
    return paths


def chord_onnx_model_files() -> tuple[Path, ...]:
    from tools.chord_models import model_paths,complete
    paths=model_paths('onnx')
    if not complete(paths):raise RuntimeError('完整且校验通过的五折 ONNX 和弦模型未安装')
    return paths


def instrument_model_path() -> Path:
    from tools.instrument_model import model_path,valid_model
    path=model_path()
    if not valid_model(path):raise RuntimeError('EfficientAT MN10 完整权重未安装或 SHA256 校验失败')
    return path
