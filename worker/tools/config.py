"""云端 GPU 配置；所有 RunPod 能力默认关闭，必须逐项显式 opt-in。"""
import os
import threading


def _enabled(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _endpoint_url(name):
    endpoint_id = os.environ.get(name, "").strip()
    return f"https://api.runpod.ai/v2/{endpoint_id}" if endpoint_id else ""


RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_RVC_ENABLED = _enabled("RUNPOD_RVC_ENABLED")
RUNPOD_SEED_VC_ENABLED = _enabled("RUNPOD_SEED_VC_ENABLED")
RUNPOD_STEM_ENABLED = _enabled("RUNPOD_STEM_ENABLED")
RUNPOD_STRUCTURE_ENABLED = _enabled("RUNPOD_STRUCTURE_ENABLED")  # deprecated compatibility flag
RUNPOD_RVC_API_URL = _endpoint_url("RUNPOD_RVC_ENDPOINT_ID")
RUNPOD_SEED_VC_API_URL = _endpoint_url("RUNPOD_SEED_VC_ENDPOINT_ID")
RUNPOD_STEM_API_URL = _endpoint_url("RUNPOD_STEM_ENDPOINT_ID")
_structure_endpoint_id = (
    os.environ.get("RUNPOD_STRUCTURE_ENDPOINT_ID", "").strip()
    or os.environ.get("RUNPOD_STEM_ENDPOINT_ID", "").strip()
    or os.environ.get("RUNPOD_RVC_ENDPOINT_ID", "").strip()
)
RUNPOD_STRUCTURE_API_URL = (
    f"https://api.runpod.ai/v2/{_structure_endpoint_id}" if _structure_endpoint_id else ""
)
RUNPOD_STEM_PROTOCOL = os.environ.get("RUNPOD_STEM_PROTOCOL", "").strip().lower()
RUNPOD_STRUCTURE_PROTOCOL = os.environ.get("RUNPOD_STRUCTURE_PROTOCOL", "allin1-v1").strip().lower()

# 兼容旧导入名，但不再让旧 USE_CLOUD_RVC/RUNPOD_ENDPOINT_ID 隐式开启计费。
USE_CLOUD_RVC = RUNPOD_RVC_ENABLED
RUNPOD_API_URL = RUNPOD_RVC_API_URL

# OpenAI 音频转文字配置
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

RUNPOD_TIMEOUT = int(os.environ.get("RUNPOD_TIMEOUT", "300"))
RUNPOD_POLL_INTERVAL = float(os.environ.get("RUNPOD_POLL_INTERVAL", "5"))
RUNPOD_MAX_POLL = int(os.environ.get("RUNPOD_MAX_POLL", "120"))
RUNPOD_MAX_AUDIO_BYTES = int(os.environ.get("RUNPOD_MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))
RUNPOD_MAX_AUDIO_SECONDS = float(os.environ.get("RUNPOD_MAX_AUDIO_SECONDS", "600"))

# Demucs ZIP 走 Render 一次性二进制回调；公共基址只能来自部署配置，绝不读取 Host。
STEM_CALLBACK_PUBLIC_BASE_URL = os.environ.get("STEM_CALLBACK_PUBLIC_BASE_URL", "").strip().rstrip("/")
STEM_CALLBACK_MAX_BYTES = int(os.environ.get("STEM_CALLBACK_MAX_BYTES", str(512 * 1024 * 1024)))
STEM_CALLBACK_TTL_SECONDS = float(os.environ.get("STEM_CALLBACK_TTL_SECONDS", "900"))
STEM_CALLBACK_WAIT_SECONDS = float(os.environ.get("STEM_CALLBACK_WAIT_SECONDS", "180"))

GPU_CONCURRENCY_LIMIT = max(1, int(os.environ.get("GPU_CONCURRENCY_LIMIT", "5")))
gpu_semaphore = threading.BoundedSemaphore(GPU_CONCURRENCY_LIMIT)
