"""
设备检测模块 (Device Detector) — 云部署版

运行时自动检测最优推理后端:
- NVIDIA CUDA (float16, 最快)
- Apple Silicon MPS / Metal (float16, 中快)
- CPU fallback (float32, 最慢但兼容性最好)

云部署模式下 (torch 不可用时), 返回云端 stub 信息。
"""

import os
import platform
import logging

logger = logging.getLogger('device_detector')
logger.setLevel(logging.INFO)

# ─── 检测 torch 是否可用 ───

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.info("[Device] torch 不可用 — 运行在云部署模式 (RVC 由 RunPod 处理)")


def detect_device():
    """
    检测最优推理设备

    返回: dict with:
        - backend: 'cuda' | 'mps' | 'cpu' | 'cloud'
        - device: torch.device 对象 (云模式下为 None)
        - half_precision: bool
        - num_threads: int
        - device_name: str
    """
    if not _TORCH_AVAILABLE:
        # ─── 云端模式: torch 不可用 ───
        return {
            'backend': 'cloud',
            'device': None,
            'half_precision': False,
            'num_threads': 1,
            'device_name': 'Cloud (RunPod Serverless)',
        }

    result = {
        'backend': 'cpu',
        'device': torch.device('cpu'),
        'half_precision': False,
        'num_threads': 1,
        'device_name': 'CPU',
    }

    # ─── 1. 检测 NVIDIA CUDA ───
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            result['backend'] = 'cuda'
            result['device'] = torch.device('cuda')
            result['half_precision'] = True
            result['num_threads'] = 1
            result['device_name'] = f'CUDA: {gpu_name}'
            logger.info(f"[Device] 检测到 NVIDIA CUDA GPU: {gpu_name}")
            return result
    except Exception as e:
        logger.warning(f"[Device] CUDA 检测失败: {e}")

    # ─── 2. 检测 Apple Silicon MPS ───
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            chip_name = _get_apple_chip_name()
            result['backend'] = 'mps'
            result['device'] = torch.device('mps')
            result['half_precision'] = False
            result['num_threads'] = _get_cpu_count()
            result['device_name'] = f'MPS: Apple {chip_name}'
            logger.info(f"[Device] 检测到 Apple Silicon MPS: {chip_name}")
            return result
    except Exception as e:
        logger.warning(f"[Device] MPS 检测失败: {e}")

    # ─── 3. CPU fallback ───
    cpu_count = _get_cpu_count()
    result['backend'] = 'cpu'
    result['device'] = torch.device('cpu')
    result['half_precision'] = False
    result['num_threads'] = min(cpu_count, 4)
    result['device_name'] = f'CPU ({cpu_count} cores)'
    logger.info(f"[Device] 使用 CPU 模式 ({cpu_count} cores)")

    return result


def _get_apple_chip_name():
    """获取 Apple Silicon 芯片名称"""
    try:
        import subprocess
        result = subprocess.run(
            ['sysctl', '-n', 'machdep.cpu.brand_string'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return platform.processor() or 'Apple Silicon'


def _get_cpu_count():
    """获取 CPU 核心数"""
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def setup_environment(device_info=None):
    """根据检测结果设置环境变量和线程数"""
    if device_info is None:
        _setup_pre_import()
        return

    if not _TORCH_AVAILABLE:
        return

    backend = device_info['backend']
    num_threads = device_info['num_threads']

    try:
        torch.set_num_threads(num_threads)
        logger.info(f"[Device] torch 线程数: {num_threads}")
    except Exception:
        pass

    if backend == 'cpu':
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)
    else:
        for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
            os.environ.pop(key, None)

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def _setup_pre_import():
    """torch 导入前的环境设置 (安全, 不依赖 torch)"""
    cpu_count = _get_cpu_count()
    threads = min(cpu_count, 4)

    os.environ.setdefault('OMP_NUM_THREADS', str(threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(threads))
    os.environ.setdefault('OPENBLAS_NUM_THREADS', str(threads))
    os.environ.setdefault('NUMEXPR_NUM_THREADS', str(threads))
    os.environ.setdefault('VECLIB_MAXIMUM_THREADS', str(threads))
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


# ─── 单例: 只检测一次 ───

_device_info = None


def get_device_info():
    """获取设备信息 (单例, 只检测一次)"""
    global _device_info
    if _device_info is None:
        _device_info = detect_device()
        setup_environment(_device_info)
    return _device_info


def get_device_summary():
    """获取设备摘要字符串 (用于 UI 显示)"""
    info = get_device_info()
    return info['device_name']
