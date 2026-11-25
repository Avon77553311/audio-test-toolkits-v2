"""Strict request parsing and validation for opt-in RunPod Stem billing."""


def parse_use_runpod(value):
    """Parse the HTML checkbox contract: absent/empty is false, only ``on`` is true."""
    if value in (None, ''):
        return False
    if value == 'on':
        return True
    raise ValueError('use_runpod 参数无效；仅支持未勾选或 on')


def validate_runpod_request(quality_mode, raw_value):
    """Validate one MIDI/chord request and return a strict bool."""
    requested = parse_use_runpod(raw_value)
    if not requested:
        return False
    if quality_mode != 'high':
        raise ValueError('使用 RunPod GPU Stem 必须选择质量优先模式')

    from tools import config
    if not config.RUNPOD_STEM_ENABLED:
        raise ValueError('服务端未启用 RunPod GPU Stem，请取消勾选后使用本地分析')
    if config.RUNPOD_STEM_PROTOCOL != 'demucs-v1':
        raise ValueError('服务端 RunPod Stem 协议未配置为 demucs-v1，无法执行本次付费请求')
    return True
