"""
F0 预测器模型校验 (共享模块)

vocal_transform.py 和 runpod_handler.py 共同使用，
避免同一逻辑在两处各自维护导致不同步。
"""

import os
import logging

logger = logging.getLogger('f0_utils')


def validate_f0_model(f0_method, rvc_internal_dir):
    """
    校验 F0 预测器模型文件，回退到可用方法。

    Args:
        f0_method: 首选 F0 方法 ('crepe' / 'rmvpe' / 'fcpe')
        rvc_internal_dir: RVC predictors 目录路径

    Returns:
        str: 可用的 F0 方法
    """
    f0_files = {
        'rmvpe': os.path.join(rvc_internal_dir, 'rmvpe.pt'),
        'fcpe': os.path.join(rvc_internal_dir, 'fcpe.pt'),
    }

    preferred = f0_method
    candidates = [preferred]
    if preferred != 'crepe':
        candidates.append('crepe')
    if preferred != 'rmvpe':
        candidates.append('rmvpe')
    if preferred != 'fcpe':
        candidates.append('fcpe')

    for method in candidates:
        if method == 'crepe':
            logger.info(f"[RVC] F0 方法: {method} (torchcrepe, 内置中值滤波)")
            return method
        model_file = f0_files.get(method, '')
        if model_file and os.path.isfile(model_file):
            file_size = os.path.getsize(model_file)
            if file_size > 1000:
                logger.info(f"[RVC] F0 方法: {method} ({file_size} bytes)")
                return method
            else:
                logger.warning(f"[RVC] F0 模型可能是占位文件: {model_file} ({file_size} bytes)")
        else:
            logger.warning(f"[RVC] F0 模型不存在: {model_file}")

    logger.info("[RVC] F0 方法: crepe (回退默认)")
    return 'crepe'
