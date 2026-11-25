"""
音频处理管线 (Audio Pipeline)
为 RVC 音色转换提供预处理和后处理能力

预处理 (转换前):
1. Vocal Isolation — 基于 librosa HPSS 的人声分离 (去除伴奏/背景)
2. Noise Reduction — 基于 noisereduce 的频谱门限降噪
3. De-reverb — 基于频谱衰减的去混响
4. Silence Trim — 去除首尾静音段
5. Peak Normalization — 峰值归一化到安全电平

后处理 (转换后):
1. High-pass Filter — 去除低频隆隆声 (80Hz)
2. De-essing — 动态去齿音 (基于频谱检测)
3. EQ — 频率响应修正 (低频 shelving + 高频 shelving + 中频峰值)
4. Compressor — 动态范围压缩 (控制响度波动)
5. Harmonic Enhancement — 谐波增强 (增加暖度/存在感)
6. LUFS Normalization — 目标响度归一化 (基于 RMS 近似 LUFS)
7. Limiter — 软限幅 (防止 clipping)

所有处理均为 CPU 模式，无需 GPU。
依赖: numpy, scipy, librosa, noisereduce, pedalboard, soundfile
"""

import os
import logging

logger = logging.getLogger('audio_pipeline')
logger.setLevel(logging.INFO)

# ─── 预处理 ───

def isolate_vocals(audio_data, sample_rate):
    """
    人声分离: 基于 HPSS (Harmonic-Percussive Source Separation)
    librosa.effects.hpss 将音频分解为谐波分量和打击分量
    人声主要在谐波分量中，伴奏/打击乐在打击分量中

    对于已经干净的纯人声输入，此步骤影响极小 (谐波分量≈原始信号)
    """
    import librosa
    import numpy as np
    logger.info("[Pipeline] 预处理: 人声分离 (HPSS)")
    try:
        harmonic, percussive = librosa.effects.hpss(audio_data, margin=2.0)
        # harmonic 即为人声主体
        # 但保留少量 percussive 以维持辅音清晰度
        result = harmonic + percussive * 0.15
        logger.info(f"[Pipeline] 人声分离完成, 形状: {result.shape}")
        return result.astype(np.float32)
    except Exception as e:
        logger.warning(f"[Pipeline] 人声分离失败，跳过: {e}")
        return audio_data


def reduce_noise(audio_data, sample_rate, prop_decrease=0.8, stationary=False):
    """
    降噪: 基于 noisereduce 频谱门限

    使用非平稳模式 (non-stationary) 自适应估计噪声
    prop_decrease: 噪声抑制比例 (0=不抑制, 1=完全抑制)
    """
    import numpy as np
    logger.info(f"[Pipeline] 预处理: 降噪 (prop_decrease={prop_decrease})")
    try:
        import noisereduce as nr

        # noisereduce 需要 (channels, samples) 格式
        if audio_data.ndim == 1:
            audio_2d = audio_data[np.newaxis, :]
        else:
            audio_2d = audio_data.T

        reduced = nr.reduce_noise(
            y=audio_2d,
            sr=sample_rate,
            stationary=stationary,
            prop_decrease=prop_decrease,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            use_torch=False,
        )

        if audio_data.ndim == 1:
            result = reduced[0]
        else:
            result = reduced.T

        logger.info("[Pipeline] 降噪完成")
        return result.astype(np.float32)
    except Exception as e:
        logger.warning(f"[Pipeline] 降噪失败，跳过: {e}")
        return audio_data


def dereverb(audio_data, sample_rate, decay=0.3, iterations=2):
    """
    去混响: 基于频谱衰减的自适应方法

    原理: 混响信号的尾部能量衰减缓慢，通过分析能量包络
    对衰减缓慢的部分进行抑制

    decay: 衰减系数 (越大去除混响越强，但也可能损失人声尾部)
    iterations: 迭代次数
    """
    import librosa
    import numpy as np
    logger.info(f"[Pipeline] 预处理: 去混响 (decay={decay}, iters={iterations})")
    try:
        n_fft = 2048
        hop_length = 512

        # STFT
        stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        decay = float(np.clip(decay, 0.0, 1.0))
        iterations = max(0, int(iterations))
        envelope_release = np.exp(-hop_length / (sample_rate * 0.15))

        for _ in range(iterations):
            # 逐频点跟踪带 150ms 释放的峰值包络；新瞬态会立即抬高包络，
            # 混响尾部则落在缓慢释放的包络之下。
            energy_envelope = np.empty_like(magnitude)
            energy_envelope[:, 0] = magnitude[:, 0]
            for frame in range(1, magnitude.shape[1]):
                energy_envelope[:, frame] = np.maximum(
                    magnitude[:, frame], energy_envelope[:, frame - 1] * envelope_release
                )

            # ratio 接近 1 表示瞬态/持续直达声，越小越可能是衰减尾部。
            ratio = np.clip(
                magnitude / np.maximum(energy_envelope, 1e-10), 0.0, 1.0
            )
            tail_weight = (1.0 - ratio) ** 2
            gain = np.clip(1.0 - decay * tail_weight, 1.0 - decay, 1.0)
            magnitude = np.nan_to_num(magnitude * gain, nan=0.0, posinf=0.0, neginf=0.0)

        # 重建
        stft_processed = magnitude * np.exp(1j * phase)
        result = librosa.istft(stft_processed, hop_length=hop_length)

        # 补齐长度
        if len(result) < len(audio_data):
            result = np.pad(result, (0, len(audio_data) - len(result)))
        else:
            result = result[:len(audio_data)]

        logger.info("[Pipeline] 去混响完成")
        return result.astype(np.float32)
    except Exception as e:
        logger.warning(f"[Pipeline] 去混响失败，跳过: {e}")
        return audio_data


def trim_silence(audio_data, sample_rate, threshold_db=-40, frame_ms=30):
    """
    去除首尾静音段

    threshold_db: 低于此 dB 值视为静音
    frame_ms: 检测窗口大小
    """
    import librosa
    import numpy as np
    logger.info(f"[Pipeline] 预处理: 静音裁剪 (threshold={threshold_db}dB)")
    try:
        frame_length = int(sample_rate * frame_ms / 1000)
        hop_length = frame_length // 4

        trimmed, _ = librosa.effects.trim(
            audio_data,
            top_db=-threshold_db,
            frame_length=frame_length,
            hop_length=hop_length,
        )

        # 至少保留 0.5 秒，防止过度裁剪
        min_samples = sample_rate // 2
        if len(trimmed) < min_samples:
            logger.warning("[Pipeline] 裁剪后音频过短，保留原始")
            return audio_data

        logger.info(f"[Pipeline] 静音裁剪完成: {len(audio_data)} -> {len(trimmed)} samples")
        return trimmed.astype(np.float32)
    except Exception as e:
        logger.warning(f"[Pipeline] 静音裁剪失败，跳过: {e}")
        return audio_data


def peak_normalize(audio_data, target_db=-1.0):
    """峰值归一化到目标 dB"""
    import numpy as np
    target_linear = 10 ** (target_db / 20)
    peak = np.max(np.abs(audio_data))
    if peak > 1e-10:
        audio_data = audio_data / peak * target_linear
    return audio_data


# ─── 后处理 ───

def apply_highpass(audio_data, sample_rate, cutoff_hz=80, order=4):
    """
    高通滤波: 去除低频隆隆声 (80Hz 以下)
    """
    import numpy as np
    from scipy.signal import butter, sosfiltfilt
    logger.info(f"[Pipeline] 后处理: 高通滤波 ({cutoff_hz}Hz)")
    try:
        sos = butter(order, cutoff_hz, btype='highpass', fs=sample_rate, output='sos')
        result = sosfiltfilt(sos, audio_data)
        return result.astype(np.float32)
    except Exception as e:
        logger.warning(f"[Pipeline] 高通滤波失败: {e}")
        return audio_data


def apply_deessing(audio_data, sample_rate, threshold=0.3, freq_range=(4000, 12000)):
    """
    去齿音: 检测并衰减高频齿音 (sibilance)

    齿音主要集中在 5-10kHz 范围
    通过分析该频段的能量，动态衰减过量高频
    """
    import librosa
    import numpy as np
    logger.info(f"[Pipeline] 后处理: 去齿音 (threshold={threshold})")
    try:
        n_fft = 2048
        hop_length = 512

        stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

        # 找到齿音频段对应的频率bin
        sibilance_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        sibilance_bins = np.where(sibilance_mask)[0]

        if len(sibilance_bins) == 0:
            return audio_data

        # 计算齿音频段的能量
        sibilance_energy = np.mean(magnitude[sibilance_bins, :], axis=0)
        # 计算全频段平均能量作为参考
        avg_energy = np.mean(magnitude, axis=0) + 1e-10
        # 齿音比率
        sibilance_ratio = sibilance_energy / avg_energy

        # 对齿音过量的帧进行衰减
        gain_reduction = np.ones_like(magnitude)
        excess_mask = sibilance_ratio > threshold
        # 渐进式衰减: 超过 threshold 的部分按比例衰减
        excess_amount = np.clip(sibilance_ratio - threshold, 0, None)
        reduction_db = -6 * excess_amount  # 最大衰减 6dB 左右

        for bin_idx in sibilance_bins:
            gain_reduction[bin_idx, :] = 10 ** (reduction_db / 20)

        magnitude_processed = magnitude * gain_reduction
        stft_processed = magnitude_processed * np.exp(1j * phase)
        result = librosa.istft(stft_processed, hop_length=hop_length)

        if len(result) < len(audio_data):
            result = np.pad(result, (0, len(audio_data) - len(result)))
        else:
            result = result[:len(audio_data)]

        return result.astype(np.float32)
    except Exception as e:
        logger.warning(f"[Pipeline] 去齿音失败: {e}")
        return audio_data


def apply_eq(audio_data, sample_rate,
             low_gain_db=2.0, low_freq_hz=200,
             mid_gain_db=-1.0, mid_freq_hz=2500, mid_q=1.0,
             high_gain_db=3.0, high_freq_hz=8000):
    """
    EQ 频率响应修正:
    - Low shelf: 暖度 (200Hz 附近提升)
    - Mid peak: 去除 muddy 频率 (2.5kHz 附近微减)
    - High shelf: 存在感/空气感 (8kHz 附近提升)

    使用 pedalboard 的 shelving/peak 滤波器
    """
    logger.info(f"[Pipeline] 后处理: EQ (low={low_gain_db}dB, mid={mid_gain_db}dB, high={high_gain_db}dB)")
    try:
        import numpy as np
        from pedalboard import Pedalboard, LowShelfFilter, PeakFilter, HighShelfFilter

        board = Pedalboard([
            LowShelfFilter(cutoff_frequency_hz=low_freq_hz, gain_db=low_gain_db),
            PeakFilter(cutoff_frequency_hz=mid_freq_hz, gain_db=mid_gain_db, q=mid_q),
            HighShelfFilter(cutoff_frequency_hz=high_freq_hz, gain_db=high_gain_db),
        ])

        # pedalboard 需要 (samples,) 或 (samples, channels) 格式
        if audio_data.ndim == 1:
            processed = board(audio_data, sample_rate)
        else:
            processed = board(audio_data.T, sample_rate).T

        return processed.astype(np.float32)
    except Exception as e:
        logger.warning(f"[Pipeline] EQ 失败: {e}")
        return audio_data


def apply_compression(audio_data, sample_rate,
                      threshold_db=-18, ratio=3.0, attack_ms=5, release_ms=100):
    """
    动态范围压缩: 控制响度波动，使声音更均匀

    使用 pedalboard Compressor
    """
    logger.info(f"[Pipeline] 后处理: 压缩 (threshold={threshold_db}dB, ratio={ratio}:1)")
    try:
        import numpy as np
        from pedalboard import Pedalboard, Compressor

        board = Pedalboard([
            Compressor(
                threshold_db=threshold_db,
                ratio=ratio,
                attack_ms=attack_ms,
                release_ms=release_ms,
            ),
        ])

        if audio_data.ndim == 1:
            processed = board(audio_data, sample_rate)
        else:
            processed = board(audio_data.T, sample_rate).T

        return processed.astype(np.float32)
    except Exception as e:
        logger.warning(f"[Pipeline] 压缩失败: {e}")
        return audio_data


def apply_harmonic_enhancement(audio_data, sample_rate, amount=0.15):
    """
    谐波增强: 通过添加二次/三次谐波增加暖度和存在感

    原理: 对信号进行非线性处理 (soft clipping) 生成谐波
    amount: 谐波混合比例 (0=无增强, 0.3=明显增强)
    """
    import numpy as np
    logger.info(f"[Pipeline] 后处理: 谐波增强 (amount={amount})")
    try:
        # 软削波生成偶次谐波 (类似电子管饱和)
        harmonic = np.tanh(audio_data * 2.0) * 0.5
        result = audio_data * (1 - amount) + harmonic * amount

        return result.astype(np.float32)
    except Exception as e:
        logger.warning(f"[Pipeline] 谐波增强失败: {e}")
        return audio_data


def apply_limiter(audio_data, sample_rate, ceiling_db=-1.0, release_ms=50):
    """
    软限幅: 防止输出超过目标电平

    使用 pedalboard Limiter
    """
    import numpy as np
    logger.info(f"[Pipeline] 后处理: 限幅 (ceiling={ceiling_db}dB)")
    try:
        from pedalboard import Pedalboard, Limiter

        board = Pedalboard([
            Limiter(
                threshold_db=ceiling_db,
                release_ms=release_ms,
            ),
        ])

        if audio_data.ndim == 1:
            processed = board(audio_data, sample_rate)
        else:
            processed = board(audio_data.T, sample_rate).T

        return processed.astype(np.float32)
    except Exception as e:
        logger.warning(f"[Pipeline] 限幅失败: {e}")
        # Fallback: 硬限幅
        ceiling = 10 ** (ceiling_db / 20)
        return np.clip(audio_data, -ceiling, ceiling).astype(np.float32)


def normalize_loudness(audio_data, sample_rate, target_rms_db=-18.0):
    """
    响度归一化: 将 RMS 调整到目标水平 (近似 LUFS)

    target_rms_db: 目标 RMS 电平 (dB)
    - -18 dB RMS ≈ -16 LUFS (适合 vocal chop 在 DAW 中使用)
    - -14 dB RMS ≈ -12 LUFS (更响，适合混音中需要突出的元素)
    """
    import numpy as np
    logger.info(f"[Pipeline] 后处理: 响度归一化 (target={target_rms_db}dB RMS)")
    try:
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms < 1e-10:
            logger.warning("[Pipeline] RMS 过低，跳过响度归一化")
            return audio_data

        target_rms = 10 ** (target_rms_db / 20)
        gain = target_rms / rms

        # 限幅由后续 limiter 统一完成，避免峰值回退破坏目标响度。
        boosted = audio_data * gain
        return np.nan_to_num(boosted, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    except Exception as e:
        logger.warning(f"[Pipeline] 响度归一化失败: {e}")
        return audio_data


# ─── 完整管线 ───

def preprocess(audio_data, sample_rate,
               do_vocal_isolation=True,
               do_noise_reduction=True,
               do_dereverb=True,
               do_trim_silence=True,
               noise_reduction_strength=0.8,
               dereverb_strength=0.3):
    """
    预处理管线 (RVC 转换前执行)

    顺序: 人声分离 → 降噪 → 去混响 → 静音裁剪 → 峰值归一化

    所有步骤都有容错: 单个步骤失败不影响整体
    """
    import numpy as np
    logger.info("=" * 50)
    logger.info("[Pipeline] === 预处理开始 ===")
    logger.info(f"[Pipeline] 输入: shape={audio_data.shape}, sr={sample_rate}")

    result = audio_data.astype(np.float32).copy()

    if do_vocal_isolation:
        result = isolate_vocals(result, sample_rate)

    if do_noise_reduction and noise_reduction_strength > 0:
        result = reduce_noise(result, sample_rate, prop_decrease=noise_reduction_strength)

    if do_dereverb and dereverb_strength > 0:
        result = dereverb(result, sample_rate, decay=dereverb_strength)

    if do_trim_silence:
        result = trim_silence(result, sample_rate)

    # 始终做峰值归一化
    result = peak_normalize(result, target_db=-1.0)

    logger.info(f"[Pipeline] 预处理完成: shape={result.shape}")
    logger.info("=" * 50)
    return result


def postprocess(audio_data, sample_rate,
                do_highpass=True,
                do_deessing=True,
                do_eq=True,
                do_compression=True,
                do_harmonic_enhancement=True,
                do_limiter=True,
                do_loudness_normalize=True,
                target_rms_db=-18.0,
                eq_low_gain=2.0,
                eq_high_gain=3.0,
                harmonic_amount=0.15):
    """
    后处理管线 (RVC 转换后执行)

    顺序: 高通 → 去齿音 → EQ → 压缩 → 谐波增强 → 响度归一化 → 限幅

    所有步骤都有容错: 单个步骤失败不影响整体

    v15: do_deessing 和 do_harmonic_enhancement 可由调用方关闭
         用于发音保真模式 (避免吞字/失真)
    """
    import numpy as np
    logger.info("=" * 50)
    logger.info("[Pipeline] === 后处理开始 ===")
    logger.info(f"[Pipeline] 输入: shape={audio_data.shape}, sr={sample_rate}")

    result = audio_data.astype(np.float32).copy()

    if do_highpass:
        result = apply_highpass(result, sample_rate, cutoff_hz=80)

    if do_deessing:
        result = apply_deessing(result, sample_rate, threshold=0.3)

    if do_eq:
        result = apply_eq(
            result, sample_rate,
            low_gain_db=eq_low_gain,
            high_gain_db=eq_high_gain,
        )

    if do_compression:
        result = apply_compression(
            result, sample_rate,
            threshold_db=-18, ratio=3.0,
            attack_ms=5, release_ms=100,
        )

    if do_harmonic_enhancement and harmonic_amount > 0:
        result = apply_harmonic_enhancement(result, sample_rate, amount=harmonic_amount)

    if do_loudness_normalize:
        result = normalize_loudness(result, sample_rate, target_rms_db=target_rms_db)

    if do_limiter:
        result = apply_limiter(result, sample_rate, ceiling_db=-1.0)

    logger.info(f"[Pipeline] 后处理完成: shape={result.shape}")
    logger.info("=" * 50)
    return result
