"""Validate model audio before publishing a successful task."""

import numpy as np
import soundfile as sf


def validate_stem_outputs(paths, source):
    from audio_admission import probe_audio_duration

    source_duration = probe_audio_duration(source)
    signature = None
    has_signal = False
    for path in paths:
        try:
            with sf.SoundFile(path) as stream:
                current = (stream.samplerate, stream.channels, len(stream))
                if not len(stream) or abs(len(stream) / stream.samplerate - source_duration) > 0.05:
                    raise ValueError("Stem 轨道时长与源音频不一致")
                if signature is not None and current != signature:
                    raise ValueError("Stem 轨道采样率、声道或帧数不一致")
                signature = current
                for block in stream.blocks(blocksize=65536, dtype="float32"):
                    if not np.all(np.isfinite(block)):
                        raise ValueError("Stem 轨道含无效采样值")
                    has_signal = has_signal or bool(np.max(np.abs(block)) > 1e-6)
        except (RuntimeError, OSError) as exc:
            raise ValueError("Stem 返回了无法解码的音频") from exc
    if not has_signal:
        raise ValueError("Stem 结果全部为静音，未生成有效音轨")
