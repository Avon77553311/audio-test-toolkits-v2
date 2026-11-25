"""
音频转 MIDI 工具
基于 librosa.pyin (概率 YIN) 基频追踪 + 音符分割

pYIN 只输出单一 f0 (基频)，从根源上解决泛音被误判为和弦的问题。
内存占用低 (~271MB)，适配 Render 512MB 限制。

算法流程:
1. 加载音频，降混为单声道
2. 逐帧提取基频 (f0) 和置信度 (pYIN)
3. 中值滤波 + 阈值过滤去除非语音帧
4. 将连续有语音帧分段，在音高变化 > onset_semitone 半音处切分音符边界
5. 每段量化为 MIDI 音符（取中位音高）
6. 过滤过短音符
7. 写入标准 MIDI 文件

参考:
  - pYIN: Mauch & Dixon, ICASSP 2014
  - CREPE Notes: Riley & Dixon, SMC 2023 (音符分割思路)
"""

import os
from pathlib import Path
from contextlib import nullcontext
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage

from tools import BaseTool, registry
from tools.audio_to_midi_options import validate_audio_to_midi_options
from tools.hybrid_ensemble import EnsembleCandidate, select_candidate
from tools.output_naming import safe_output_stem


# ─── 频率 ↔ MIDI 音符转换 ───


def freq_to_midi(freq):
    """频率 (Hz) → MIDI 音符编号 (浮点数)"""
    import numpy as np

    return 69 + 12 * np.log2(freq / 440.0)


def midi_to_freq(midi_note):
    """MIDI 音符编号 → 频率 (Hz)"""
    return 440.0 * 2 ** ((midi_note - 69) / 12.0)


def midi_note_name(note):
    """MIDI 音符编号 → 音名 (如 C4, F#5)"""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (note // 12) - 1
    return f"{names[note % 12]}{octave}"


# ─── 核心转换引擎 ───


class AudioToMidiConverter:
    """音频转 MIDI 引擎 (支持 librosa.pyin 单音与 CQT 多音)"""

    def __init__(
        self,
        hop_ms=10,
        fmin=50.0,
        fmax=1000.0,
        periodicity_threshold=0.21,
        median_filter_width=3,
        min_note_duration_ms=80,
        onset_semitone=0.8,
        pitch_quantize="median",
        bpm=120,
        onset_smooth_ms=30,
        transcription_profile="balanced",
        transcription_mode="mono",
        target_sample_rate=22050,
        quality_mode="balanced",
        use_runpod=False,
        cloud_view_provider=None,
        max_hpss_duration_sec=180.0,
        debug=False,
    ):
        """
        Args:
            hop_ms: 帧间隔 (毫秒)
            fmin: 最低检测频率 (Hz)
            fmax: 最高检测频率 (Hz)
            periodicity_threshold: 置信度阈值 (pYIN 专用)
            median_filter_width: 中值滤波窗口大小
            min_note_duration_ms: 最短音符持续时间 (毫秒)
            onset_semitone: 音符边界检测阈值 (半音)
            pitch_quantize: 音高量化方式
            bpm: 输出 MIDI 的 BPM
            onset_smooth_ms: onset 检测前平滑窗口
            transcription_profile: 'balanced'|'vocal_melisma'|'fast_guitar'
            transcription_mode: 'mono' (单音 pYIN) | 'poly' (多音 CQT)
            target_sample_rate: 加载目标采样率；会按 fmax 自动提高，避免截断高音
            debug: 是否输出调试信息
        """
        self.hop_ms = hop_ms
        self.fmin = fmin
        self.fmax = fmax
        self.periodicity_threshold = periodicity_threshold
        self.median_filter_width = median_filter_width
        self.min_note_duration_ms = min_note_duration_ms
        self.onset_semitone = onset_semitone
        self.pitch_quantize = pitch_quantize
        self.bpm = bpm
        self.onset_smooth_ms = onset_smooth_ms
        self.transcription_profile = transcription_profile
        self.transcription_mode = transcription_mode
        self.target_sample_rate = target_sample_rate
        self.quality_mode = quality_mode
        self.use_runpod = use_runpod is True
        self.cloud_view_provider = cloud_view_provider
        self.max_hpss_duration_sec = float(max_hpss_duration_sec)
        self.debug = debug

    def convert(self, input_path, output_path):
        """执行完整转换；质量优先时运行多视图候选并保守重排。"""
        import librosa

        audio_data, sample_rate = self._load_audio(input_path)
        self._validate_parameters(sample_rate)
        hop_length = max(1, int(round(sample_rate * self.hop_ms / 1000)))
        duration_sec = len(audio_data) / sample_rate
        if self.debug:
            print(
                f"[AudioToMIDI] Mode: {self.transcription_mode}, quality={self.quality_mode}, "
                f"Audio: {len(audio_data) / sample_rate:.2f}s, sr={sample_rate}, hop={hop_length}"
            )

        notes, model_name, voiced_ratio = self._extract_candidate(
            audio_data,
            sample_rate,
            hop_length,
            self.transcription_mode,
            force_onsets=self.transcription_profile == "fast_guitar",
        )
        baseline = EnsembleCandidate(
            "baseline",
            notes,
            self._notes_quality(notes, len(audio_data) / sample_rate),
            self.transcription_mode,
            "original",
        )
        decision = select_candidate([baseline], "baseline", self._notes_similarity)
        views = ["original"]
        experts = [self.transcription_mode]
        failures = {}
        skipped_views = {}
        cloud_used = False
        cloud_attempted = False
        cloud_stem = None

        if self.quality_mode == "high":
            candidates = [baseline]
            if duration_sec > self.max_hpss_duration_sec:
                skipped_views["harmonic"] = (
                    f"audio_too_long_for_full_hpss:{duration_sec:.2f}s>"
                    f"{self.max_hpss_duration_sec:.2f}s"
                )
            else:
                try:
                    harmonic = librosa.effects.harmonic(audio_data, margin=3.0)
                    harmonic_notes, harmonic_model, _ = self._extract_candidate(
                        harmonic,
                        sample_rate,
                        hop_length,
                        self.transcription_mode,
                        force_onsets=True,
                    )
                    candidates.append(
                        EnsembleCandidate(
                            "harmonic",
                            harmonic_notes,
                            self._notes_quality(harmonic_notes, duration_sec),
                            self.transcription_mode,
                            "harmonic",
                        )
                    )
                    views.append("harmonic")
                    model_name = f"{model_name} + {harmonic_model}"
                except Exception as exc:
                    failures["harmonic"] = str(exc) or type(exc).__name__

            alternate = "poly" if self.transcription_mode == "mono" else "mono"
            try:
                onset_notes, onset_model, _ = self._extract_candidate(
                    audio_data, sample_rate, hop_length, alternate, force_onsets=True
                )
                candidates.append(
                    EnsembleCandidate(
                        "onset_auxiliary",
                        onset_notes,
                        self._notes_quality(onset_notes, len(audio_data) / sample_rate),
                        alternate,
                        "onset_auxiliary",
                    )
                )
                views.append("onset_auxiliary")
                experts.append(alternate)
                model_name = f"{model_name} + {onset_model}"
            except Exception as exc:
                failures["onset_auxiliary"] = str(exc) or type(exc).__name__

            if self.use_runpod and self.cloud_view_provider is not None:
                cloud_attempted = True
                try:
                    cloud_view = self.cloud_view_provider(
                        input_path, purpose="midi", sample_rate=sample_rate
                    )
                    cloud_stem = getattr(cloud_view, "stem", None)
                    cloud_path = os.fspath(getattr(cloud_view, "path", cloud_view))
                    cloud_audio, _ = librosa.load(cloud_path, sr=sample_rate, mono=True)
                    cloud_notes, cloud_model, _ = self._extract_candidate(
                        cloud_audio,
                        sample_rate,
                        hop_length,
                        self.transcription_mode,
                        force_onsets=True,
                    )
                    candidates.append(
                        EnsembleCandidate(
                            "cloud_stem",
                            cloud_notes,
                            self._notes_quality(
                                cloud_notes, len(audio_data) / sample_rate
                            ),
                            self.transcription_mode,
                            "cloud_stem",
                        )
                    )
                    views.append("cloud_stem")
                    cloud_used = True
                    model_name = f"{model_name} + {cloud_model}"
                except Exception as exc:
                    raise RuntimeError(
                        f"RunPod GPU Stem 处理失败: {str(exc) or type(exc).__name__}"
                    ) from exc
            decision = select_candidate(
                candidates,
                "baseline",
                self._notes_similarity,
                failures=failures,
                fallback_on_any_failure=True,
                minimum_gain=0.025,
            )
            notes = decision.value

        if not notes:
            raise ValueError(
                "未识别到有效音符，请检查音频内容、识别模式或降低激活阈值"
            )
        self._write_midi(notes, output_path, sample_rate, hop_length)
        pitch_range = (
            (min(n["midi"] for n in notes), max(n["midi"] for n in notes))
            if notes
            else (0, 0)
        )
        ensemble = {
            "quality_mode": self.quality_mode,
            "views": views,
            "experts": experts,
            "chosen": decision.chosen,
            "fallback": decision.fallback,
            "reason": decision.reason,
            "runpod_requested": self.use_runpod,
            "cloud_view_attempted": cloud_attempted,
            "cloud_view_used": cloud_used,
            "cloud_stem": cloud_stem,
            "gpu_view_used": cloud_used,
            "skipped_views": skipped_views,
            "gpu_routing": (
                "RunPod GPU model is fixed by Endpoint deployment; "
                "quality mode does not switch GPU per request."
            ),
            "trace": decision.trace,
        }
        return {
            "total_notes": len(notes),
            "notes": notes,
            "pitch_range": (
                f"{midi_note_name(pitch_range[0])} - {midi_note_name(pitch_range[1])}"
                if notes
                else "N/A"
            ),
            "duration_sec": round(len(audio_data) / sample_rate, 2),
            "sample_rate": sample_rate,
            "hop_ms": self.hop_ms,
            "transcription_profile": self.transcription_profile,
            "transcription_mode": self.transcription_mode,
            "model": model_name,
            "voiced_ratio": voiced_ratio,
            "ensemble": ensemble,
            "warning": "Polyphonic transcription is an estimation and may not reach professional DAW accuracy."
            if self.transcription_mode == "poly"
            else None,
        }

    def _extract_candidate(
        self, audio_data, sample_rate, hop_length, mode, force_onsets=False
    ):
        import numpy as np

        if mode == "poly":
            notes, model = self._extract_polyphonic(audio_data, sample_rate, hop_length)
            frame_count = max(1, len(audio_data) // hop_length)
            active = sum(
                int(n["duration_sec"] * sample_rate / hop_length) for n in notes
            )
            return notes, model, round(min(1.0, active / frame_count), 3)
        frame_length = max(
            2048, 1 << (int(np.ceil(2 * sample_rate / self.fmin)) - 1).bit_length()
        )
        f0, conf, model = self._extract_pyin(
            audio_data, sample_rate, hop_length, frame_length
        )
        onsets = (
            self._detect_instrument_onsets(audio_data, sample_rate, hop_length)
            if force_onsets
            else None
        )
        notes = self._segment_notes(
            f0, conf, hop_length, sample_rate, onset_frames=onsets
        )
        voiced = round(float(np.sum(~np.isnan(f0)) / len(f0)), 3) if len(f0) else 0
        return notes, model, voiced

    @staticmethod
    def _notes_quality(notes, duration):
        if not notes or duration <= 0:
            return 0.0
        coverage = min(
            1.0, sum(max(0.0, n.get("duration_sec", 0)) for n in notes) / duration
        )
        confidence = sum(min(1.0, n.get("velocity", 0) / 127.0) for n in notes) / len(
            notes
        )
        density_penalty = max(0.0, len(notes) / duration - 24.0) / 24.0
        return max(
            0.0, min(1.0, 0.58 * confidence + 0.42 * coverage - 0.15 * density_penalty)
        )

    @staticmethod
    def _notes_similarity(left, right):
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        matches = 0
        for note in left:
            if any(
                abs(note["onset_sec"] - other["onset_sec"]) <= 0.08
                and abs(note["midi"] - other["midi"]) <= 1
                for other in right
            ):
                matches += 1
        return 2.0 * matches / max(1, len(left) + len(right))

    def _extract_polyphonic(self, audio_data, sample_rate, hop_length):
        """使用 librosa CQT 进行多音转录，结合频率轴局部峰筛选与频谱泄漏抑制。"""
        import librosa
        import numpy as np
        import scipy.ndimage

        # 1. 仅构建用户音域内的半音 CQT 频率轴。
        first_midi = max(0, int(np.ceil(freq_to_midi(self.fmin))))
        last_midi = min(127, int(np.floor(freq_to_midi(self.fmax))))
        if first_midi > last_midi:
            return [], "librosa.cqt (Polyphonic)"
        cqt_fmin = midi_to_freq(first_midi)
        n_bins = last_midi - first_midi + 1
        C = np.abs(
            librosa.cqt(
                audio_data.astype(np.float32),
                sr=sample_rate,
                hop_length=hop_length,
                fmin=cqt_fmin,
                n_bins=n_bins,
            )
        )

        # 2. 时间轴中值滤波，抑制瞬态噪声
        C = scipy.ndimage.median_filter(C, size=(1, 3))

        # 3. 频率轴局部峰检测与邻近 bin 抑制 (抑制谱泄漏)
        # 对每一帧找局部最大值，且显著高于其两侧邻居
        C_refined = np.zeros_like(C)
        for t in range(C.shape[1]):
            frame = C[:, t]
            if np.max(frame) == 0:
                continue
            # 边界 bin 也可能是用户音域内的真实音符，不能因缺少一侧邻居丢弃。
            for idx in range(len(frame)):
                val = frame[idx]
                left = frame[idx - 1] if idx > 0 else -np.inf
                right = frame[idx + 1] if idx + 1 < len(frame) else -np.inf
                if (
                    val > left
                    and val > right
                    and val > np.max(frame) * 0.18
                ):
                    C_refined[idx, t] = val

        # 4. 全局阈值
        global_max = np.max(C_refined)
        if global_max == 0:
            return [], "librosa.cqt (Polyphonic)"

        # 多音模式：阈值表示相对于整段最强 CQT 峰值的最低激活比例。
        threshold = global_max * self.periodicity_threshold
        mask = C_refined > threshold

        notes = []
        frame_sec = hop_length / sample_rate
        min_duration_frames = max(
            2,
            int(np.ceil(self.min_note_duration_ms / (frame_sec * 1000))),
        )

        for pitch_idx in range(C_refined.shape[0]):
            midi_note = int(first_midi + pitch_idx)
            active = mask[pitch_idx]

            # 找到连续激活段
            diff = np.diff(active.astype(int))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0] + 1
            if active[0]:
                starts = np.insert(starts, 0, 0)
            if active[-1]:
                ends = np.append(ends, len(active))

            for s, e in zip(starts, ends):
                if e - s < min_duration_frames:
                    continue

                mag = np.max(C[pitch_idx, s:e])
                velocity = int(min(127, (mag / global_max) * 127 * 1.5))

                notes.append(
                    {
                        "midi": midi_note,
                        "velocity": max(10, velocity),  # 给予一定的最小力度
                        "onset_sec": float(s * frame_sec),
                        "duration_sec": float((e - s) * frame_sec),
                        "frame": int(s),
                        "duration_frames": int(e - s),
                    }
                )

        notes.sort(key=lambda x: x["onset_sec"])
        return notes, "librosa.cqt (Polyphonic)"

    # ─── pYIN 基频提取 ───

    def _extract_pyin(self, audio_data, sample_rate, hop_length, frame_length):
        """使用 librosa.pyin (概率 YIN) 提取 f0 和置信度"""
        import librosa
        import numpy as np
        from scipy.ndimage import median_filter

        # 确保输入是 float32 且连续，减少内存拷贝
        y = np.ascontiguousarray(audio_data, dtype=np.float32)

        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
            fill_na=np.nan,
        )

        # pYIN 返回的 voiced_prob 范围 0~1，作为置信度
        conf = voiced_prob

        # 无语音帧的 f0 设为 NaN
        f0 = np.where(voiced_flag, f0, np.nan)

        # 阈值过滤
        conf_mask = ~np.isfinite(conf) | (conf < self.periodicity_threshold)
        f0[conf_mask] = np.nan

        # 仅在连续有声段内滤波，避免把静音两侧互不相关的音高拼接到同一窗口。
        voiced_mask = np.isfinite(f0)
        transitions = np.diff(np.pad(voiced_mask.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        for start, end in zip(starts, ends):
            segment_length = end - start
            kernel = int(self.median_filter_width)
            if kernel % 2 == 0:
                kernel += 1
            kernel = min(kernel, segment_length if segment_length % 2 else segment_length - 1)
            if kernel >= 3:
                f0[start:end] = median_filter(
                    f0[start:end].astype(np.float64),
                    size=kernel,
                    mode="nearest",
                )

        return f0, conf, "librosa.pyin (pYIN)"

    def _detect_instrument_onsets(self, audio_data, sample_rate, hop_length):
        """检测快速拨弦/击弦起音，用于把同音高的连续拨弦拆成多个 MIDI 音符。"""
        import librosa
        import numpy as np

        y = np.ascontiguousarray(audio_data, dtype=np.float32)
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sample_rate, hop_length=hop_length
        )
        if onset_env.size == 0 or not np.isfinite(onset_env).any():
            return set()
        # 快速吉他需要较高召回，但避免把普通包络抖动全部切开。
        frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sample_rate,
            hop_length=hop_length,
            units="frames",
            backtrack=True,
            pre_max=2,
            post_max=2,
            pre_avg=4,
            post_avg=4,
            delta=0.08,
            wait=1,
        )
        return {int(frame) for frame in frames}

    # ─── 音符分割 ───

    def _segment_notes(self, f0, conf, hop_length, sample_rate, onset_frames=None):
        """
        将 f0 序列分割为离散音符。

        算法：
        1. 找到连续的有语音帧段 (voiced segments)
        2. 在每个有语音段内，检测音高变化 > onset_semitone 半音的位置作为音符边界
        3. 对每个音符段，量化为 MIDI 音高（取中位数）
        4. 计算音符的 onset/offset 时间和 velocity
        5. 过滤过短的音符
        """
        import numpy as np

        if self.transcription_profile == "vocal_melisma":
            return self._segment_vocal_melisma(f0, conf, hop_length, sample_rate)
        onset_frames = set(onset_frames or [])

        n_frames = len(f0)
        if n_frames == 0:
            return []

        frame_sec = hop_length / sample_rate
        min_duration_frames = max(
            1, int(np.ceil(self.min_note_duration_ms / self.hop_ms))
        )

        notes = []

        voiced = ~np.isnan(f0)
        if not voiced.any():
            return []

        voiced_diff = np.diff(voiced.astype(int))
        segment_starts = np.where(voiced_diff == 1)[0] + 1
        segment_ends = np.where(voiced_diff == -1)[0] + 1

        if voiced[0]:
            segment_starts = np.insert(segment_starts, 0, 0)
        if voiced[-1]:
            segment_ends = np.append(segment_ends, n_frames)

        for seg_start, seg_end in zip(segment_starts, segment_ends):
            if seg_end <= seg_start:
                continue

            seg_f0 = f0[seg_start:seg_end]
            seg_conf = conf[seg_start:seg_end]

            if len(seg_f0) < min_duration_frames:
                continue

            seg_midi = freq_to_midi(seg_f0)

            smooth_frames = max(1, int(self.onset_smooth_ms / self.hop_ms))
            if smooth_frames > 1 and len(seg_midi) > smooth_frames:
                kernel = np.ones(smooth_frames) / smooth_frames
                left_pad = (smooth_frames - 1) // 2
                right_pad = smooth_frames // 2
                padded_midi = np.pad(seg_midi, (left_pad, right_pad), mode="edge")
                seg_midi_smooth = np.convolve(padded_midi, kernel, mode="valid")
            else:
                seg_midi_smooth = seg_midi

            boundaries = [0]
            # ─── 换音边界检测 (Item 1) ───
            # pYIN 有时会将连续的阶梯音高粘在一起。使用 run-length 逻辑检测显著音高变化。
            last_stable_midi = int(round(seg_midi_smooth[0]))
            stable_count = 0
            for i in range(1, len(seg_midi_smooth)):
                current_midi = int(round(seg_midi_smooth[i]))
                pitch_diff = abs(seg_midi_smooth[i] - seg_midi_smooth[i - 1])

                # 显著瞬时跳变 (> 0.8 st) 直接切分
                if pitch_diff > self.onset_semitone:
                    boundaries.append(i)
                    last_stable_midi = current_midi
                    stable_count = 0
                # 渐进式换音：音级改变且在新音级上保持了至少 3 帧 (约 30ms)
                elif current_midi != last_stable_midi:
                    stable_count += 1
                    if stable_count >= 3:
                        # 在变化开始处切分（回溯稳定计数的起点）
                        boundaries.append(i - 2)
                        last_stable_midi = current_midi
                        stable_count = 0
                else:
                    stable_count = 0

                abs_frame = seg_start + i
                if abs_frame in onset_frames:
                    # 强起音边界 (fast_guitar)
                    boundaries.append(i)
                    last_stable_midi = current_midi
                    stable_count = 0

            boundaries.append(len(seg_midi))
            boundaries = sorted(set(b for b in boundaries if 0 <= b <= len(seg_midi)))

            for j in range(len(boundaries) - 1):
                note_start = boundaries[j]
                note_end = boundaries[j + 1]
                note_len = note_end - note_start

                if note_len < min_duration_frames:
                    continue

                note_f0 = seg_f0[note_start:note_end]
                note_conf = seg_conf[note_start:note_end]
                note_midi = seg_midi[note_start:note_end]

                if self.pitch_quantize == "median":
                    midi_val = np.nanmedian(note_midi)
                elif self.pitch_quantize == "mean":
                    midi_val = np.nanmean(note_midi)
                else:
                    rounded = np.round(note_midi).astype(int)
                    vals, counts = np.unique(rounded, return_counts=True)
                    midi_val = vals[np.argmax(counts)]

                midi_note = int(round(midi_val))
                if not (0 <= midi_note <= 127):
                    continue

                abs_start = seg_start + note_start
                abs_end = seg_start + note_end

                onset_sec = abs_start * frame_sec
                duration_sec = note_len * frame_sec

                velocity = max(1, min(127, int(np.nanmean(note_conf) * 127)))

                notes.append(
                    {
                        "midi": int(midi_note),
                        "velocity": int(velocity),
                        "onset_sec": float(onset_sec),
                        "duration_sec": float(duration_sec),
                        "frame": int(abs_start),
                        "duration_frames": int(note_len),
                    }
                )

        notes.sort(key=lambda x: x["onset_sec"])
        return notes

    def _segment_vocal_melisma(self, f0, conf, hop_length, sample_rate):
        """以稳定音级路径切分人声转音，而不是依赖相邻帧瞬时跳变。"""
        import numpy as np
        from scipy.signal import medfilt

        n_frames = len(f0)
        if n_frames == 0:
            return []

        frame_sec = hop_length / sample_rate
        min_frames = max(1, int(np.ceil(self.min_note_duration_ms / self.hop_ms)))
        stable_frames = max(2, int(np.ceil(35 / self.hop_ms)))
        voiced = np.isfinite(f0)
        if not voiced.any():
            return []

        changes = np.diff(voiced.astype(int))
        starts = list(np.where(changes == 1)[0] + 1)
        ends = list(np.where(changes == -1)[0] + 1)
        if voiced[0]:
            starts.insert(0, 0)
        if voiced[-1]:
            ends.append(n_frames)

        notes = []
        for seg_start, seg_end in zip(starts, ends):
            if seg_end - seg_start < min_frames:
                continue
            observed = freq_to_midi(f0[seg_start:seg_end])

            # 仅在当前连续发声段内做局部中值，避免跨静音/换气互相污染。
            width = max(1, int(round(15 / self.hop_ms)))
            if width % 2 == 0:
                width += 1
            if width > 1 and len(observed) >= width:
                pad = width // 2
                smooth = medfilt(np.pad(observed, (pad, pad), mode="edge"), width)[
                    pad:-pad
                ]
            else:
                smooth = observed.copy()

            seed_frames = min(
                len(smooth), max(stable_frames, int(np.ceil(50 / self.hop_ms)))
            )
            current = int(round(np.median(smooth[:seed_frames])))
            states = np.full(len(smooth), current, dtype=int)
            candidate = None
            candidate_start = 0
            candidate_count = 0

            for i, value in enumerate(smooth):
                # 0.60 st 进入、0.38 st 退出形成滞回；持续 35 ms 才确认新音级。
                target = int(round(value))
                if target != current and abs(value - current) >= 0.60:
                    if target == candidate:
                        candidate_count += 1
                    else:
                        candidate = target
                        candidate_start = i
                        candidate_count = 1
                    if candidate_count >= stable_frames:
                        states[candidate_start : i + 1] = target
                        current = target
                        candidate = None
                        candidate_count = 0
                elif abs(value - current) <= 0.38:
                    candidate = None
                    candidate_count = 0
                states[i] = current

            cuts = np.r_[0, np.where(np.diff(states) != 0)[0] + 1, len(states)]
            runs = [
                [int(cuts[i]), int(cuts[i + 1]), int(states[cuts[i]])]
                for i in range(len(cuts) - 1)
            ]

            # A-B-A 且 B 很短通常是颤音/量化边界抖动，恢复成一个持续音。
            changed = True
            while changed and len(runs) >= 3:
                changed = False
                for i in range(1, len(runs) - 1):
                    middle = runs[i]
                    if runs[i - 1][2] == runs[i + 1][2] and middle[1] - middle[
                        0
                    ] <= int(np.ceil(120 / self.hop_ms)):
                        runs[i - 1][1] = runs[i + 1][1]
                        del runs[i : i + 2]
                        changed = True
                        break

            # 不能独立成立的短段吸收到音高更接近的邻段，而不是留下缺口。
            changed = True
            while changed and len(runs) > 1:
                changed = False
                for i, run in enumerate(runs):
                    if run[1] - run[0] >= min_frames:
                        continue
                    if i == 0:
                        runs[1][0] = run[0]
                    elif i == len(runs) - 1:
                        runs[i - 1][1] = run[1]
                    elif abs(run[2] - runs[i - 1][2]) <= abs(run[2] - runs[i + 1][2]):
                        runs[i - 1][1] = run[1]
                    else:
                        runs[i + 1][0] = run[0]
                    runs.pop(i)
                    changed = True
                    break

            for note_start, note_end, path_pitch in runs:
                if note_end - note_start < min_frames:
                    continue
                note_midi = observed[note_start:note_end]
                # 路径音级决定离散音符；中位数只用于防止异常路径偏离观测。
                observed_pitch = int(round(np.nanmedian(note_midi)))
                midi_note = (
                    path_pitch
                    if abs(path_pitch - observed_pitch) <= 1
                    else observed_pitch
                )
                if not 0 <= midi_note <= 127:
                    continue
                abs_start = seg_start + note_start
                duration_frames = note_end - note_start
                note_conf = conf[abs_start : seg_start + note_end]
                notes.append(
                    {
                        "midi": int(midi_note),
                        "velocity": int(
                            max(1, min(127, int(np.nanmean(note_conf) * 127)))
                        ),
                        "onset_sec": float(abs_start * frame_sec),
                        "duration_sec": float(duration_frames * frame_sec),
                        "frame": int(abs_start),
                        "duration_frames": int(duration_frames),
                    }
                )

        notes.sort(key=lambda item: item["onset_sec"])
        return notes

    # ─── MIDI 写入 ───

    def _write_midi(self, notes, output_path, sample_rate, hop_length):
        """写入 MIDI 文件"""
        mid = MidiFile(ticks_per_beat=480)
        track = MidiTrack()
        mid.tracks.append(track)

        tempo = mido.bpm2tempo(self.bpm)
        track.append(MetaMessage("set_tempo", tempo=tempo, time=0))
        track.append(MetaMessage("track_name", name="Audio Transcription", time=0))

        events = []
        for note in notes:
            start_tick = int(note["onset_sec"] * mid.ticks_per_beat * self.bpm / 60)
            duration_ticks = max(
                1, int(note["duration_sec"] * mid.ticks_per_beat * self.bpm / 60)
            )
            events.append((start_tick, "note_on", note["midi"], note["velocity"]))
            events.append((start_tick + duration_ticks, "note_off", note["midi"], 0))

        events.sort(key=lambda x: (x[0], 0 if x[1] == "note_off" else 1))

        last_tick = 0
        for tick, event_type, note, velocity in events:
            delta = max(0, tick - last_tick)
            track.append(Message(event_type, note=note, velocity=velocity, time=delta))
            last_tick = tick

        mid.save(output_path)

    # ─── 音频加载与参数校验 ───

    def _validate_parameters(self, sample_rate):
        import math

        numeric_values = {
            "帧间隔": self.hop_ms,
            "激活阈值": self.periodicity_threshold,
            "最低频率": self.fmin,
            "最高频率": self.fmax,
            "最短音符时长": self.min_note_duration_ms,
            "BPM": self.bpm,
        }
        if any(not math.isfinite(float(value)) for value in numeric_values.values()):
            raise ValueError("音频转 MIDI 数值参数必须是有限数")
        if not 1 <= self.hop_ms <= 500:
            raise ValueError("时间窗口（hop_length 间隔）必须在 1~500 ms 之间")
        if not 0 <= self.periodicity_threshold <= 1:
            raise ValueError("激活阈值必须在 0~1 之间")
        if not 0 < self.fmin < self.fmax <= sample_rate / 2:
            raise ValueError(
                f"音域必须满足 0 < 最低频率 < 最高频率 <= Nyquist 频率 "
                f"({sample_rate / 2:.1f} Hz)"
            )
        if self.fmin < midi_to_freq(0) or self.fmax > midi_to_freq(127):
            raise ValueError("音域必须在 MIDI 0~127 可表示的频率范围内")
        if self.transcription_profile not in {
            "balanced",
            "vocal_melisma",
            "fast_guitar",
        }:
            raise ValueError("转录模式仅支持 balanced、vocal_melisma 或 fast_guitar")
        if self.transcription_mode not in {"mono", "poly"}:
            raise ValueError("识别模式仅支持 mono 或 poly")
        if self.quality_mode not in {"balanced", "high"}:
            raise ValueError("quality_mode 仅支持 balanced 或 high")
        if self.use_runpod and self.quality_mode != "high":
            raise ValueError("使用 RunPod GPU Stem 必须选择质量优先模式")
        if self.use_runpod and self.cloud_view_provider is None:
            raise ValueError("服务端 RunPod GPU Stem 未启用或协议不是 demucs-v1")
        if self.min_note_duration_ms <= 0:
            raise ValueError("最短音符时长必须大于 0 ms")
        if self.target_sample_rate is not None and self.target_sample_rate <= 0:
            raise ValueError("目标采样率必须大于 0")
        if not 1 <= self.bpm <= 999:
            raise ValueError("BPM 必须在 1~999 之间")

    def _load_audio(self, input_path):
        """加载单声道音频，并限制分析采样率以控制 pYIN 峰值内存。

        目标采样率始终为最高检测频率预留 10% Nyquist 余量，因此调整
        ``fmax`` 到高音范围时不会被固定的 22.05 kHz 采样率截断。
        """
        import math
        import librosa

        target_sr = None
        if self.target_sample_rate is not None:
            target_sr = max(
                int(self.target_sample_rate),
                int(math.ceil(self.fmax * 2.2)),
            )
        try:
            audio_data, sample_rate = librosa.load(input_path, sr=target_sr, mono=True)
        except Exception as e:
            # 捕获异常 (如文件损坏、空文件或格式不支持)
            raise ValueError(
                f"无法读取音频文件 (可能已损坏或格式不支持): {str(e) or type(e).__name__}"
            ) from e

        if audio_data.size == 0:
            raise ValueError("音频内容为空")
        return audio_data, sample_rate


# ─── 注册为工具 ───


class AudioToMidiTool(BaseTool):
    name = "audio-to-midi"
    display_name = "音频转 MIDI"
    description = (
        "将音频文件 (WAV/MP3/FLAC/OGG) 转换为 MIDI 文件，基于 pYIN 概率基频追踪"
    )
    short_desc = "基频追踪 · 单声转录"
    accepted_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"]

    def process(self, input_path, output_dir, **kwargs):
        """Public API: isolate librosa/pYIN in a short-lived child process."""
        from tools.audio_to_midi_worker import run_audio_to_midi_subprocess

        worker_timeout = kwargs.pop("_worker_timeout", None)
        cancel_event = kwargs.pop("_cancel_event", None)
        context = kwargs.pop("_task_context", None)
        call_kwargs = {}
        if worker_timeout is not None:
            call_kwargs["timeout"] = float(worker_timeout)
        if cancel_event is not None:
            from task_scheduler import notify_runpod_started

            call_kwargs["cancel_event"] = cancel_event
            call_kwargs["on_runpod_started"] = notify_runpod_started
        if context:
            call_kwargs["on_progress"] = lambda p, m: context.update(
                progress=p, message=m
            )

        output_path = Path(output_dir).resolve()
        for parent in (output_path, *output_path.parents):
            if parent.name in {"midi_tasks", "user_files"}:
                kwargs["_stem_callback_root"] = os.fspath(parent.parent)
                break

        return run_audio_to_midi_subprocess(
            input_path, output_dir, kwargs, **call_kwargs
        )

    def _process_direct(self, input_path, output_dir, **kwargs):
        """Conversion implementation used only by the isolated worker."""
        import time as _time
        from tools.audio_to_midi_worker import PROGRESS_PREFIX
        import json

        _t0 = _time.time()

        def report_progress(p, m):
            print(
                f"{PROGRESS_PREFIX}{json.dumps({'progress': p, 'message': m})}",
                flush=True,
            )

        report_progress(5, "正在初始化转换引擎")
        callback_root = kwargs.pop("_stem_callback_root", None)
        if callback_root:
            from tools.stem_callback import stem_callback_store

            stem_callback_store.configure(callback_root)

        options = validate_audio_to_midi_options(kwargs)
        profile = options.get("transcription_profile", "balanced")
        mode = options.get("transcription_mode", "mono")
        quality_mode = options.get("quality_mode", "balanced")
        use_runpod = kwargs.get("use_runpod", False)
        if not isinstance(use_runpod, bool):
            raise ValueError("use_runpod 必须是布尔值")
        if use_runpod and quality_mode != "high":
            raise ValueError("使用 RunPod GPU Stem 必须选择质量优先模式")

        provider = None
        if quality_mode == "high" and use_runpod:
            from tools.cloud_view_provider import create_cloud_stem_view_provider

            provider = create_cloud_stem_view_provider()
            if provider is None:
                raise ValueError("服务端 RunPod GPU Stem 未启用或协议不是 demucs-v1")
        provider_context = provider if provider is not None else nullcontext(None)
        with provider_context as active_provider:
            converter = AudioToMidiConverter(
                # 新 UI key 优先；继续兼容旧调用方的 hop_ms/periodicity_threshold。
                hop_ms=options.get("time_window_ms", 10),
                fmin=options.get("fmin", 50.0),
                fmax=options.get("fmax", 1000.0),
                periodicity_threshold=options.get("activation_level", 0.21),
                min_note_duration_ms=options.get("min_note_duration_ms", 80),
                onset_semitone=options.get("onset_semitone", 0.8),
                transcription_profile=profile,
                transcription_mode=mode,
                target_sample_rate=options.get("target_sample_rate", 22050),
                quality_mode=quality_mode,
                use_runpod=use_runpod,
                cloud_view_provider=active_provider,
                bpm=options.get("bpm", 120),
                debug=True,
            )

            base_name = safe_output_stem(
                kwargs.get("original_filename") or input_path
            )
            output_filename = f"{base_name}.mid"
            output_path = os.path.join(output_dir, output_filename)
            stats = converter.convert(input_path, output_path)
        _elapsed = round(_time.time() - _t0, 2)
        print(f"[AudioToMIDI] 总处理时间: {_elapsed}s", flush=True)
        stats["elapsed_sec"] = _elapsed

        return {
            "success": True,
            "output_files": [output_path],
            "message": f"转换完成！共识别 {stats['total_notes']} 个音符 ({stats['model']})",
            "metadata": stats,
        }


# 自动注册
registry.register(AudioToMidiTool())
