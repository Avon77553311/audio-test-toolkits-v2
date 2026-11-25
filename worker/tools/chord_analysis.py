"""CPU 友好的 CQT 和弦分析：HPSS、三和弦模板匹配与时序分段。"""

import csv
import io
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path

import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack

from tools import BaseTool, registry
from tools.bass_analysis import BassAnalyzer
from tools.chord_analysis_options import validate_chord_analysis_options
from tools.chord_notation import analyze_notation_timeline
from tools.chord_segmentation import analyze_with_segmentation
from tools.hybrid_ensemble import EnsembleCandidate, select_candidate
from tools.mix_analysis import MIX_QUALITIES, MixChordAnalyzer
from tools.neural_chord_analysis import NeuralChordAnalyzer
from tools.output_naming import safe_output_stem
from tools.runtime_resources import runtime_directory

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
HOP_LENGTH = 512
TARGET_SR = 22050


def _get_c1():
    import librosa

    return librosa.note_to_hz("C1")


C1_HZ = 32.70319566257483  # Pre-calculated to avoid librosa import at startup
BASE_QUALITIES = {
    "maj": ((0, 4, 7), (1.35, 1.0, 0.85)),
    "min": ((0, 3, 7), (1.35, 1.0, 0.85)),
    "dim": ((0, 3, 6), (1.35, 1.0, 0.85)),
    "aug": ((0, 4, 8), (1.35, 1.0, 0.85)),
    "sus2": ((0, 2, 7), (1.30, 1.0, 0.85)),
    "sus4": ((0, 5, 7), (1.30, 1.0, 0.85)),
}
EXTENSIONS = {
    "7": ("maj", 10),
    "maj7": ("maj", 11),
    "min7": ("min", 10),
    "6": ("maj", 9),
    "m6": ("min", 9),
    "add9": ("maj", 14),
    "maj9": ("maj7", 14),
    "min9": ("min7", 14),
    "9": ("7", 14),
    "11": ("9", 17),
    "13": ("11", 21),
}
QUALITY_SUFFIX = {
    "maj": "",
    "min": "m",
    "dim": "dim",
    "aug": "aug",
    "maj7": "maj7",
    "min7": "m7",
    "7": "7",
    "sus2": "sus2",
    "sus4": "sus4",
    "7sus4": "7sus4",
    "6": "6",
    "m6": "m6",
    "add9": "add9",
    "maj9": "maj9",
    "min9": "m9",
    "9": "9",
    "11": "11",
    "13": "13",
    "minmaj7": "mMaj7",
    "dim7": "dim7",
    "hdim7": "m7b5",
}
ADDITIONAL_QUALITY_INTERVALS = {
    "minmaj7": (0, 3, 7, 11),
    "dim7": (0, 3, 6, 9),
    "hdim7": (0, 3, 6, 10),
    "7sus4": (0, 5, 7, 10),
}


def _finite_float(value, name, minimum, maximum):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数字") from exc
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} 必须在 {minimum} 到 {maximum} 之间")
    return result


class ChordAnalyzer:
    """分层识别三和弦、挂留、七和弦、六和弦及转位低音。"""

    def __init__(
        self,
        time_window_ms=500,
        min_frequency=None,
        max_frequency=2093.0,
        min_chord_duration_ms=1100,
        confidence_threshold=0.15,
        expected_chord_count=0,
        experimental=False,
    ):
        if min_frequency is None:
            min_frequency = C1_HZ
        self.time_window_ms = _finite_float(time_window_ms, "time_window_ms", 100, 5000)
        self.min_frequency = _finite_float(
            min_frequency, "min_frequency", 20, TARGET_SR / 2
        )
        self.max_frequency = _finite_float(
            max_frequency, "max_frequency", 40, TARGET_SR / 2
        )
        if self.min_frequency >= self.max_frequency:
            raise ValueError("min_frequency 必须小于 max_frequency")
        self.min_chord_duration_ms = _finite_float(
            min_chord_duration_ms, "min_chord_duration_ms", 500, 10000
        )
        self.confidence_threshold = _finite_float(
            confidence_threshold, "confidence_threshold", 0, 1
        )
        try:
            count = int(expected_chord_count or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("expected_chord_count 必须是整数") from exc
        if count < 0 or count > 128:
            raise ValueError("expected_chord_count 必须在 0 到 128 之间")
        self.expected_chord_count = count
        self.experimental = bool(experimental)
        self._templates, self._labels = self._build_templates()

    @staticmethod
    def _build_templates():
        import numpy as np

        templates = []
        labels = []
        for root in range(12):
            for quality, (intervals, weights) in BASE_QUALITIES.items():
                template = np.zeros(12, dtype=np.float32)
                for interval, weight in zip(intervals, weights):
                    template[(root + interval) % 12] = weight
                templates.append(template)
                labels.append((root, quality))
        return np.asarray(templates), labels

    def analyze(self, input_path, cancel_event=None):
        import librosa
        import numpy as np

        def check_cancelled():
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("和弦分析任务已取消")

        check_cancelled()
        audio, sample_rate = librosa.load(input_path, sr=TARGET_SR, mono=True)
        check_cancelled()
        if audio.size == 0:
            raise ValueError("音频为空")

        (
            chroma,
            chroma_cqt,
            chroma_cens,
            rms,
            onset,
            times,
            low_cqt,
            low_frequencies,
        ) = self._extract_features(audio, sample_rate)
        check_cancelled()
        active = self._active_frames(rms)
        boundaries = self._choose_boundaries(chroma, onset, active)
        progression = self._classify_segments(
            chroma,
            chroma_cqt,
            chroma_cens,
            rms,
            times,
            boundaries,
            active,
            low_cqt,
            low_frequencies,
            cancel_event=cancel_event,
        )
        check_cancelled()
        progression = self._merge_adjacent(progression)
        key = self._estimate_key(chroma[:, active] if np.any(active) else chroma)
        stats = self._compute_stats(progression, audio, sample_rate)
        return {
            "analysis_mode": "harmony",
            "key": key,
            "chord_progression": progression,
            "stats": stats,
        }

    def _extract_features(self, audio, sample_rate):
        import librosa
        import numpy as np
        from scipy.ndimage import median_filter

        harmonic = librosa.effects.harmonic(audio, margin=3.0)
        n_octaves = min(
            6,
            max(1, int(math.ceil(math.log2(self.max_frequency / self.min_frequency)))),
        )
        n_bins = n_octaves * 36
        cqt = np.abs(
            librosa.cqt(
                harmonic,
                sr=sample_rate,
                hop_length=HOP_LENGTH,
                fmin=self.min_frequency,
                n_bins=n_bins,
                bins_per_octave=36,
            )
        )
        cqt_frequencies = librosa.cqt_frequencies(
            n_bins=n_bins, fmin=self.min_frequency, bins_per_octave=36
        )
        cqt[cqt_frequencies > self.max_frequency] = 0
        chroma_cqt = librosa.feature.chroma_cqt(
            C=cqt,
            sr=sample_rate,
            hop_length=HOP_LENGTH,
            fmin=self.min_frequency,
            n_chroma=12,
            bins_per_octave=36,
            norm=1,
        )
        chroma_cens = librosa.feature.chroma_cens(
            C=cqt,
            sr=sample_rate,
            hop_length=HOP_LENGTH,
            fmin=self.min_frequency,
            n_chroma=12,
            bins_per_octave=36,
            win_len_smooth=21,
        )
        chroma_cens /= np.maximum(chroma_cens.sum(axis=0, keepdims=True), 1e-8)
        chroma = 0.8 * chroma_cqt + 0.2 * chroma_cens

        frame_seconds = HOP_LENGTH / sample_rate
        smooth_frames = max(3, int(round(self.time_window_ms / 1000 / frame_seconds)))
        if smooth_frames % 2 == 0:
            smooth_frames += 1
        chroma = median_filter(chroma, size=(1, smooth_frames), mode="nearest")
        chroma /= np.maximum(chroma.sum(axis=0, keepdims=True), 1e-8)

        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=HOP_LENGTH)[0]
        onset = librosa.onset.onset_strength(
            y=audio, sr=sample_rate, hop_length=HOP_LENGTH, aggregate=np.median
        )
        frame_count = min(chroma.shape[1], rms.size, onset.size)
        chroma = chroma[:, :frame_count]
        chroma_cqt = chroma_cqt[:, :frame_count]
        chroma_cqt /= np.maximum(chroma_cqt.sum(axis=0, keepdims=True), 1e-8)
        chroma_cens = chroma_cens[:, :frame_count]
        rms = rms[:frame_count]
        onset = onset[:frame_count]
        times = librosa.frames_to_time(
            np.arange(frame_count), sr=sample_rate, hop_length=HOP_LENGTH
        )
        # 保留逐音高 CQT 以估计真正最低的稳定峰；不可只用折叠后的 chroma。
        low_limit = min(cqt.shape[0], 4 * 36)
        low_cqt = cqt[:low_limit, :frame_count]
        low_frequencies = cqt_frequencies[:low_limit]
        return (
            chroma,
            chroma_cqt,
            chroma_cens,
            rms,
            onset,
            times,
            low_cqt,
            low_frequencies,
        )

    @staticmethod
    def _active_frames(rms):
        import librosa
        import numpy as np

        if rms.size == 0 or float(np.max(rms)) <= 1e-10:
            return np.zeros(rms.size, dtype=bool)
        db = librosa.amplitude_to_db(rms, ref=np.max)
        return db >= -45.0

    def _boundary_strength(self, chroma, onset, active):
        import numpy as np
        from scipy.ndimage import gaussian_filter1d

        frame_seconds = HOP_LENGTH / TARGET_SR
        lag = max(1, int(round(0.16 / frame_seconds)))
        change = np.zeros(chroma.shape[1], dtype=np.float32)
        left = chroma[:, :-lag]
        right = chroma[:, lag:]
        denom = np.linalg.norm(left, axis=0) * np.linalg.norm(right, axis=0)
        change[lag:] = 1.0 - np.sum(left * right, axis=0) / np.maximum(denom, 1e-8)

        def robust_scale(values):
            values = np.maximum(values, 0)
            scale = np.percentile(values[active], 95) if np.any(active) else 0
            return np.clip(values / max(float(scale), 1e-8), 0, 1)

        score = 0.58 * robust_scale(change) + 0.42 * robust_scale(onset)
        score = gaussian_filter1d(score, sigma=1.5)
        score[~active] = 0
        return score

    def _choose_boundaries(self, chroma, onset, active):
        import numpy as np
        from scipy.signal import find_peaks

        frame_count = chroma.shape[1]
        if frame_count == 0 or not np.any(active):
            return [0, frame_count]
        active_indices = np.flatnonzero(active)
        start = int(active_indices[0])
        end = min(frame_count, int(active_indices[-1]) + 1)
        # 保留很短的开头静音在首段中，但彻底裁掉尾部静音，避免伪和弦。
        if start * HOP_LENGTH / TARGET_SR < 0.35:
            start = 0
        score = self._boundary_strength(chroma, onset, active)
        min_frames = max(
            1,
            int(math.ceil(self.min_chord_duration_ms / 1000 * TARGET_SR / HOP_LENGTH)),
        )

        if self.expected_chord_count:
            return self._fixed_count_boundaries(
                score, start, end, min_frames, self.expected_chord_count
            )

        interior = score[start + min_frames : end - min_frames]
        if interior.size == 0:
            return [start, end]
        threshold = max(
            float(np.percentile(interior, 68)),
            float(np.median(interior) + 0.45 * np.std(interior)),
        )
        peaks, _ = find_peaks(
            interior,
            height=threshold,
            prominence=max(0.035, 0.16 * np.max(interior)),
            distance=min_frames,
        )
        candidates = (peaks + start + min_frames).tolist()
        # 弱拨弦起音通常只有 onset，真正换和弦同时具备明显色度变化。
        selected = [p for p in candidates if score[p] >= threshold]
        return [start] + selected + [end]

    @staticmethod
    def _fixed_count_boundaries(score, start, end, min_frames, count):
        import numpy as np

        if count <= 1:
            return [start, end]
        available = end - start
        if available < count:
            raise ValueError("有效音频帧数不足以按预计数量分段")
        if available < count * min_frames:
            min_frames = max(1, available // count)

        # 动态规划：在最短段约束下，全局选择总边界强度最大的 N-1 个位置。
        # 只保留前一层分数，避免长音频产生 (N × 帧数) 的浮点矩阵。
        picks = count - 1
        previous = np.full(end + 1, -np.inf)
        previous[start] = 0.0
        parent = np.full((picks + 1, end + 1), -1, dtype=np.int32)
        for k in range(1, picks + 1):
            prefix_argmax = np.full(end + 1, -1, dtype=np.int32)
            best_position = -1
            best_score = -np.inf
            for pos in range(start, end + 1):
                if previous[pos] > best_score:
                    best_score = previous[pos]
                    best_position = pos
                prefix_argmax[pos] = best_position

            current = np.full(end + 1, -np.inf)
            earliest = start + k * min_frames
            latest = end - (count - k) * min_frames
            for pos in range(earliest, latest + 1):
                prev = int(prefix_argmax[pos - min_frames])
                if prev >= 0:
                    current[pos] = previous[prev] + float(score[pos])
                    parent[k, pos] = prev
            previous = current

        final_start = start + picks * min_frames
        final_end = end - min_frames
        if final_start > final_end or not np.any(
            np.isfinite(previous[final_start : final_end + 1])
        ):
            # 极短音频的安全回退仍按等距分段，不依赖具体时长或文件。
            return np.linspace(start, end, count + 1).round().astype(int).tolist()
        cursor = int(final_start + np.argmax(previous[final_start : final_end + 1]))
        result = [cursor]
        for k in range(picks, 1, -1):
            cursor = int(parent[k, cursor])
            result.append(cursor)
        return [start] + sorted(result) + [end]

    def _score_base_chords(self, vector):
        import numpy as np

        vector = np.maximum(np.asarray(vector, dtype=float), 0)
        vector /= max(float(vector.sum()), 1e-8)
        masks = self._templates > 0
        weighted = self._templates @ vector
        outside = np.asarray([vector[~mask].sum() for mask in masks])
        coverage = np.asarray([vector[mask].sum() for mask in masks])
        scores = weighted - 0.58 * outside + 0.12 * coverage
        for index, (_, quality) in enumerate(self._labels):
            if quality == "min":
                scores[index] += 0.02
        # 挂留必须同时有挂留音且大小三度都弱，避免把普通三和弦泛音判成 sus。
        for index, (root, quality) in enumerate(self._labels):
            if quality.startswith("sus"):
                suspended = 2 if quality == "sus2" else 5
                third_energy = max(vector[(root + 3) % 12], vector[(root + 4) % 12])
                if vector[(root + suspended) % 12] < 0.075:
                    scores[index] -= 0.20
                scores[index] -= 0.65 * third_energy
        order = np.argsort(scores)
        best = int(order[-1])
        return (
            self._labels[best],
            float(scores[best]),
            float(scores[best] - scores[order[-2]]),
        )

    def _upgrade_quality(self, vector, root, base_quality, base_score):
        """用扩展音残差、绝对证据和替代音 margin 分层升级。"""
        import numpy as np

        diagnostics = {}
        current_quality = base_quality
        current_score = base_score
        total_gain = 0.0

        # 多步升级可能 (如 maj -> 7 -> 9)，直至无匹配扩展或条件不符
        while True:
            candidates = []
            compatible = [
                (q, i) for q, (p, i) in EXTENSIONS.items() if p == current_quality
            ]
            if not compatible:
                break

            for quality, added_interval in compatible:
                if quality in {"11", "13"} and not self.experimental:
                    continue

                added = (root + added_interval) % 12
                evidence = float(vector[added])
                alternatives = [
                    float(vector[(root + i) % 12])
                    for q, i in compatible
                    if q != quality
                ]
                alternative = max(alternatives, default=0.0)
                alternative_margin = evidence - alternative

                # 寻找基础骨架音能量作为参考
                parent_info = BASE_QUALITIES.get(base_quality)
                if parent_info:
                    intervals = parent_info[0]
                    triad_values = [float(vector[(root + i) % 12]) for i in intervals]
                    third = triad_values[1]
                    fifth = triad_values[2]
                else:
                    third = fifth = max(float(np.max(vector)), 1e-8)

                relative_third = evidence / max(third, 1e-8)
                relative_fifth = evidence / max(fifth, 1e-8)

                if quality in {"6", "m6", "add9"}:
                    absolute_ok = evidence >= 0.070
                    balance_ok = relative_fifth >= 0.40
                    margin_ok = alternative_margin >= 0.008
                    gain = 1.25 * evidence - 0.05
                elif quality in {"11", "13"}:
                    absolute_ok = evidence >= 0.06
                    balance_ok = relative_third >= 0.30
                    margin_ok = alternative_margin >= 0.015
                    gain = 1.00 * evidence - 0.04
                else:  # 7, maj7, min7, 9
                    # 提高绝对证据阈值，防止三和弦泛音误升七和弦 (Item 4)
                    absolute_ok = evidence >= 0.048
                    balance_ok = relative_third >= 0.20 and current_score >= 0.15
                    # 7/maj7 对抗 color notes (6, add9) 应当有更高容忍度
                    margin_ok = alternative_margin >= -0.025
                    gain = 1.30 * evidence - 0.035

                accepted = absolute_ok and balance_ok and margin_ok and gain >= 0.001
                diagnostics[quality] = {
                    "absolute": evidence,
                    "relative_third": relative_third,
                    "relative_fifth": relative_fifth,
                    "alternative_margin": alternative_margin,
                    "gain": gain,
                    "accepted": accepted,
                }
                if accepted:
                    candidates.append((current_score + gain, quality, gain))

            if not candidates:
                break
            score, quality, gain = max(candidates)
            current_quality, current_score = quality, score
            total_gain += gain

        return current_quality, current_score, total_gain, diagnostics

    @staticmethod
    def _bass_pitch_class(low_cqt, low_frequencies, indices):
        import librosa
        import numpy as np
        from scipy.signal import find_peaks

        if not indices.size or low_cqt.size == 0:
            return None, 0.0
        spectrum = np.median(low_cqt[:, indices], axis=1)
        if not np.any(spectrum):
            return None, 0.0
        # C1 附近是 CQT 边缘；从约 B1 起找最低稳定局部峰。
        valid = np.flatnonzero(low_frequencies >= librosa.note_to_hz("B1"))
        peak_floor = 0.30 * float(np.max(spectrum[valid]))
        peaks, _ = find_peaks(spectrum, distance=2, height=peak_floor)
        peaks = [p for p in peaks if p in set(valid)]
        if not peaks:
            return None, 0.0
        lowest = min(peaks, key=lambda p: low_frequencies[p])
        midi = int(round(float(librosa.hz_to_midi(low_frequencies[lowest]))))
        strength = float(spectrum[lowest] / max(float(np.max(spectrum[valid])), 1e-8))
        return midi % 12, strength

    @staticmethod
    def _bass_chroma(low_cqt, low_frequencies, indices):
        import librosa
        import numpy as np

        if not indices.size or low_cqt.size == 0:
            return np.zeros(12, dtype=float)
        spectrum = np.median(low_cqt[:, indices], axis=1)
        result = np.zeros(12, dtype=float)
        midi = np.rint(librosa.hz_to_midi(low_frequencies)).astype(int)
        for pitch_class in range(12):
            result[pitch_class] = float(spectrum[midi % 12 == pitch_class].sum())
        result /= max(float(result.sum()), 1e-8)
        return result

    def _classify_segments(
        self,
        chroma,
        chroma_cqt,
        chroma_cens,
        rms,
        times,
        boundaries,
        active,
        low_cqt,
        low_frequencies,
        cancel_event=None,
    ):
        import numpy as np

        result = []
        frame_seconds = HOP_LENGTH / TARGET_SR
        previous_root = None
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("和弦分析任务已取消")
            indices = np.arange(left, right)
            indices = indices[active[left:right]]
            evidence = {}
            if indices.size == 0:
                label, confidence = "N/A", 0.0
            else:
                trim = min(int(round(0.12 / frame_seconds)), indices.size // 5)
                core = indices[trim : indices.size - trim] if trim else indices
                weights = rms[core] / max(float(np.max(rms[core])), 1e-8)
                weighted_frames = 0.5 + 0.5 * weights
                vector = np.median(chroma[:, core] * weighted_frames, axis=1)
                vector /= max(float(vector.sum()), 1e-8)
                cqt_vector = np.median(chroma_cqt[:, core] * weighted_frames, axis=1)
                cqt_vector /= max(float(cqt_vector.sum()), 1e-8)
                cens_vector = np.median(chroma_cens[:, core] * weighted_frames, axis=1)
                cens_vector /= max(float(cens_vector.sum()), 1e-8)
                bass_vector = self._bass_chroma(low_cqt, low_frequencies, core)
                (root, quality), base_score, base_margin = self._score_base_chords(
                    vector
                )
                original_root, original_quality = root, quality

                # 相同音集歧义：优先可靠低音；其次使用前后功能连续性。
                bass, bass_strength = self._bass_pitch_class(
                    low_cqt, low_frequencies, core
                )
                # 低音上的大三和弦若覆盖充分，优先根位而不是误作其六度小和弦转位。
                if (
                    quality == "min"
                    and bass == (root + 3) % 12
                    and bass_strength >= 0.8
                    and vector[bass] > 1.5 * vector[root]
                ):
                    major_tones = [bass, (bass + 4) % 12, (bass + 7) % 12]
                    if float(vector[major_tones].sum()) >= 0.65:
                        root, quality = bass, "maj"
                # C-D-G-B 一类结构先由上方 G-B-D 骨架定根，再保留 C pedal bass。
                if (
                    quality == "sus2"
                    and bass == root
                    and vector[(root + 11) % 12] >= 0.10
                    and vector[(root + 2) % 12] >= 0.10
                ):
                    root, quality = (root + 7) % 12, "maj"
                if quality == "sus4":
                    alternate_root = (root + 5) % 12
                    if bass == alternate_root or previous_root == alternate_root:
                        root, quality = alternate_root, "sus2"
                quality, final_score, extension_gain, upgrade_diagnostics = (
                    self._upgrade_quality(vector, root, quality, base_score)
                )
                if quality == "6" and (
                    (root, "maj") != (original_root, original_quality)
                    or bass != root
                    or bass_strength < 0.30
                ):
                    quality, final_score, extension_gain = "maj", base_score, 0.0
                    upgrade_diagnostics["6"]["accepted"] = False
                    upgrade_diagnostics["6"]["rejected_by_bass"] = True
                # 递归寻找所有音程，支持多级扩展 (如 9 和弦包含 7 音)
                temp_q, intervals_list = quality, []
                while temp_q in EXTENSIONS:
                    p_q, interval = EXTENSIONS[temp_q]
                    intervals_list.append(interval)
                    temp_q = p_q
                intervals = (
                    BASE_QUALITIES[temp_q][0] if temp_q in BASE_QUALITIES else (0, 4, 7)
                ) + tuple(intervals_list)

                chord_tones = {(root + interval) % 12 for interval in intervals}
                label = NOTE_NAMES[root] + QUALITY_SUFFIX.get(quality, "")
                slash_supported = bass in chord_tones or (
                    bass_strength >= 0.30 and vector[bass] >= 0.10
                )
                if (
                    bass is not None
                    and bass != root
                    and slash_supported
                    and bass_strength >= 0.18
                ):
                    label += "/" + NOTE_NAMES[bass]
                coverage = float(vector[list(chord_tones)].sum())
                confidence = float(
                    np.clip(
                        0.62 * coverage
                        + 0.22 * (1 - np.exp(-5 * max(base_margin, 0)))
                        + 0.10 * min(1.0, bass_strength)
                        + 0.06 * min(1.0, extension_gain * 12),
                        0,
                        1,
                    )
                )
                if confidence < self.confidence_threshold:
                    label = "N/A"
                evidence = {
                    "chroma": {
                        NOTE_NAMES[i]: round(float(vector[i]), 4) for i in range(12)
                    },
                    "cqt_chroma": {
                        NOTE_NAMES[i]: round(float(cqt_vector[i]), 4) for i in range(12)
                    },
                    "cens_chroma": {
                        NOTE_NAMES[i]: round(float(cens_vector[i]), 4)
                        for i in range(12)
                    },
                    "bass_chroma": {
                        NOTE_NAMES[i]: round(float(bass_vector[i]), 4)
                        for i in range(12)
                    },
                    "bass": NOTE_NAMES[bass] if bass is not None else None,
                    "bass_strength": round(bass_strength, 4),
                    "base_score": round(base_score, 4),
                    "base_margin": round(base_margin, 4),
                    "final_score": round(final_score, 4),
                    "extension_gain": round(extension_gain, 4),
                    "upgrade_candidates": {
                        name: {
                            key: (
                                round(value, 4) if isinstance(value, float) else value
                            )
                            for key, value in values.items()
                        }
                        for name, values in upgrade_diagnostics.items()
                    },
                }
                previous_root = root
            start_time = (
                float(times[left]) if left < len(times) else left * frame_seconds
            )
            end_time = min(
                right * frame_seconds,
                float(times[-1] + frame_seconds)
                if len(times)
                else right * frame_seconds,
            )
            result.append(
                {
                    "chord": label,
                    "start_time": round(start_time, 2),
                    "duration_sec": round(max(0, end_time - start_time), 2),
                    "confidence": round(confidence, 3),
                    "evidence": evidence,
                }
            )
        return result

    @staticmethod
    def _merge_adjacent(progression):
        merged = []
        for segment in progression:
            if merged and merged[-1]["chord"] == segment["chord"]:
                end = segment["start_time"] + segment["duration_sec"]
                merged[-1]["duration_sec"] = round(end - merged[-1]["start_time"], 2)
                merged[-1]["confidence"] = round(
                    max(merged[-1]["confidence"], segment["confidence"]), 3
                )
            else:
                merged.append(dict(segment))
        return merged

    @staticmethod
    def _estimate_key(chroma):
        import numpy as np

        major = np.asarray(
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        )
        minor = np.asarray(
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        )
        average = np.mean(chroma, axis=1) if chroma.size else np.zeros(12)
        if float(average.sum()) <= 0:
            return {"key": "Unknown", "scale": "Unknown", "confidence": 0.0}
        candidates = []
        for root in range(12):
            candidates.append(
                (np.corrcoef(average, np.roll(major, root))[0, 1], root, "Major")
            )
            candidates.append(
                (np.corrcoef(average, np.roll(minor, root))[0, 1], root, "Minor")
            )
        score, root, scale = max(candidates, key=lambda item: item[0])
        return {
            "key": NOTE_NAMES[root],
            "scale": scale,
            "confidence": round(float(score), 3),
        }

    @staticmethod
    def _compute_stats(progression, audio, sample_rate):
        counts = {}
        for segment in progression:
            counts[segment["chord"]] = counts.get(segment["chord"], 0) + 1
        common = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        average = (
            sum(item["confidence"] for item in progression) / len(progression)
            if progression
            else 0.0
        )
        return {
            "duration_sec": round(len(audio) / sample_rate, 2),
            "total_segments": len(progression),
            "unique_chords": len(counts),
            "most_common_chords": [
                {"chord": chord, "count": count} for chord, count in common[:5]
            ],
            "avg_confidence": round(average, 3),
        }


def _mix_chord_parts(label):
    if label in {"N", "N/A"}:
        return None

    # 分离 Slash Chord 信息
    main_chord = label
    slash_note = None
    if "/" in label:
        parts = label.split("/")
        main_chord = parts[0]
        slash_name = parts[1]
        if slash_name in NOTE_NAMES:
            slash_note = NOTE_NAMES.index(slash_name)

    # 提取根音 (处理 # 变音符号)
    if len(main_chord) > 1 and main_chord[1] == "#":
        root_name = main_chord[:2]
        suffix = main_chord[2:]
    else:
        root_name = main_chord[:1]
        suffix = main_chord[1:]

    # 建立后缀到质量键的反向映射 (Item 7)
    quality = {v: k for k, v in QUALITY_SUFFIX.items()}.get(suffix)

    if quality is None or root_name not in NOTE_NAMES:
        return None
    return NOTE_NAMES.index(root_name), quality, slash_note


def _bass_pitch_class_for_label(label):
    """Return explicit inversion bass, or the root for a root-position chord."""
    parsed = _mix_chord_parts(label)
    if parsed is not None:
        root, _quality, slash_note = parsed
        return root if slash_note is None else slash_note

    bass_note = re.fullmatch(r"([A-G](?:#)?)(-?\d+)", str(label or ""))
    if bass_note:
        return NOTE_NAMES.index(bass_note.group(1))
    return None


def _export_mix_csv(result, path):
    content = io.StringIO(newline="")
    writer = csv.writer(content)
    writer.writerow(["start", "end", "chord", "confidence"])
    for item in result.get("chord_events") or result["chord_progression"]:
        writer.writerow(
            [
                f"{item['start_time']:.3f}",
                f"{item['start_time'] + item['duration_sec']:.3f}",
                item["chord"],
                f"{item['confidence']:.3f}",
            ]
        )
    _atomic_write_text(path, content.getvalue())


def _atomic_write_text(path, content):
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _get_chord_intervals(quality):
    """支持扩展和弦的完整音程解析 (Item 7)。"""
    if quality in ADDITIONAL_QUALITY_INTERVALS:
        return ADDITIONAL_QUALITY_INTERVALS[quality]
    if quality in BASE_QUALITIES:
        return BASE_QUALITIES[quality][0]
    temp_q, intervals_list = quality, []
    while temp_q in EXTENSIONS:
        p_q, interval = EXTENSIONS[temp_q]
        intervals_list.append(interval)
        temp_q = p_q
    if temp_q in BASE_QUALITIES:
        return BASE_QUALITIES[temp_q][0] + tuple(intervals_list)
    # 再次检查 mix_analysis 中的 triad 常量作为兜底

    return MIX_QUALITIES.get(temp_q, (0, 4, 7))


def _midi_notes_for_label(label):
    parsed = _mix_chord_parts(label)
    if parsed is not None:
        root, quality, slash_note = parsed
        notes = [48 + root + interval for interval in _get_chord_intervals(quality)]
        if slash_note is not None:
            bass_note = 36 + slash_note
            notes.append(bass_note if bass_note not in notes else 24 + slash_note)
        return notes

    bass_note = re.fullmatch(r"([A-G](?:#)?)(-?\d+)", str(label or ""))
    if not bass_note:
        return []
    pitch_class = NOTE_NAMES.index(bass_note.group(1))
    midi_note = (int(bass_note.group(2)) + 1) * 12 + pitch_class
    return [midi_note] if 0 <= midi_note <= 127 else []


def export_chord_progression_midi(progression, path, bpm=120.0):
    """Export edited chord or bass-note segments as an atomic MIDI file."""
    bpm = float(bpm)
    if not math.isfinite(bpm) or not 1 <= bpm <= 999:
        raise ValueError("MIDI BPM 必须在 1 到 999 之间")
    ticks_per_beat = 480
    ticks_per_second = ticks_per_beat * bpm / 60.0
    events = []
    for item in progression:
        start_time = float(item["start_time"])
        duration = float(item["duration_sec"])
        if (
            not math.isfinite(start_time)
            or not math.isfinite(duration)
            or start_time < 0
            or duration <= 0
        ):
            raise ValueError("和弦时间线包含无效时间")
        label = str(item.get("chord") or "N").strip() or "N"
        notes = _midi_notes_for_label(label)
        if not notes:
            if label in {"N", "N/A"}:
                continue
            raise ValueError(f"无法导出不支持的和弦标签：{label}")
        start = int(round(start_time * ticks_per_second))
        end = int(round((start_time + duration) * ticks_per_second))
        for note in notes:
            events.append((start, 1, max(0, min(127, int(note)))))
            events.append((max(start + 1, end), 0, max(0, min(127, int(note)))))
    # 同 tick 先 note_off，避免 DAW 中相邻同音重叠。
    events.sort(key=lambda event: (event[0], event[1]))
    midi = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(MetaMessage("track_name", name="Mix chord approximation", time=0))
    track.append(MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    cursor = 0
    for tick, is_on, note in events:
        track.append(
            Message(
                "note_on" if is_on else "note_off",
                note=note,
                velocity=80 if is_on else 0,
                time=tick - cursor,
            )
        )
        cursor = tick
    track.append(MetaMessage("end_of_track", time=0))
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        midi.save(temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _export_mix_midi(result, path):
    export_chord_progression_midi(
        result.get("chord_events") or result["chord_progression"],
        path,
        bpm=float(result.get("tempo_bpm") or 120.0),
    )


def _mix_harmonic_stems(
    tracks,
    output_path,
    *,
    cancel_event=None,
):
    import numpy as np
    import soundfile as sf

    selected = {
        item["stem_id"]: Path(item["path"])
        for item in tracks
        if item.get("stem_id") in {"guitar", "piano", "other"}
    }
    if set(selected) != {"guitar", "piano", "other"}:
        raise RuntimeError("六轨结果缺少吉他、钢琴或其他轨")

    inputs = []
    target = Path(output_path)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.wav")
    try:
        for stem in ("guitar", "piano", "other"):
            inputs.append(sf.SoundFile(os.fspath(selected[stem])))
        signature = (
            inputs[0].samplerate,
            inputs[0].channels,
            len(inputs[0]),
        )
        if any(
            (stream.samplerate, stream.channels, len(stream)) != signature
            for stream in inputs[1:]
        ):
            raise RuntimeError("和声增强轨的采样率、声道或帧数不一致")
        target.parent.mkdir(parents=True, exist_ok=True)
        with sf.SoundFile(
            os.fspath(temporary),
            mode="w",
            samplerate=signature[0],
            channels=signature[1],
            format="WAV",
            subtype="FLOAT",
        ) as output:
            remaining = signature[2]
            while remaining:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("和弦分析任务已取消")
                count = min(65536, remaining)
                chunks = [
                    stream.read(count, dtype="float32", always_2d=True)
                    for stream in inputs
                ]
                if any(len(chunk) != count for chunk in chunks):
                    raise RuntimeError("和声增强轨读取不完整")
                mixed = np.sum(
                    np.stack(chunks, axis=0),
                    axis=0,
                    dtype=np.float32,
                )
                if not np.all(np.isfinite(mixed)):
                    raise RuntimeError("和声增强轨含无效采样值")
                output.write(mixed)
                remaining -= count
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
        for stream in inputs:
            stream.close()
    return target


def _has_recognized_chords(result):
    return any(
        str(item.get("chord") or "").strip() not in {"", "N", "N/A"}
        for item in result.get("chord_progression", [])
    )


class ChordAnalysisTool(BaseTool):
    name = "chord-analysis"
    display_name = "和弦分析"
    description = "MPS/CPU 神经和弦识别、DSP 回退与 pYIN 转位低音分析"
    short_desc = "神经和弦 · 精细边界 · 转位低音"
    accepted_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"]

    def process(self, input_path, output_dir, **kwargs):
        cancel_event = kwargs.pop("_cancel_event", None)
        context = kwargs.pop("_task_context", None)
        progress = kwargs.pop("_progress_callback", None) or (
            lambda _value, _message: None
        )
        auxiliary_input = kwargs.pop("_auxiliary_input", None)
        auxiliary_stem_tracks = kwargs.pop("_auxiliary_stem_tracks", None)
        kwargs.pop("_stem_backend", None)
        options = validate_chord_analysis_options(kwargs)
        analysis_mode = options.get("analysis_mode", "harmony")
        quality_mode = options.get("quality_mode", "balanced")
        experimental = bool(options.get("experimental", False))
        use_runpod = kwargs.get("use_runpod", False)
        if not isinstance(use_runpod, bool):
            raise ValueError("use_runpod 必须是布尔值")
        if use_runpod and quality_mode != "high":
            raise ValueError("使用 RunPod 神经和弦分析必须选择质量优先模式")
        if use_runpod and analysis_mode == "bass":
            raise ValueError("贝斯模式不支持 use_runpod=True（RunPod 神经和弦分析）")

        def harmony_analyzer():
            harmony_duration = max(
                500,
                float(options.get("min_chord_duration_ms", 1100)),
            )
            return ChordAnalyzer(
                time_window_ms=options.get("time_window_ms", 500),
                min_frequency=options.get("min_frequency", C1_HZ),
                max_frequency=options.get("max_frequency", 2093.0),
                min_chord_duration_ms=harmony_duration,
                confidence_threshold=options.get("confidence_threshold", 0.15),
                expected_chord_count=options.get("expected_chord_count", 0),
                experimental=experimental,
            )

        if analysis_mode == "bass":
            analyzer = BassAnalyzer(
                confidence_threshold=options.get("confidence_threshold", 0.3),
                min_note_duration_ms=options.get("min_chord_duration_ms", 80),
            )
        elif analysis_mode == "mix":
            analyzer = MixChordAnalyzer(
                confidence_threshold=options.get("confidence_threshold", 0.18),
                min_chord_duration_ms=options.get(
                    "min_chord_duration_ms", 500
                ),
            )
        else:
            analyzer = harmony_analyzer()

        if quality_mode == "high" and analysis_mode in {"harmony", "mix"}:
            secondary = (
                MixChordAnalyzer(
                    confidence_threshold=options.get(
                        "confidence_threshold", 0.18
                    ),
                    min_chord_duration_ms=options.get(
                        "min_chord_duration_ms", 500
                    ),
                )
                if analysis_mode == "harmony"
                else harmony_analyzer()
            )
            bass_analyzer = BassAnalyzer(
                confidence_threshold=0.3,
                min_note_duration_ms=min(
                    5000,
                    max(
                        80,
                        float(options.get("min_chord_duration_ms", 1100)),
                    ),
                ),
            )
            dsp_fallback = HybridChordAnalyzer(
                analyzer,
                secondary,
                bass=bass_analyzer,
                debug=options.get("debug", False),
            )
            analyzer = QualityFirstChordAnalyzer(
                NeuralChordAnalyzer(
                    analysis_mode=analysis_mode,
                    confidence_threshold=options.get(
                        "confidence_threshold", 0.18
                    ),
                    min_chord_duration_ms=options.get(
                        "min_chord_duration_ms", 500
                    ),
                    bass_analyzer=bass_analyzer,
                    progress=progress,
                ),
                dsp_fallback,
            )
        segmentation_kwargs = {"cancel_event": cancel_event}
        if context is not None:
            segmentation_kwargs["context"] = context
        started_at = time.perf_counter()
        progress(10, "正在分析原混音和弦")
        if use_runpod:
            from tools.cloud_chord_client import CloudChordClient

            result = CloudChordClient().analyze(
                input_path,
                {**options, "analysis_mode": analysis_mode},
                cancel_event=cancel_event,
                progress=progress,
            )
            neural = dict(result.get("neural") or {})
            ensemble = dict(result.get("ensemble") or {})
            ensemble.setdefault(
                "views",
                ["original_mix", "structured_root_quality_bass"],
            )
            ensemble.setdefault(
                "experts",
                [neural.get("model", "chord_structured_ensemble_v1")],
            )
            ensemble.setdefault("chosen", "neural_primary")
            ensemble.setdefault("fallback", False)
            ensemble.setdefault("reason", "runpod_neural_primary")
            ensemble.setdefault("trace", [])
            ensemble.update(
                {
                    "quality_mode": "high",
                    "execution": "runpod",
                    "device": neural.get("device", "cuda"),
                    "runpod_requested": True,
                }
            )
            result["ensemble"] = ensemble
        else:
            result = analyze_with_segmentation(
                input_path,
                analyzer.analyze,
                **segmentation_kwargs,
            )
        baseline_elapsed = time.perf_counter() - started_at
        if (
            quality_mode == "high"
            and analysis_mode in {"harmony", "mix"}
            and not use_runpod
            and _has_recognized_chords(result)
        ):
            multiview_started = time.perf_counter()
            preparation_failure = None
            auxiliary_source = None
            try:
                if auxiliary_input is not None:
                    harmonic_path = Path(auxiliary_input).expanduser().resolve()
                    if not harmonic_path.is_file():
                        raise RuntimeError("指定的和声增强音频不存在")
                    auxiliary_source = "explicit_harmonic_audio"
                    result = ConservativeMultiViewChordAnalyzer.fuse(
                        result,
                        analyze_with_segmentation(
                            harmonic_path,
                            analyzer.analyze,
                            **segmentation_kwargs,
                        ),
                    )
                elif auxiliary_stem_tracks:
                    runtime = runtime_directory()
                    with tempfile.TemporaryDirectory(
                        prefix="chord-multiview-",
                        dir=os.fspath(runtime),
                    ) as temp_dir:
                        temp_root = Path(temp_dir)
                        progress(46, "正在复用已完成的六轨结果")
                        auxiliary_source = "reused_project_stems"
                        harmonic_path = _mix_harmonic_stems(
                            auxiliary_stem_tracks,
                            temp_root / "harmonic-view.wav",
                            cancel_event=cancel_event,
                        )
                        progress(76, "正在分析和声增强视图")
                        auxiliary_result = analyze_with_segmentation(
                            harmonic_path,
                            analyzer.analyze,
                            **segmentation_kwargs,
                        )
                        result = ConservativeMultiViewChordAnalyzer.fuse(
                            result,
                            auxiliary_result,
                        )
                else:
                    result = ConservativeMultiViewChordAnalyzer.record_skipped(
                        result,
                        "auxiliary_stems_unavailable",
                    )
            except Exception as exc:
                if cancel_event is not None and cancel_event.is_set():
                    raise
                preparation_failure = str(exc) or type(exc).__name__
                result = ConservativeMultiViewChordAnalyzer.record_failure(
                    result,
                    preparation_failure,
                )
            multiview_elapsed = time.perf_counter() - multiview_started
            multiview = dict(result.get("multiview") or {})
            multiview.update(
                {
                    "generated_auxiliary": False,
                    "auxiliary_source": auxiliary_source,
                    "baseline_runtime_sec": round(baseline_elapsed, 3),
                    "auxiliary_pipeline_runtime_sec": round(
                        multiview_elapsed,
                        3,
                    ),
                    "preparation_failure": preparation_failure,
                }
            )
            result["multiview"] = multiview
        elif (
            quality_mode == "high"
            and analysis_mode in {"harmony", "mix"}
            and not use_runpod
        ):
            result = ConservativeMultiViewChordAnalyzer.record_skipped(
                result,
                "baseline_has_no_recognized_chords",
            )
            result["multiview"]["baseline_runtime_sec"] = round(
                baseline_elapsed,
                3,
            )
        progress(94, "正在整理和弦结果")
        if experimental and analysis_mode in {"mix", "harmony"}:
            notation_source = (
                result.get("harmonic_regions")
                or result.get("chord_progression")
                or []
            )
            try:
                notation_result = analyze_notation_timeline(
                    input_path,
                    notation_source,
                    cancel_event=cancel_event,
                )
                result.update(notation_result)
                notation = notation_result["notation"]
                if notation.get("status") == "recognized":
                    result["raw_tempo_bpm"] = result.get("tempo_bpm")
                    result["tempo_bpm"] = notation["tempo_bpm"]
                    result["beat_count"] = notation["beat_count"]
                stats = dict(result.get("stats") or {})
                stats.update(
                    {
                        "notation_event_count": notation["event_count"],
                        "measure_count": notation["measure_count"],
                    }
                )
                result["stats"] = stats
            except Exception as exc:
                if cancel_event is not None and cancel_event.is_set():
                    raise
                result["notation"] = {
                    "status": "failed",
                    "failure": str(exc) or type(exc).__name__,
                }
        if isinstance(result.get("multiview"), dict):
            result["multiview"]["total_runtime_sec"] = round(
                time.perf_counter() - started_at,
                3,
            )
        progression = (
            result.get("chord_events")
            or result.get("chord_progression")
            or []
        )
        recognized_event_count = sum(
            item.get("chord") not in {None, "", "N", "N/A"}
            for item in progression
        )
        stats = dict(result.get("stats") or {})
        stats["event_count"] = len(progression)
        stats["recognized_event_count"] = recognized_event_count
        stats["unrecognized_event_count"] = (
            len(progression) - recognized_event_count
        )
        result["stats"] = stats
        has_chords = any(
            str(item.get("chord") or "").strip() not in {"", "N", "N/A"}
            for item in progression
        )
        if not has_chords and result.get("recognition_status") not in {
            "insufficient_harmony", "no_confident_chords",
        }:
            raise ValueError(
                "未识别到有效和弦或低音，请检查音频内容或调整参数"
            )
        if "ensemble" not in result:
            result["ensemble"] = {
                "quality_mode": quality_mode,
                "views": [analysis_mode],
                "experts": [type(analyzer).__name__],
                "chosen": "baseline",
                "fallback": False,
                "reason": "balanced_mode"
                if quality_mode == "balanced"
                else "no_chord_rerank_for_bass",
                "execution": "local",
                "runpod_requested": use_runpod,
                "trace": [],
            }
        else:
            result["ensemble"].setdefault("runpod_requested", use_runpod)
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("和弦分析任务已取消")
        Path(output_dir).expanduser().mkdir(parents=True, exist_ok=True)
        base_name = safe_output_stem(
            kwargs.get("original_filename") or input_path
        )
        suffix = {"bass": "bass", "mix": "mix_chords"}.get(analysis_mode, "chords")
        output_path = os.path.join(output_dir, f"{base_name}_{suffix}.json")
        summary_path = os.path.join(output_dir, f"{base_name}_{suffix}.txt")
        output_files = [output_path, summary_path]
        _atomic_write_text(
            output_path,
            json.dumps(result, ensure_ascii=False, indent=2),
        )
        summary = io.StringIO()
        if not has_chords:
            summary.write("未检测到足够可靠的和弦；请参考原音频，勿将低音候选当作完整和弦。\n\n")
        if analysis_mode == "bass":
            summary.write("贝斯低音分析报告 (Bass Note Analysis Report)\n")
            summary.write("=" * 60 + "\n\n")
            summary.write("低音/根音候选（不推断大调或小调性质）:\n")
        elif analysis_mode == "mix":
            summary.write("成品/混音和弦分析报告 (Mix Chord Analysis Report)\n")
            summary.write("=" * 60 + "\n\n")
            summary.write(f"稳定 BPM: {result['tempo_bpm']:.2f}\n")
            summary.write(
                f"调性: {result['key']['key']} {result['key']['scale']}\n\n"
            )
            notation = result.get("notation") or {}
            if notation.get("status") == "recognized":
                summary.write(
                    f"谱面小节: {notation['measure_count']}，"
                    f"谱面和弦符号: {notation['event_count']}\n"
                )
                if notation.get("tempo_octave_corrected"):
                    summary.write(
                        f"拍速倍频校正: {notation['raw_tempo_bpm']:.2f} → "
                        f"{notation['tempo_bpm']:.2f} BPM\n"
                    )
                summary.write("\n谱面级和弦进行:\n")
                for item in result.get("notation_events", []):
                    summary.write(
                        f"  M{int(item['measure_number']) + 1:<4d} "
                        f"beat {float(item['beat_offset']) + 1:>4.1f}  "
                        f"{item['chord']:<8s} "
                        f"(支持: {float(item['measure_support']):.2f})\n"
                    )
                summary.write("\n详细演奏事件:\n")
            else:
                summary.write("根音优先的节拍同步和弦进行:\n")
        else:
            summary.write("和弦分析报告 (Chord Analysis Report)\n")
            summary.write("=" * 60 + "\n\n")
            summary.write(
                f"调性 (Key): {result['key']['key']} {result['key']['scale']} "
                f"(置信度: {result['key']['confidence']})\n\n"
            )
            summary.write("和弦进行 (Chord Progression):\n")
        for item in progression:
            end = item["start_time"] + item["duration_sec"]
            summary.write(
                f"  {item['start_time']:>7.2f}s - {end:>7.2f}s  "
                f"{item['chord']:<8s} (置信度: {item['confidence']:.2f})\n"
            )
        _atomic_write_text(summary_path, summary.getvalue())
        if analysis_mode in {"mix", "harmony"}:
            csv_path = os.path.join(output_dir, f"{base_name}_{suffix}.csv")
            midi_path = os.path.join(output_dir, f"{base_name}_{suffix}.mid")
            _export_mix_csv(result, csv_path)
            _export_mix_midi(result, midi_path)
            output_files.extend([csv_path, midi_path])
        label = "低音段" if analysis_mode == "bass" else "和弦段"
        return {
            "success": True,
            "output_files": output_files,
            "message": (
                f"分析完成！共识别 {recognized_event_count} 个有效{label}"
                f"（全部事件 {len(progression)} 个）"
                if has_chords else "分析完成，未检测到足够可靠的和弦"
            ),
            "metadata": result,
        }


class HybridChordAnalyzer:
    """High-quality adapter that fuses existing analyzers without hard deps."""

    def __init__(
        self,
        baseline,
        secondary,
        bass=None,
        cloud_view_provider=None,
        cloud_analyzer=None,
        runpod_requested=False,
        debug=False,
    ):
        self.baseline = baseline
        self.secondary = secondary
        self.bass = bass
        self.cloud_view_provider = cloud_view_provider
        self.cloud_analyzer = cloud_analyzer
        self.runpod_requested = runpod_requested is True
        self.debug = bool(debug)

    @staticmethod
    def _label_at(progression, timestamp):
        for item in progression or []:
            start = float(item.get("start_time", 0))
            end = start + float(item.get("duration_sec", 0))
            if start <= timestamp < end:
                return item
        return None

    @staticmethod
    def _root(label):
        label = str(label or "")
        if not label or label in {"N", "N/A"}:
            return None
        return label[:2] if len(label) > 1 and label[1] == "#" else label[:1]

    @classmethod
    def _segment_similarity(cls, left, right):
        if left.get("chord") == right.get("chord"):
            return 1.0
        if cls._root(left.get("chord")) == cls._root(right.get("chord")):
            return 0.65
        return 0.0

    def analyze(self, input_path, cancel_event=None):
        baseline = self.baseline.analyze(input_path, cancel_event=cancel_event)
        failures = {}
        experts = ["baseline"]
        baseline_mode = baseline.get("analysis_mode", "harmony")
        views = [baseline_mode]
        secondary_mode = "secondary"
        try:
            secondary = self.secondary.analyze(input_path, cancel_event=cancel_event)
            experts.append(type(self.secondary).__name__)
            secondary_mode = secondary.get("analysis_mode", "secondary")
            views.append(secondary_mode)
        except Exception as exc:
            secondary = None
            failures["secondary"] = str(exc) or type(exc).__name__
        try:
            bass = (
                self.bass.analyze(input_path, cancel_event=cancel_event)
                if self.bass is not None
                else None
            )
            if bass is not None:
                experts.append(type(self.bass).__name__)
                views.append("bass_root")
        except Exception as exc:
            bass = None
            failures["bass_root"] = str(exc) or type(exc).__name__

        cloud_stem = None
        if self.runpod_requested:
            if self.cloud_view_provider is None or self.cloud_analyzer is None:
                raise RuntimeError("RunPod GPU Stem 提供器不可用")
            cloud_view = self.cloud_view_provider(input_path, purpose="chord")
            cloud_stem = getattr(cloud_view, "stem", None)
            cloud_path = os.fspath(getattr(cloud_view, "path", cloud_view))
            cloud = self.cloud_analyzer.analyze(
                cloud_path,
                cancel_event=cancel_event,
            )
            experts.append("cloud_adapter")
            views.append("cloud_stem")
        else:
            cloud = None

        traces = []
        progression = []
        for index, segment in enumerate(baseline.get("chord_progression", [])):
            candidates = [
                EnsembleCandidate(
                    "baseline",
                    segment,
                    float(segment.get("confidence", 0)),
                    "baseline",
                    "harmony",
                )
            ]
            midpoint = (
                float(segment.get("start_time", 0))
                + float(segment.get("duration_sec", 0)) / 2.0
            )
            for name, result, view in (
                (secondary_mode, secondary, secondary_mode),
                ("cloud", cloud, "cloud_stem"),
            ):
                match = self._label_at(
                    (result or {}).get("chord_progression"), midpoint
                )
                if match is None:
                    continue
                quality = float(match.get("confidence", 0))
                bass_match = self._label_at(
                    (bass or {}).get("chord_progression"), midpoint
                )
                if bass_match and self._root(match.get("chord")) == self._root(
                    bass_match.get("chord")
                ):
                    quality = min(1.0, quality + 0.04)
                candidates.append(EnsembleCandidate(name, match, quality, name, view))
            decision = select_candidate(
                candidates,
                "baseline",
                self._segment_similarity,
                minimum_gain=0.03,
                failures=failures,
                fallback_on_any_failure=True,
            )
            chosen = dict(decision.value)
            # Preserve the baseline time grid so expert timelines cannot create gaps.
            chosen["start_time"] = segment["start_time"]
            chosen["duration_sec"] = segment["duration_sec"]
            progression.append(chosen)
            traces.append({"segment": index, **decision.trace})

        baseline["chord_progression"] = ChordAnalyzer._merge_adjacent(progression)
        stats = dict(baseline.get("stats") or {})
        stats["total_segments"] = len(baseline["chord_progression"])
        counts = {}
        for item in baseline["chord_progression"]:
            label = item.get("chord", "N")
            counts[label] = counts.get(label, 0) + 1
        common = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        stats["unique_chords"] = len(counts)
        stats["most_common_chords"] = [
            {"chord": chord, "count": count} for chord, count in common[:5]
        ]
        confidences = [
            float(item.get("confidence", 0)) for item in baseline["chord_progression"]
        ]
        stats["avg_confidence"] = (
            round(sum(confidences) / len(confidences), 3) if confidences else 0.0
        )
        baseline["stats"] = stats
        any_fallback = bool(failures) or any(item["fallback"] for item in traces)
        chosen_counts = {}
        for item in traces:
            chosen = item.get("chosen", "baseline")
            chosen_counts[chosen] = chosen_counts.get(chosen, 0) + 1
        fallback_count = sum(1 for item in traces if item.get("fallback"))
        ensemble = {
            "quality_mode": "high",
            "views": views,
            "experts": experts,
            "chosen": "segment_rerank",
            "fallback": any_fallback,
            "reason": "expert_failure"
            if failures
            else ("segment_fallback" if any_fallback else "consensus_rerank"),
            "execution": "local",
            "runpod_requested": self.runpod_requested,
            "cloud_view_attempted": self.runpod_requested,
            "cloud_view_used": cloud is not None,
            "cloud_stem": cloud_stem,
            "summary": {
                "segment_count": len(traces),
                "chosen_counts": chosen_counts,
                "fallback_count": fallback_count,
                "failure_count": len(failures),
            },
            "failures": failures,
        }
        if self.debug:
            ensemble["trace"] = traces
        baseline["ensemble"] = ensemble
        return baseline


class QualityFirstChordAnalyzer:
    """Prefer the bundled neural model and retain the DSP ensemble as fallback."""

    def __init__(self, neural, dsp_fallback):
        self.neural = neural
        self.dsp_fallback = dsp_fallback
        self.cloud_view_provider = None
        self.runpod_requested = False

    def _fallback(self, input_path, cancel_event, reason, detail):
        result = self.dsp_fallback.analyze(
            input_path,
            cancel_event=cancel_event,
        )
        ensemble = dict(result.get("ensemble") or {})
        failures = dict(ensemble.get("failures") or {})
        failures["neural"] = detail
        ensemble.update(
            {
                "quality_mode": "high",
                "chosen": "dsp_fallback",
                "fallback": True,
                "reason": reason,
                "execution": "local",
                "failures": failures,
            }
        )
        result["ensemble"] = ensemble
        return result

    def analyze(self, input_path, cancel_event=None):
        try:
            result = self.neural.analyze(input_path, cancel_event=cancel_event)
        except Exception as exc:
            if cancel_event is not None and cancel_event.is_set():
                raise
            return self._fallback(
                input_path,
                cancel_event,
                "neural_failure",
                str(exc) or type(exc).__name__,
            )

        neural = dict(result.get("neural") or {})
        if not neural.get("quality_gate_passed", True) and not neural.get("abstained"):
            return self._fallback(
                input_path,
                cancel_event,
                "neural_quality_gate",
                str(
                    neural.get(
                        "quality_gate_reason",
                        "neural quality gate rejected the result",
                    )
                ),
            )

        bass_failed = bool(neural.get("bass_failure"))
        result["ensemble"] = {
            "quality_mode": "high",
            "views": ["original_mix", "structured_root_quality_bass"],
            "experts": [
                neural.get("model", "chord_structured_ensemble_v1"),
            ],
            "chosen": "neural_primary",
            "fallback": False,
            "reason": (
                "neural_abstention" if neural.get("abstained")
                else "neural_primary_bass_unavailable"
                if bass_failed
                else "neural_primary"
            ),
            "execution": "local",
            "device": neural.get("device", "cpu"),
            "mps_fallback": bool(neural.get("mps_fallback")),
            "failures": (
                {"bass_root": neural["bass_failure"]} if bass_failed else {}
            ),
            "trace": [],
        }
        return result


class ConservativeMultiViewChordAnalyzer:
    """Upgrade chord quality without changing the original-mix time grid."""

    MIN_AUXILIARY_COVERAGE = 0.7
    MIN_AUXILIARY_CONFIDENCE = 0.58
    MAX_CONFIDENCE_DEFICIT = 0.12
    QUALITY_UPGRADES = {
        "": {"7", "maj7"},
        "m": {"m7"},
        "sus4": {"7sus4"},
        "dim": {"m7b5", "dim7"},
    }

    @staticmethod
    def _base_and_bass(label):
        value = str(label or "").strip()
        if value in {"", "N", "N/A"}:
            return None, None, None
        match = re.fullmatch(
            r"([A-G](?:#|b)?)(.*?)(?:/([A-G](?:#|b)?))?",
            value,
        )
        if match is None:
            return None, None, None
        return match.group(1), match.group(2), match.group(3)

    @classmethod
    def _supported_upgrade(cls, baseline_label, auxiliary_label):
        baseline_root, baseline_quality, baseline_bass = cls._base_and_bass(
            baseline_label
        )
        auxiliary_root, auxiliary_quality, _ = cls._base_and_bass(
            auxiliary_label
        )
        if (
            baseline_root is None
            or auxiliary_root != baseline_root
            or auxiliary_quality
            not in cls.QUALITY_UPGRADES.get(baseline_quality, set())
        ):
            return None
        upgraded = f"{baseline_root}{auxiliary_quality}"
        if baseline_bass is not None and baseline_bass != baseline_root:
            upgraded = f"{upgraded}/{baseline_bass}"
        return upgraded

    @classmethod
    def _candidate(cls, baseline, auxiliary):
        start = float(baseline.get("start_time", 0))
        duration = float(baseline.get("duration_sec", 0))
        end = start + duration
        if duration <= 0:
            return None
        candidates = {}
        for item in auxiliary or []:
            item_start = float(item.get("start_time", 0))
            item_end = item_start + float(item.get("duration_sec", 0))
            overlap = max(0.0, min(end, item_end) - max(start, item_start))
            upgraded = cls._supported_upgrade(
                baseline.get("chord"),
                item.get("chord"),
            )
            if overlap <= 0 or upgraded is None:
                continue
            entry = candidates.setdefault(
                upgraded,
                {"duration": 0.0, "weighted_confidence": 0.0},
            )
            entry["duration"] += overlap
            entry["weighted_confidence"] += overlap * float(
                item.get("confidence", 0)
            )
        if not candidates:
            return None
        label, evidence = max(
            candidates.items(),
            key=lambda item: (
                item[1]["duration"],
                item[1]["weighted_confidence"],
            ),
        )
        coverage = evidence["duration"] / duration
        confidence = evidence["weighted_confidence"] / max(
            evidence["duration"],
            1e-9,
        )
        baseline_confidence = float(baseline.get("confidence", 0))
        if (
            coverage < cls.MIN_AUXILIARY_COVERAGE
            or confidence < cls.MIN_AUXILIARY_CONFIDENCE
            or confidence + cls.MAX_CONFIDENCE_DEFICIT < baseline_confidence
        ):
            return None
        return label, coverage, confidence

    @classmethod
    def _upgrade_timeline(cls, baseline, auxiliary):
        upgraded = []
        count = 0
        for segment in baseline or []:
            item = dict(segment)
            candidate = cls._candidate(item, auxiliary)
            if candidate is not None:
                label, coverage, confidence = candidate
                evidence = dict(item.get("evidence") or {})
                evidence["multiview_upgrade"] = {
                    "baseline_chord": item.get("chord"),
                    "auxiliary_chord": label,
                    "coverage": round(coverage, 4),
                    "auxiliary_confidence": round(confidence, 4),
                    "policy": "same_root_one_tone_quality_upgrade",
                }
                item["chord"] = label
                item["confidence"] = round(confidence, 3)
                item["evidence"] = evidence
                count += 1
            upgraded.append(item)
        return upgraded, count

    @staticmethod
    def _refresh_stats(result):
        progression = result.get("chord_progression") or []
        counts = {}
        confidences = []
        for item in progression:
            label = item.get("chord", "N")
            counts[label] = counts.get(label, 0) + 1
            if isinstance(item.get("confidence"), (int, float)):
                confidences.append(float(item["confidence"]))
        stats = dict(result.get("stats") or {})
        stats["total_segments"] = len(progression)
        stats["unique_chords"] = len(counts)
        stats["most_common_chords"] = [
            {"chord": chord, "count": count}
            for chord, count in sorted(
                counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
        ]
        stats["avg_confidence"] = (
            round(sum(confidences) / len(confidences), 3)
            if confidences
            else 0.0
        )
        active = [
            item
            for item in progression
            if item.get("chord") not in {None, "", "N", "N/A"}
        ]
        active_duration = sum(
            float(item.get("duration_sec", 0))
            for item in active
        )
        stats["duration_weighted_chord_confidence"] = round(
            sum(
                float(item.get("confidence", 0))
                * float(item.get("duration_sec", 0))
                for item in active
            )
            / max(active_duration, 1e-9),
            5,
        )
        result["stats"] = stats

    @staticmethod
    def _update_ensemble(result, *, include_harmonic=True, **values):
        ensemble = dict(result.get("ensemble") or {})
        views = list(ensemble.get("views") or ["original_mix"])
        if include_harmonic and "harmonic_enhanced" not in views:
            views.append("harmonic_enhanced")
        ensemble.update(values)
        ensemble["views"] = views
        result["ensemble"] = ensemble
        return result

    @classmethod
    def record_failure(cls, result, detail):
        result = dict(result)
        ensemble = dict(result.get("ensemble") or {})
        failures = dict(ensemble.get("failures") or {})
        failures["harmonic_view"] = detail
        cls._update_ensemble(
            result,
            chosen="original_mix_baseline",
            fallback=True,
            reason="harmonic_view_failure",
            failures=failures,
        )
        result["multiview"] = {
            "policy": "same_root_one_tone_quality_upgrade",
            "status": "baseline_fallback",
            "upgraded_segments": 0,
            "upgraded_events": 0,
        }
        return result

    @classmethod
    def record_skipped(cls, result, reason, *, auxiliary_available=False):
        result = dict(result)
        cls._update_ensemble(
            result,
            include_harmonic=auxiliary_available,
            chosen="original_mix_baseline",
            reason=reason,
        )
        result["multiview"] = {
            "policy": "same_root_one_tone_quality_upgrade",
            "status": "skipped",
            "reason": reason,
            "upgraded_segments": 0,
            "upgraded_events": 0,
        }
        return result

    @classmethod
    def fuse(cls, baseline, auxiliary):
        result = dict(baseline)
        auxiliary_neural = dict(auxiliary.get("neural") or {})
        auxiliary_ensemble = dict(auxiliary.get("ensemble") or {})
        accepted_auxiliary = bool(
            auxiliary_neural.get("quality_gate_passed")
            and not auxiliary_neural.get("abstained")
            and auxiliary_ensemble.get("chosen")
            not in {"dsp_fallback", "original_mix_baseline"}
        )
        if not accepted_auxiliary:
            return cls.record_skipped(
                result,
                "auxiliary_quality_gate",
                auxiliary_available=True,
            )

        progression, progression_count = cls._upgrade_timeline(
            result.get("chord_progression"),
            auxiliary.get("chord_progression"),
        )
        result["chord_progression"] = progression
        region_count = 0
        if isinstance(result.get("harmonic_regions"), list):
            regions, region_count = cls._upgrade_timeline(
                result["harmonic_regions"],
                auxiliary.get("harmonic_regions")
                or auxiliary.get("chord_progression"),
            )
            result["harmonic_regions"] = regions
        event_count = 0
        if isinstance(result.get("chord_events"), list):
            events, event_count = cls._upgrade_timeline(
                result["chord_events"],
                auxiliary.get("chord_events")
                or auxiliary.get("chord_progression"),
            )
            result["chord_events"] = events

        cls._refresh_stats(result)
        cls._update_ensemble(
            result,
            chosen="original_timeline_conservative_quality_fusion",
            fallback=bool((result.get("ensemble") or {}).get("fallback")),
            reason=(
                "conservative_quality_upgrade"
                if progression_count or event_count
                else "no_supported_quality_upgrade"
            ),
        )
        result["multiview"] = {
            "policy": "same_root_one_tone_quality_upgrade",
            "status": "fused",
            "minimum_auxiliary_coverage": cls.MIN_AUXILIARY_COVERAGE,
            "minimum_auxiliary_confidence": cls.MIN_AUXILIARY_CONFIDENCE,
            "maximum_confidence_deficit": cls.MAX_CONFIDENCE_DEFICIT,
            "upgraded_segments": progression_count,
            "upgraded_harmonic_regions": region_count,
            "upgraded_events": event_count,
            "timeline_source": "original_mix",
            "auxiliary_source": "guitar+piano+other",
        }
        return result


registry.register(ChordAnalysisTool())
