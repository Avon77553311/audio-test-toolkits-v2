"""单贝斯低音识别：低阈值 pYIN 候选、谐波投票和重复拨奏分段。"""

import math
import numpy as np

TARGET_SR = 22050
HOP_LENGTH = 256


def _get_bass_fmin():
    import librosa

    return librosa.note_to_hz("E1")


BASS_FMIN = 41.20344461410875  # Pre-calculated to avoid librosa import at startup
BASS_FMAX = 330.0
BASS_FRAME_LENGTH = 4096


class BassAnalyzer:
    """跟踪单贝斯基频，并以明显起音拆分同音重复拨奏。"""

    def __init__(self, confidence_threshold=0.3, min_note_duration_ms=80):
        self.confidence_threshold = self._finite_float(
            confidence_threshold, "confidence_threshold", 0, 1
        )
        self.min_note_duration_ms = self._finite_float(
            min_note_duration_ms, "min_note_duration_ms", 50, 5000
        )

    @staticmethod
    def _finite_float(value, name, minimum, maximum):
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 必须是数字") from exc
        if not math.isfinite(result) or not minimum <= result <= maximum:
            raise ValueError(f"{name} 必须在 {minimum} 到 {maximum} 之间")
        return result

    def analyze(self, input_path, cancel_event=None):
        import librosa

        def check_cancelled():
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("和弦分析任务已取消")

        check_cancelled()
        audio, sample_rate = librosa.load(input_path, sr=TARGET_SR, mono=True)
        check_cancelled()
        if audio.size == 0:
            raise ValueError("音频为空")

        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            sr=sample_rate,
            fmin=BASS_FMIN,
            fmax=BASS_FMAX,
            frame_length=BASS_FRAME_LENGTH,
            hop_length=HOP_LENGTH,
            center=True,
            fill_na=np.nan,
        )
        check_cancelled()
        frame_count = len(f0)
        if frame_count == 0:
            return self._build_result([], audio, sample_rate)

        rms = librosa.feature.rms(
            y=audio, frame_length=BASS_FRAME_LENGTH, hop_length=HOP_LENGTH, center=True
        )[0][:frame_count]
        rms_db = (
            librosa.amplitude_to_db(rms, ref=np.max)
            if np.any(rms)
            else np.full(frame_count, -80.0)
        )
        probability = np.nan_to_num(voiced_prob[:frame_count], nan=0.0)
        harmonic = self._harmonic_consistency(audio, sample_rate, f0)
        check_cancelled()

        # 低阈值只生成候选；最终需通过局部票数、谐波或强 pYIN 证据。
        candidate_floor = min(0.12, max(0.04, self.confidence_threshold * 0.35))
        candidate = (
            np.isfinite(f0)
            & (rms_db >= -48.0)
            & ((probability >= candidate_floor) | (harmonic >= 0.46))
        )
        midi = np.full(frame_count, np.nan)
        midi[candidate] = np.rint(librosa.hz_to_midi(f0[candidate]))
        midi, support = self._local_vote(
            midi, probability, harmonic, rms_db, self.confidence_threshold
        )

        frame_seconds = HOP_LENGTH / TARGET_SR
        min_frames = max(
            5, int(math.ceil(self.min_note_duration_ms / 1000 / frame_seconds))
        )
        midi = self._retain_stable_pitch_classes(
            midi, probability, harmonic, min_frames
        )
        midi = self._bridge_short_dropouts(midi, max_gap=min(10, min_frames))
        midi = self._remove_short_pitch_islands(midi, support, min_frames)

        onset = librosa.onset.onset_strength(
            y=audio, sr=sample_rate, hop_length=HOP_LENGTH, aggregate=np.median
        )[:frame_count]
        attack_rms = librosa.feature.rms(
            y=audio, frame_length=1024, hop_length=HOP_LENGTH, center=True
        )[0][:frame_count]
        attack_rms_db = (
            librosa.amplitude_to_db(attack_rms, ref=np.max)
            if np.any(attack_rms)
            else np.full(frame_count, -80.0)
        )
        progression = self._segment_notes(
            midi,
            probability,
            harmonic,
            attack_rms_db,
            onset,
            len(audio) / sample_rate,
            min_frames,
        )
        return self._build_result(progression, audio, sample_rate)

    @staticmethod
    def _harmonic_consistency(audio, sample_rate, f0):
        import librosa

        spectrum = np.abs(
            librosa.stft(
                audio, n_fft=BASS_FRAME_LENGTH, hop_length=HOP_LENGTH, center=True
            )
        )
        frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=BASS_FRAME_LENGTH)
        result = np.zeros(len(f0), dtype=np.float32)
        for frame, fundamental in enumerate(f0):
            if not np.isfinite(fundamental) or frame >= spectrum.shape[1]:
                continue
            values = []
            for multiple, weight in ((1, 0.50), (2, 0.30), (3, 0.20)):
                index = int(np.argmin(np.abs(frequencies - fundamental * multiple)))
                left, right = max(0, index - 4), min(len(frequencies), index + 5)
                floor = float(np.median(spectrum[left:right, frame])) + 1e-9
                ratio = float(spectrum[index, frame]) / floor
                values.append(weight * min(1.0, ratio / 6.0))
            result[frame] = sum(values)
        return result

    @staticmethod
    def _local_vote(midi, probability, harmonic, rms_db, strong_threshold):
        voted = np.full_like(midi, np.nan)
        support = np.zeros(len(midi), dtype=np.float32)
        radius = 21
        for index in range(len(midi)):
            left, right = max(0, index - radius), min(len(midi), index + radius + 1)
            values = midi[left:right]
            valid = np.isfinite(values)
            if np.count_nonzero(valid) < 3:
                continue
            weights = (
                0.45 * probability[left:right]
                + 0.35 * harmonic[left:right]
                + 0.20 * np.clip((rms_db[left:right] + 48.0) / 36.0, 0, 1)
            )
            choices = np.unique(values[valid]).astype(int)
            totals = np.asarray(
                [weights[valid][values[valid] == note].sum() for note in choices]
            )
            best_at = int(np.argmax(totals))
            best_note = choices[best_at]
            local_support = float(totals[best_at] / max(weights[valid].sum(), 1e-8))
            center_strong = np.isfinite(midi[index]) and (
                probability[index] >= strong_threshold or harmonic[index] >= 0.58
            )
            # 三帧一致仍可能是瞬态误音；低置信候选需占邻域多数。
            if local_support >= 0.52 or (center_strong and local_support >= 0.38):
                voted[index] = best_note
                support[index] = local_support
        return voted, support

    @staticmethod
    def _retain_stable_pitch_classes(midi, probability, harmonic, min_frames):
        """低阈值候选须由同音的强证据核心支撑，过滤滑音和衰减尾音。"""
        result = midi.copy()
        notes = np.unique(result[np.isfinite(result)]).astype(int)
        for note in notes:
            frames = result == note
            strong = frames & ((probability >= 0.18) | (harmonic >= 0.72))
            # 稳定帧数阈值：至少达到最短音符时长的一半，且不少于 5 帧，防止抖动。
            threshold = max(5, min_frames // 2)
            if np.count_nonzero(strong) < threshold:
                result[frames] = np.nan
        return result

    @staticmethod
    def _bridge_short_dropouts(midi, max_gap):
        result = midi.copy()
        index = 0
        while index < len(result):
            if np.isfinite(result[index]):
                index += 1
                continue
            right = index
            while right < len(result) and not np.isfinite(result[right]):
                right += 1
            if (
                index > 0
                and right < len(result)
                and right - index <= max_gap
                and result[index - 1] == result[right]
            ):
                result[index:right] = result[index - 1]
            index = right
        return result

    @staticmethod
    def _remove_short_pitch_islands(midi, support, min_frames):
        result = midi.copy()
        index = 0
        while index < len(result):
            if not np.isfinite(result[index]):
                index += 1
                continue
            note = result[index]
            right = index + 1
            while right < len(result) and result[right] == note:
                right += 1
            # 孤立音块阈值：采用自适应 min_frames
            if (
                right - index < min_frames
                and float(np.median(support[index:right])) < 0.70
            ):
                result[index:right] = np.nan
            index = right
        return result

    @staticmethod
    def _suppress_short_excursions(midi):
        """把夹在同音之间、短于约 350 ms 的衰减期误跳吸回稳定音。"""
        result = midi.copy()
        runs = []
        index = 0
        while index < len(result):
            if not np.isfinite(result[index]):
                index += 1
                continue
            note = result[index]
            right = index + 1
            while right < len(result) and result[right] == note:
                right += 1
            runs.append((index, right, note))
            index = right
        max_excursion = int(round(0.35 * TARGET_SR / HOP_LENGTH))
        for previous, middle, following in zip(runs, runs[1:], runs[2:]):
            if (
                previous[2] == following[2]
                and middle[2] != previous[2]
                and middle[1] - middle[0] <= max_excursion
                and middle[0] - previous[1] <= 10
                and following[0] - middle[1] <= 10
            ):
                result[middle[0] : middle[1]] = previous[2]
        return result

    def _segment_notes(
        self, midi, probability, harmonic, rms_db, onset, audio_duration, min_frames
    ):
        import librosa
        from scipy.signal import find_peaks

        frame_seconds = HOP_LENGTH / TARGET_SR
        pitch_runs = []
        index = 0
        while index < len(midi):
            if not np.isfinite(midi[index]):
                index += 1
                continue
            note = int(midi[index])
            right = index + 1
            while right < len(midi) and midi[right] == note:
                right += 1
            if right - index >= min_frames:
                pitch_runs.append((index, right, note))
            index = right

        segments = []
        onset_scale = float(np.percentile(onset, 95)) if onset.size else 0.0
        for left, right, note in pitch_runs:
            local = onset[left:right]
            cuts = [left]
            if local.size and onset_scale > 1e-8:
                # 适配 attack_frames
                attack_frames = max(4, min_frames)
                peaks, _ = find_peaks(
                    local,
                    prominence=max(0.08 * onset_scale, 1e-8),
                    height=max(float(np.percentile(onset, 65)), 0.05 * onset_scale),
                    distance=attack_frames,
                )
                for peak in peaks:
                    cut = left + int(peak)
                    # RMS 回升检查：适配最短音符时长的一半
                    lookback = max(1, min_frames // 2)
                    before = rms_db[max(left, cut - lookback) : cut]
                    rise = rms_db[cut] - (
                        float(np.median(before)) if before.size else rms_db[cut]
                    )
                    if (
                        cut - cuts[-1] >= attack_frames
                        and right - cut >= attack_frames
                        and rise >= 2.2
                    ):
                        cuts.append(cut)
            cuts.append(right)
            for start, end in zip(cuts[:-1], cuts[1:]):
                if end - start < min_frames:
                    continue
                confidence = np.clip(
                    0.55 * float(np.median(probability[start:end]))
                    + 0.35 * float(np.median(harmonic[start:end]))
                    + 0.10,
                    0,
                    1,
                )
                start_time = max(0.0, start * frame_seconds)
                end_time = min(audio_duration, end * frame_seconds)
                if end_time <= start_time:
                    continue
                segments.append(
                    {
                        "chord": str(
                            librosa.midi_to_note(
                                note, octave=True, cents=False, unicode=False
                            )
                        ),
                        "start_time": round(start_time, 2),
                        "duration_sec": round(end_time - start_time, 2),
                        "confidence": round(float(confidence), 3),
                        "evidence": {
                            "median_voiced_probability": round(
                                float(np.median(probability[start:end])), 3
                            ),
                            "median_harmonic_consistency": round(
                                float(np.median(harmonic[start:end])), 3
                            ),
                            "onset_strength": round(
                                float(onset[start]) / max(onset_scale, 1e-8), 3
                            ),
                        },
                    }
                )
        return segments

    @staticmethod
    def _build_result(progression, audio, sample_rate):
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
            "analysis_mode": "bass",
            "key": {"key": "Unknown", "scale": "Bass notes", "confidence": 0.0},
            "chord_progression": progression,
            "stats": {
                "duration_sec": round(len(audio) / sample_rate, 2),
                "total_segments": len(progression),
                "unique_chords": len(counts),
                "most_common_chords": [
                    {"chord": chord, "count": count} for chord, count in common[:5]
                ],
                "avg_confidence": round(average, 3),
            },
        }
