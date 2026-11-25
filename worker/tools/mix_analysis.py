"""纯 CPU 成品混音和弦估计：节拍同步色度、低频根音与 Viterbi 平滑。"""

import math

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
HOP_LENGTH = 512
def _get_c1():
    import librosa
    return librosa.note_to_hz('C1')

C1_HZ = 32.70319566257483  # Pre-calculated
TARGET_SR = 22050
MIX_QUALITIES = {
    'maj': (0, 4, 7), 'min': (0, 3, 7), '7': (0, 4, 7, 10),
    'maj7': (0, 4, 7, 11), 'min7': (0, 3, 7, 10),
    'sus2': (0, 2, 7), 'sus4': (0, 5, 7),
    '6': (0, 4, 7, 9), 'dim': (0, 3, 6), 'aug': (0, 4, 8),
}
SUFFIX = {'maj': '', 'min': 'm', '7': '7', 'maj7': 'maj7',
          'min7': 'm7', 'sus2': 'sus2', 'sus4': 'sus4', '6': '6',
          'dim': 'dim', 'aug': 'aug'}


class MixChordAnalyzer:
    """面向节拍清晰成品歌曲的保守、根音优先和弦分析器。"""

    def __init__(self, confidence_threshold=0.18, min_chord_duration_ms=500):
        self.confidence_threshold = float(confidence_threshold)
        self.min_chord_duration_ms = float(min_chord_duration_ms)
        if (
            not math.isfinite(self.min_chord_duration_ms)
            or not 50 <= self.min_chord_duration_ms <= 10000
        ):
            raise ValueError("min_chord_duration_ms 必须在 50 到 10000 之间")
        self.states = [(root, quality) for root in range(12)
                       for quality in MIX_QUALITIES] + [(None, 'N')]
        self.templates = self._build_templates()

    def _build_templates(self):
        import numpy as np
        rows = []
        for root, quality in self.states[:-1]:
            row = np.zeros(12, dtype=np.float32)
            tones = MIX_QUALITIES[quality]
            for interval in tones:
                row[(root + interval) % 12] = 1.0
            row[root] = 1.35
            row /= np.linalg.norm(row)
            rows.append(row)
        return np.asarray(rows)

    def analyze(self, input_path, cancel_event=None):
        import librosa
        import numpy as np

        def check_cancelled():
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError('和弦分析任务已取消')

        check_cancelled()
        audio, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)
        check_cancelled()
        if audio.size == 0:
            raise ValueError('音频为空')
        harmonic, _ = librosa.effects.hpss(audio, margin=(1.5, 3.0))
        cqt = np.abs(librosa.cqt(harmonic, sr=sr, hop_length=HOP_LENGTH,
                                 fmin=C1_HZ, n_bins=216,
                                 bins_per_octave=36))
        chord_freqs = librosa.cqt_frequencies(
            216, fmin=C1_HZ, bins_per_octave=36
        )
        # Keep the sub-bass in its dedicated view so a pedal note cannot
        # overwrite the harmonic root used for chord-template matching.
        chord_weight = np.clip((chord_freqs - 70.0) / 100.0, 0.0, 1.0)
        chord_cqt = cqt * chord_weight[:, None]
        chroma_cqt = librosa.feature.chroma_cqt(
            C=chord_cqt, sr=sr, hop_length=HOP_LENGTH, fmin=C1_HZ,
            bins_per_octave=36, norm=1)
        chroma_cens = librosa.feature.chroma_cens(
            C=chord_cqt, sr=sr, hop_length=HOP_LENGTH, fmin=C1_HZ,
            bins_per_octave=36, win_len_smooth=21)
        frames = min(chroma_cqt.shape[1], chroma_cens.shape[1])
        chroma = 0.72 * chroma_cqt[:, :frames] + 0.28 * chroma_cens[:, :frames]
        chroma /= np.maximum(chroma.sum(axis=0, keepdims=True), 1e-8)
        check_cancelled()

        # 原音低频副本只辅助根音，不参与 quality 判断。
        low_cqt = np.abs(librosa.cqt(
            audio, sr=sr, hop_length=HOP_LENGTH, fmin=C1_HZ,
            n_bins=108, bins_per_octave=36))[:, :frames]
        low_freqs = librosa.cqt_frequencies(
            108, fmin=C1_HZ, bins_per_octave=36)
        low_weight = np.clip(1.0 - (low_freqs - 32.7) / 230.0, 0.15, 1.0)
        low_cqt *= low_weight[:, None]
        bass_chroma = np.zeros((12, frames), dtype=np.float32)
        midi = np.rint(librosa.hz_to_midi(low_freqs)).astype(int)
        for pc in range(12):
            bass_chroma[pc] = low_cqt[midi % 12 == pc].sum(axis=0)
        bass_chroma /= np.maximum(bass_chroma.sum(axis=0, keepdims=True), 1e-8)
        check_cancelled()

        onset = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=HOP_LENGTH)
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset, sr=sr, hop_length=HOP_LENGTH, trim=False)
        tempo = float(np.asarray(tempo).reshape(-1)[0]) if np.size(tempo) else 0.0
        beat_frames = np.asarray(beat_frames, dtype=int)
        beat_frames = beat_frames[(beat_frames > 0) & (beat_frames < frames)]
        if beat_frames.size < 2:
            fallback = max(1, int(round((60.0 / 120.0) * sr / HOP_LENGTH)))
            beat_frames = np.arange(fallback, frames, fallback)
            tempo = 120.0
        boundaries = np.unique(np.r_[0, beat_frames, frames]).astype(int)
        boundaries = boundaries[np.r_[True, np.diff(boundaries) > 0]]

        beat_chroma, beat_bass, activity = [], [], []
        rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0][:frames]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            beat_chroma.append(np.median(chroma[:, left:right], axis=1))
            # 中位数抑制经过低音；持续低频才会影响根音。
            beat_bass.append(np.median(bass_chroma[:, left:right], axis=1))
            activity.append(float(np.median(rms[left:right])))
        beat_chroma = self._normalise(np.asarray(beat_chroma))
        beat_bass = self._normalise(np.asarray(beat_bass))
        activity = np.asarray(activity)
        beat_duration_ms = float(
            np.median(np.diff(boundaries)) * HOP_LENGTH * 1000.0 / sr
        )
        minimum_run_beats = max(
            1,
            int(math.ceil(self.min_chord_duration_ms / max(beat_duration_ms, 1.0))),
        )
        labels, confidences = self._decode(
            beat_chroma,
            beat_bass,
            activity,
            cancel_event=cancel_event,
            minimum_run_beats=minimum_run_beats,
        )

        duration = len(audio) / sr
        segments = []
        for i, (label, confidence) in enumerate(zip(labels, confidences)):
            start = boundaries[i] * HOP_LENGTH / sr
            end = min(duration, boundaries[i + 1] * HOP_LENGTH / sr)
            item = {'chord': label, 'start_time': round(start, 3),
                    'duration_sec': round(max(0.0, end - start), 3),
                    'confidence': round(float(confidence), 3)}
            if segments and segments[-1]['chord'] == label:
                old = segments[-1]
                old['duration_sec'] = round(end - old['start_time'], 3)
                old['confidence'] = round(max(old['confidence'], item['confidence']), 3)
            else:
                segments.append(item)
        key = self._estimate_key(beat_chroma)
        counts = {}
        for segment in segments:
            counts[segment['chord']] = counts.get(segment['chord'], 0) + 1
        return {
            'analysis_mode': 'mix', 'key': key,
            'tempo_bpm': round(tempo if math.isfinite(tempo) and tempo > 0 else 120.0, 2),
            'beat_count': len(labels), 'chord_progression': segments,
            'stats': {
                'duration_sec': round(duration, 2), 'total_segments': len(segments),
                'unique_chords': len(counts),
                'most_common_chords': [{'chord': k, 'count': v} for k, v in
                                       sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]],
                'avg_confidence': round(float(np.mean(confidences)) if confidences else 0.0, 3),
            },
        }

    @staticmethod
    def _normalise(matrix):
        import numpy as np
        matrix = np.maximum(matrix, 0)
        return matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1e-8)

    @staticmethod
    def _smooth_short_runs(path, emissions, minimum_run_beats):
        import numpy as np

        path = np.asarray(path, dtype=int).copy()
        minimum_run_beats = max(1, int(minimum_run_beats))
        if minimum_run_beats <= 1 or len(path) < 2:
            return path
        for _ in range(len(path)):
            starts = np.r_[0, np.flatnonzero(np.diff(path)) + 1]
            ends = np.r_[starts[1:], len(path)]
            changed = False
            for start, end in zip(starts, ends):
                if path[start] == emissions.shape[1] - 1:
                    continue
                if end - start >= minimum_run_beats:
                    continue
                candidates = []
                if start > 0:
                    candidates.append(int(path[start - 1]))
                if end < len(path):
                    candidates.append(int(path[end]))
                if not candidates:
                    continue
                candidates = [state for state in dict.fromkeys(candidates)
                              if state != emissions.shape[1] - 1]
                if not candidates:
                    continue
                replacement = max(
                    candidates,
                    key=lambda state: float(np.sum(emissions[start:end, state])),
                )
                path[start:end] = replacement
                changed = True
            if not changed:
                break
        return path

    def _labels_from_path(self, path, chroma, bass):
        import numpy as np

        labels = ["N"] * len(path)
        starts = np.r_[0, np.flatnonzero(np.diff(path)) + 1]
        ends = np.r_[starts[1:], len(path)]
        for start, end in zip(starts, ends):
            root, quality = self.states[int(path[start])]
            if quality == "N":
                continue
            bass_vector = np.mean(bass[start:end], axis=0)
            bass_note = int(np.argmax(bass_vector))
            bass_strength = float(bass_vector[bass_note])
            chroma_support = float(np.mean(chroma[start:end, bass_note]))
            if (
                quality == "sus4"
                and bass_note == (root + 5) % 12
                and bass_strength >= 0.25
                and chroma_support >= 0.08
            ):
                root, quality = bass_note, "sus2"
            elif (
                quality == "min7"
                and bass_note == (root + 3) % 12
                and bass_strength >= 0.25
                and chroma_support >= 0.08
            ):
                root, quality = bass_note, "6"

            label = NOTE_NAMES[root] + SUFFIX[quality]
            root_strength = float(bass_vector[root])
            chord_tones = {
                (root + interval) % 12 for interval in MIX_QUALITIES[quality]
            }
            if (
                bass_note != root
                and bass_strength >= 0.30
                and bass_strength >= root_strength + 0.08
                and (
                    chroma_support >= (
                        0.05 if bass_note in chord_tones else 0.08
                    )
                    or bass_strength >= 0.65
                )
            ):
                label += "/" + NOTE_NAMES[bass_note]
            labels[start:end] = [label] * (end - start)
        return labels

    def _decode(
        self,
        chroma,
        bass,
        activity,
        cancel_event=None,
        minimum_run_beats=2,
    ):
        import numpy as np
        count = len(chroma)
        state_count = len(self.states)
        emissions = np.full((count, state_count), -2.0, dtype=float)
        active_floor = max(float(np.max(activity)) * 0.035, 1e-8)
        for t in range(count):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError('和弦分析任务已取消')
            vec = chroma[t]
            cosine = self.templates @ (vec / max(np.linalg.norm(vec), 1e-8))
            for s, (root, quality) in enumerate(self.states[:-1]):
                tones = [(root + x) % 12 for x in MIX_QUALITIES[quality]]
                coverage = float(vec[tones].sum())
                outside = 1.0 - coverage
                root_support = 0.88 * float(vec[root]) + 0.12 * float(bass[t, root])
                score = 1.15 * cosine[s] + 0.90 * coverage - 0.50 * outside + 0.72 * root_support
                if quality in {'7', 'maj7', 'min7'}:
                    seventh = MIX_QUALITIES[quality][-1]
                    evidence = float(vec[(root + seventh) % 12])
                    if evidence < 0.075 or evidence < 0.38 * max(float(vec[(root + 3) % 12]), float(vec[(root + 4) % 12])):
                        score -= 0.34  # 证据不足时回退基础大/小
                if quality == '6':
                    evidence = float(vec[(root + 9) % 12])
                    # 6 和弦需严格扩展音证据且防止与 m7 混淆
                    if evidence < 0.075 or root_support < 0.55:
                        score -= 0.40
                    if float(bass[t, (root + 9) % 12]) > float(bass[t, root]) + 0.08:
                        score -= 0.35
                if quality.startswith('sus'):
                    suspended = 2 if quality == 'sus2' else 5
                    suspended_support = float(vec[(root + suspended) % 12])
                    third_support = float(
                        max(vec[(root + 3) % 12], vec[(root + 4) % 12])
                    )
                    if suspended_support < 0.08:
                        score -= 0.30
                    score -= 0.50 * third_support
                    score += 0.50 * max(0.0, suspended_support - third_support)
                emissions[t, s] = score
            emissions[t, -1] = 1.5 if activity[t] < active_floor else -0.35

        dp = np.full_like(emissions, -np.inf)
        parent = np.zeros((count, state_count), dtype=np.int16)
        dp[0] = emissions[0]
        for t in range(1, count):
            for current, (root, quality) in enumerate(self.states):
                transitions = np.full(state_count, -0.46)
                transitions[current] = 0.20
                if root is not None:
                    for previous, (prev_root, prev_quality) in enumerate(self.states):
                        if prev_root == root:
                            transitions[previous] = -0.05 if prev_quality != quality else 0.20
                        elif prev_root is not None and (root - prev_root) % 12 in {5, 7}:
                            transitions[previous] = -0.24
                values = dp[t - 1] + transitions
                parent[t, current] = int(np.argmax(values))
                dp[t, current] = emissions[t, current] + values[parent[t, current]]
        path = np.zeros(count, dtype=int)
        path[-1] = int(np.argmax(dp[-1]))
        for t in range(count - 1, 0, -1):
            path[t - 1] = parent[t, path[t]]
        path[activity < active_floor] = state_count - 1
        path = self._smooth_short_runs(path, emissions, minimum_run_beats)

        labels = self._labels_from_path(path, chroma, bass)
        confidences = []
        for t, state_index in enumerate(path):
            _, quality = self.states[state_index]
            competing = np.delete(emissions[t], state_index)
            margin = float(emissions[t, state_index] - np.max(competing))
            confidence = float(np.clip(0.38 + 0.34 * margin, 0, 0.95))
            if quality == 'N' or confidence < self.confidence_threshold:
                labels[t] = 'N'
            confidences.append(confidence)
        return labels, confidences

    @staticmethod
    def _estimate_key(chroma):
        import numpy as np
        if not chroma.size:
            return {'key': 'Unknown', 'scale': 'Unknown', 'confidence': 0.0}
        average = np.mean(chroma, axis=0)
        major = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                            2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                            2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        candidates = []
        for root in range(12):
            candidates.append((np.corrcoef(average, np.roll(major, root))[0, 1], root, 'Major'))
            candidates.append((np.corrcoef(average, np.roll(minor, root))[0, 1], root, 'Minor'))
        score, root, scale = max(candidates)
        return {'key': NOTE_NAMES[root], 'scale': scale,
                'confidence': round(float(score), 3)}
