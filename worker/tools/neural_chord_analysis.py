"""Structured chord recognition for local Apple Silicon and RunPod CUDA."""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from tools.harmony_evidence import measure_harmony
from tools.runtime_resources import chord_model_files, chord_onnx_model_files
from tools.chord_models import model_provenance

MODEL_NAME = "chord_structured_ensemble_v1"
SAMPLE_RATE = 22050
HOP_LENGTH = 512
CQT_BINS = 288
FEATURE_OFFSET = 18
FEATURE_BINS = 252
INFERENCE_WINDOW = 1024
INFERENCE_OVERLAP = 128
EVENT_INFERENCE_WINDOW = 384
EVENT_INFERENCE_OVERLAP = 96

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
TRIAD_TYPES = {
    "maj": 1,
    "min": 2,
    "sus4": 3,
    "sus2": 4,
    "dim": 5,
    "aug": 6,
}
QUALITY_INTERVALS = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "sus4": (0, 5, 7),
    "sus2": (0, 2, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "7sus4": (0, 5, 7, 10),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "dim7": (0, 3, 6, 9),
    "hdim7": (0, 3, 6, 10),
}


@lru_cache(maxsize=1)
def _onnx_sessions():
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(
        1,
        int(os.environ.get("CHORD_ONNX_THREADS", "1")),
    )
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return tuple(
        ort.InferenceSession(
            os.fspath(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        for path in chord_onnx_model_files()
    )


@dataclass(frozen=True)
class ChordCandidate:
    label: str
    component_indices: tuple[int, int, int, int, int, int]


def _display_label(root: int, suffix: str, bass_interval: int) -> str:
    root_name = NOTE_NAMES[root]
    suffixes = {
        "maj": "",
        "min": "m",
        "dim": "dim",
        "aug": "aug",
        "sus2": "sus2",
        "sus4": "sus4",
        "sus4(b7)": "7sus4",
        "13": "13",
        "11": "11",
        "min9": "m9",
        "9": "9",
        "maj9": "maj9",
        "dim7": "dim7",
        "hdim7": "m7b5",
        "min7": "m7",
        "7": "7",
        "maj7": "maj7",
    }
    base_suffix = suffix.split("/", 1)[0]
    label = root_name + suffixes.get(base_suffix, "")
    if "/" in suffix:
        label += "/" + NOTE_NAMES[(root + bass_interval) % 12]
    return label


def _build_candidates() -> tuple[ChordCandidate, ...]:
    # suffix, triad, bass interval, seventh, ninth, eleventh, thirteenth
    specifications = (
        ("min/b7", "min", 10, 0, 0, 0, 0),
        ("min/2", "min", 2, 0, 0, 0, 0),
        ("maj/b7", "maj", 10, 0, 0, 0, 0),
        ("maj/2", "maj", 2, 0, 0, 0, 0),
        ("sus4(b7)", "sus4", 0, 2, 0, 0, 0),
        ("sus2", "sus2", 0, 0, 0, 0, 0),
        ("sus4", "sus4", 0, 0, 0, 0, 0),
        ("13", "maj", 0, 2, 1, 1, 1),
        ("11", "maj", 0, 2, 1, 1, 0),
        ("min9", "min", 0, 2, 1, 0, 0),
        ("9", "maj", 0, 2, 1, 0, 0),
        ("maj9", "maj", 0, 1, 1, 0, 0),
        ("dim7", "dim", 0, 3, 0, 0, 0),
        ("hdim7", "dim", 0, 2, 0, 0, 0),
        ("min7", "min", 0, 2, 0, 0, 0),
        ("7", "maj", 0, 2, 0, 0, 0),
        ("maj7", "maj", 0, 1, 0, 0, 0),
        ("min/5", "min", 7, 0, 0, 0, 0),
        ("min/b3", "min", 3, 0, 0, 0, 0),
        ("maj/5", "maj", 7, 0, 0, 0, 0),
        ("maj/3", "maj", 4, 0, 0, 0, 0),
        ("dim", "dim", 0, 0, 0, 0, 0),
        ("aug", "aug", 0, 0, 0, 0, 0),
        ("min", "min", 0, 0, 0, 0, 0),
        ("maj", "maj", 0, 0, 0, 0, 0),
    )
    candidates = [ChordCandidate("N", (0, 0, 0, 0, 0, 0))]
    for root in range(12):
        for suffix, triad, bass, seventh, ninth, eleventh, thirteenth in specifications:
            triad_index = (TRIAD_TYPES[triad] - 1) * 12 + 1 + root
            bass_index = (root + bass) % 12 + 1
            candidates.append(
                ChordCandidate(
                    _display_label(root, suffix, bass),
                    (
                        triad_index,
                        bass_index,
                        seventh,
                        ninth,
                        eleventh,
                        thirteenth,
                    ),
                )
            )
    return tuple(candidates)


CANDIDATES = _build_candidates()


def model_label(index: int) -> str:
    """Return a canonical display label from the 301-state decoder."""
    index = int(index)
    return CANDIDATES[index].label if 0 <= index < len(CANDIDATES) else "N"


def _root_pitch_class(label: str) -> int | None:
    match = re.match(r"^([A-G](?:#)?)", str(label or ""))
    if not match:
        return None
    try:
        return NOTE_NAMES.index(match.group(1))
    except ValueError:
        return None


def _quality_for_label(label: str) -> str | None:
    match = re.match(r"^[A-G](?:#)?(.*?)(?:/[A-G](?:#)?)?$", str(label or ""))
    if not match:
        return None
    suffix = match.group(1)
    return {
        "": "maj",
        "m": "min",
        "sus4": "sus4",
        "sus2": "sus2",
        "dim": "dim",
        "aug": "aug",
        "7": "7",
        "maj7": "maj7",
        "m7": "min7",
        "7sus4": "7sus4",
        "dim7": "dim7",
        "m7b5": "hdim7",
    }.get(suffix)


def _bass_pitch_class(label: str) -> int | None:
    match = re.match(r"^([A-G](?:#)?)(?:-?\d+)?$", str(label or ""))
    return _root_pitch_class(match.group(1)) if match else None


def apply_bass_inversions(
    progression: list[dict],
    bass_progression: list[dict],
) -> tuple[list[dict], int]:
    """Add a slash bass when a stable external bass view supports an inversion."""
    result = []
    added = 0
    for chord in progression:
        item = dict(chord)
        label = str(item.get("chord") or "N")
        root = _root_pitch_class(label)
        quality = _quality_for_label(label)
        if "/" in label or root is None or quality not in QUALITY_INTERVALS:
            result.append(item)
            continue

        start = float(item.get("start_time", 0.0))
        end = start + float(item.get("duration_sec", 0.0))
        support: dict[int, float] = {}
        confidence_weight: dict[int, float] = {}
        for bass in bass_progression or []:
            bass_start = float(bass.get("start_time", 0.0))
            bass_end = bass_start + float(bass.get("duration_sec", 0.0))
            overlap = max(0.0, min(end, bass_end) - max(start, bass_start))
            pitch_class = _bass_pitch_class(str(bass.get("chord") or ""))
            if overlap <= 0 or pitch_class is None:
                continue
            confidence = max(0.0, min(1.0, float(bass.get("confidence", 0.0))))
            support[pitch_class] = support.get(pitch_class, 0.0) + overlap
            confidence_weight[pitch_class] = (
                confidence_weight.get(pitch_class, 0.0) + overlap * confidence
            )

        if support:
            bass_pc = max(support, key=lambda value: confidence_weight[value])
            coverage = support[bass_pc] / max(end - start, 1e-8)
            confidence = confidence_weight[bass_pc] / max(support[bass_pc], 1e-8)
            chord_tones = {
                (root + interval) % 12 for interval in QUALITY_INTERVALS[quality]
            }
            if (
                bass_pc != root
                and bass_pc in chord_tones
                and coverage >= 0.55
                and confidence >= 0.45
            ):
                item["chord"] = f"{label}/{NOTE_NAMES[bass_pc]}"
                evidence = dict(item.get("evidence") or {})
                evidence["bass_fusion"] = {
                    "bass": NOTE_NAMES[bass_pc],
                    "coverage": round(coverage, 3),
                    "confidence": round(confidence, 3),
                }
                item["evidence"] = evidence
                added += 1
        result.append(item)
    return result, added


class NeuralChordAnalyzer:
    """Run a five-fold structured neural ensemble with MPS and CPU fallback."""

    def __init__(
        self,
        *,
        analysis_mode: str = "mix",
        confidence_threshold: float = 0.18,
        min_chord_duration_ms: float = 500,
        bass_analyzer=None,
        progress=None,
    ):
        self.analysis_mode = analysis_mode
        self.confidence_threshold = float(confidence_threshold)
        self.min_chord_duration_ms = float(min_chord_duration_ms)
        self.bass_analyzer = bass_analyzer
        self.progress = progress or (lambda _value, _message: None)
        self._models = None
        self._backend = None
        self._device = None

    @staticmethod
    def _check_cancelled(cancel_event) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("和弦分析任务已取消")

    @staticmethod
    def _preferred_backend():
        from tools.chord_models import model_available
        has_torch=model_available('pt')
        has_onnx=model_available('onnx')
        try:
            import torch
        except ImportError:
            torch = None

        if has_torch and torch is not None and torch.cuda.is_available():
            return "torch", torch.device("cuda"), "cuda"
        if (
            has_torch and torch is not None
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        ):
            return "torch", torch.device("mps"), "mps"
        try:
            import onnxruntime  # noqa: F401

            if has_onnx:return "onnx", None, "onnx-cpu"
        except ImportError:
            pass
        if has_torch and torch is not None:
            return "torch", torch.device("cpu"), "cpu"
        raise RuntimeError("结构化和弦模型运行时不可用")

    def _load_models(self, backend, device):
        if backend == "onnx":
            models = list(_onnx_sessions())
        else:
            import torch

            models = []
            for model_path in chord_model_files():
                model = torch.jit.load(model_path, map_location="cpu")
                models.append(model.eval().to(device))
        self._models = models
        self._backend = backend
        self._device = device
        return models

    @staticmethod
    def _window_starts(
        frame_count: int,
        window_size: int = INFERENCE_WINDOW,
        overlap: int = INFERENCE_OVERLAP,
    ) -> list[int]:
        if frame_count <= window_size:
            return [0]
        stride = window_size - overlap
        starts = list(range(0, frame_count - window_size + 1, stride))
        final = frame_count - window_size
        if starts[-1] != final:
            starts.append(final)
        return starts

    def _run_models(
        self,
        features,
        backend,
        device,
        cancel_event,
        *,
        window_size: int = INFERENCE_WINDOW,
        overlap: int = INFERENCE_OVERLAP,
        progress_start: int = 28,
        progress_end: int = 60,
        progress_message: str = "结构化神经模型正在推理",
    ):
        import numpy as np

        self.progress(progress_start, "正在加载结构化神经模型")
        models = (
            self._load_models(backend, device)
            if (
                self._models is None
                or self._backend != backend
                or self._device != device
            )
            else self._models
        )
        frame_count = int(features.shape[0])
        totals = [
            np.zeros((frame_count, size), dtype=np.float32)
            for size in (73, 13, 4, 4, 3, 3)
        ]
        counts = np.zeros(frame_count, dtype=np.float32)
        window_count = len(
            self._window_starts(
                frame_count,
                window_size=window_size,
                overlap=overlap,
            )
        )
        total_steps = max(1, len(models) * window_count)
        completed_steps = 0
        last_progress = progress_start
        for model in models:
            for start in self._window_starts(
                frame_count,
                window_size=window_size,
                overlap=overlap,
            ):
                self._check_cancelled(cancel_event)
                end = min(frame_count, start + window_size)
                batch = features[start:end][None].astype(np.float32, copy=False)
                if backend == "onnx":
                    outputs = model.run(None, {"features": batch})
                    probabilities = [
                        self._numpy_softmax(output, axis=-1).squeeze(0)
                        for output in outputs
                    ]
                else:
                    import torch

                    tensor = torch.from_numpy(batch).to(device)
                    with torch.inference_mode():
                        outputs = model(tensor)
                        probabilities = [
                            torch.softmax(output, dim=-1)
                            .squeeze(0)
                            .to("cpu")
                            .numpy()
                            for output in outputs
                        ]
                for index, probability in enumerate(probabilities):
                    totals[index][start:end] += probability
                counts[start:end] += 1.0
                completed_steps += 1
                current_progress = progress_start + int(
                    (progress_end - progress_start)
                    * completed_steps
                    / total_steps
                )
                if current_progress > last_progress:
                    last_progress = current_progress
                    self.progress(current_progress, progress_message)
        return [
            values / np.maximum(counts[:, None], 1.0) for values in totals
        ]

    @staticmethod
    def _numpy_softmax(values, axis=-1):
        import numpy as np

        shifted = values - np.max(values, axis=axis, keepdims=True)
        exponent = np.exp(shifted)
        return exponent / np.maximum(
            np.sum(exponent, axis=axis, keepdims=True),
            1e-12,
        )

    def _predict_probabilities(self, features, cancel_event):
        backend, device, label = self._preferred_backend()
        try:
            return (
                self._run_models(
                    features,
                    backend,
                    device,
                    cancel_event,
                ),
                label,
                False,
            )
        except RuntimeError:
            if label != "mps" or (
                cancel_event is not None and cancel_event.is_set()
            ):
                raise
            self._models = None
            self._backend = None
            self._device = None
            import torch

            if hasattr(torch, "mps"):
                torch.mps.empty_cache()
            try:
                import onnxruntime  # noqa: F401

                fallback_backend = "onnx"
                fallback_device = None
                fallback_label = "onnx-cpu"
            except ImportError:
                fallback_backend = "torch"
                fallback_device = torch.device("cpu")
                fallback_label = "cpu"
            return (
                self._run_models(
                    features,
                    fallback_backend,
                    fallback_device,
                    cancel_event,
                ),
                fallback_label,
                True,
            )

    @staticmethod
    def _candidate_emissions(probabilities):
        import numpy as np

        indices = np.asarray(
            [candidate.component_indices for candidate in CANDIDATES],
            dtype=np.int64,
        )
        emissions = np.zeros(
            (probabilities[0].shape[0], len(CANDIDATES)),
            dtype=np.float32,
        )
        for component, values in enumerate(probabilities):
            emissions += np.log(
                np.maximum(values[:, indices[:, component]], 1e-9)
            )
        return emissions

    @staticmethod
    def _boundary_strength(chroma, onset, beat_frames):
        import numpy as np
        from scipy.ndimage import gaussian_filter1d

        frame_count = chroma.shape[1]
        novelty = np.zeros(frame_count, dtype=np.float32)
        if frame_count > 1:
            left = chroma[:, :-1]
            right = chroma[:, 1:]
            denominator = np.linalg.norm(left, axis=0) * np.linalg.norm(right, axis=0)
            novelty[1:] = 1.0 - np.sum(left * right, axis=0) / np.maximum(
                denominator, 1e-8
            )

        def scale(values):
            high = float(np.percentile(values, 95)) if values.size else 0.0
            return np.clip(values / max(high, 1e-8), 0.0, 1.0)

        score = 0.72 * scale(novelty) + 0.28 * scale(onset[:frame_count])
        for frame in beat_frames:
            left = max(0, int(frame) - 2)
            right = min(frame_count, int(frame) + 3)
            score[left:right] += 0.08
        return np.clip(gaussian_filter1d(score, sigma=1.0), 0.0, 1.0)

    @staticmethod
    def _decode(emissions, active, boundary_strength):
        import numpy as np

        frame_count, state_count = emissions.shape
        adjusted = emissions.copy()
        adjusted[~active, 1:] = -50.0
        adjusted[~active, 0] = 0.0
        dp = np.empty_like(adjusted)
        parent = np.zeros((frame_count, state_count), dtype=np.int16)
        dp[0] = adjusted[0]
        states = np.arange(state_count)
        for frame in range(1, frame_count):
            previous_best = int(np.argmax(dp[frame - 1]))
            transition_penalty = 8.5 - 4.5 * float(boundary_strength[frame])
            changed = dp[frame - 1, previous_best] - transition_penalty
            stay = dp[frame - 1]
            use_stay = stay >= changed
            dp[frame] = np.maximum(stay, changed) + adjusted[frame]
            parent[frame] = previous_best
            parent[frame, use_stay] = states[use_stay]
        path = np.zeros(frame_count, dtype=np.int64)
        path[-1] = int(np.argmax(dp[-1]))
        for frame in range(frame_count - 1, 0, -1):
            path[frame - 1] = parent[frame, path[frame]]
        path[~active] = 0
        return path, adjusted

    @staticmethod
    def _runs(path):
        import numpy as np

        starts = np.r_[0, np.flatnonzero(np.diff(path)) + 1]
        ends = np.r_[starts[1:], len(path)]
        return [
            (int(start), int(end), int(path[start]))
            for start, end in zip(starts, ends)
        ]

    @staticmethod
    def _path_from_runs(runs, frame_count):
        import numpy as np

        result = np.zeros(int(frame_count), dtype=np.int64)
        for start, end, state in runs:
            result[int(start) : int(end)] = int(state)
        return result

    @classmethod
    def _remove_short_runs(cls, path, emissions, active, minimum_frames):
        import numpy as np

        result = np.asarray(path, dtype=np.int64).copy()
        for _ in range(8):
            changed = False
            runs = cls._runs(result)
            for index, (start, end, _label) in enumerate(runs):
                if end - start >= minimum_frames or not np.any(active[start:end]):
                    continue
                neighbours = []
                if index > 0:
                    neighbours.append(runs[index - 1][2])
                if index + 1 < len(runs):
                    neighbours.append(runs[index + 1][2])
                neighbours = list(dict.fromkeys(neighbours))
                if not neighbours:
                    continue
                replacement = max(
                    neighbours,
                    key=lambda state: float(np.mean(emissions[start:end, state])),
                )
                result[start:end] = replacement
                changed = True
            if not changed:
                break
        return result

    @staticmethod
    def _boundary_peak(boundary_strength, frame, radius=2):
        left = max(0, int(frame) - int(radius))
        right = min(len(boundary_strength), int(frame) + int(radius) + 1)
        if left >= right:
            return 0.0
        return float(boundary_strength[left:right].max())

    @classmethod
    def _absorb_weak_same_root_fragments(
        cls,
        path,
        emissions,
        boundary_strength,
        beat_frames,
    ):
        """Absorb only a weak major/minor fragment within one quantized beat."""
        import numpy as np

        result = np.asarray(path, dtype=np.int64).copy()
        beat_frames = np.asarray(beat_frames, dtype=int)
        beat_length = (
            float(np.median(np.diff(beat_frames)))
            if beat_frames.size >= 2
            else 0.5 * SAMPLE_RATE / HOP_LENGTH
        )
        maximum_fragment_frames = max(1, int(round(beat_length)))
        absorbed = 0
        for _pass in range(4):
            changed = False
            runs = cls._runs(result)
            for index, (start, end, state) in enumerate(runs):
                fragment_frames = end - start
                if state == 0 or fragment_frames > maximum_fragment_frames:
                    continue
                label = model_label(state)
                root = _root_pitch_class(label)
                quality = _quality_for_label(label)
                if root is None or quality not in {"maj", "min"} or "/" in label:
                    continue

                neighbours = []
                if index > 0:
                    neighbours.append((runs[index - 1], start))
                if index + 1 < len(runs):
                    neighbours.append((runs[index + 1], end))
                candidates = []
                for neighbour, shared_boundary in neighbours:
                    neighbour_start, neighbour_end, neighbour_state = neighbour
                    neighbour_label = model_label(neighbour_state)
                    neighbour_quality = _quality_for_label(neighbour_label)
                    if (
                        neighbour_state == 0
                        or _root_pitch_class(neighbour_label) != root
                        or {quality, neighbour_quality} != {"maj", "min"}
                        or "/" in neighbour_label
                        or neighbour_end - neighbour_start
                        < max(fragment_frames + 4, int(round(1.5 * fragment_frames)))
                        or cls._boundary_peak(
                            boundary_strength,
                            shared_boundary,
                        )
                        > 0.35
                    ):
                        continue
                    advantage = float(
                        np.mean(
                            emissions[start:end, state]
                            - emissions[start:end, neighbour_state]
                        )
                    )
                    confidence = cls._segment_confidence(
                        emissions,
                        start,
                        end,
                        state,
                    )
                    if advantage <= 0.75 and confidence <= 0.68:
                        candidates.append(
                            (
                                neighbour_end - neighbour_start,
                                -advantage,
                                neighbour_state,
                            )
                        )
                if not candidates:
                    continue
                _length, _advantage, replacement = max(candidates)
                result[start:end] = replacement
                absorbed += 1
                changed = True
            if not changed:
                break
        return result, absorbed

    @classmethod
    def _restore_supported_passing_chords(
        cls,
        stable_path,
        event_path,
        event_emissions,
        boundary_strength,
        beat_frames,
        stable_minimum_frames,
    ):
        """Promote only short, strongly bounded local root changes."""
        import numpy as np

        result = np.asarray(stable_path, dtype=np.int64).copy()
        local_path = np.asarray(event_path, dtype=np.int64)
        beat_frames = np.asarray(beat_frames, dtype=int)
        beat_length = (
            float(np.median(np.diff(beat_frames)))
            if beat_frames.size >= 2
            else 0.5 * SAMPLE_RATE / HOP_LENGTH
        )
        minimum_frames = max(4, int(round(0.30 * beat_length)))
        maximum_frames = max(
            minimum_frames,
            min(
                max(1, int(stable_minimum_frames) - 1),
                int(round(1.35 * beat_length)),
            ),
        )
        restored = 0
        for start, end, candidate_state in cls._runs(local_path):
            if (
                candidate_state == 0
                or end - start < minimum_frames
                or end - start > maximum_frames
                or start <= 0
                or end >= len(result)
            ):
                continue
            stable_values = result[start:end]
            stable_state = int(
                np.bincount(
                    stable_values,
                    minlength=event_emissions.shape[1],
                ).argmax()
            )
            if candidate_state in {
                stable_state,
                int(result[start - 1]),
                int(result[end]),
            }:
                continue
            candidate_root = _root_pitch_class(model_label(candidate_state))
            surrounding_roots = {
                _root_pitch_class(model_label(int(result[start - 1]))),
                _root_pitch_class(model_label(int(result[end]))),
            }
            if (
                candidate_root is None
                or None in surrounding_roots
                or candidate_root in surrounding_roots
            ):
                continue

            left_boundary = cls._boundary_peak(boundary_strength, start)
            right_boundary = cls._boundary_peak(boundary_strength, end)
            if min(left_boundary, right_boundary) < 0.32:
                continue
            beat_radius = max(2, int(round(0.18 * beat_length)))
            if beat_frames.size and (
                int(np.min(np.abs(beat_frames - start))) > beat_radius
                or int(np.min(np.abs(beat_frames - end))) > beat_radius
            ):
                continue

            selected = event_emissions[start:end, candidate_state]
            stable = event_emissions[start:end, stable_state]
            top_share = float(
                np.mean(
                    np.argmax(event_emissions[start:end], axis=1)
                    == candidate_state
                )
            )
            confidence = cls._segment_confidence(
                event_emissions,
                start,
                end,
                candidate_state,
            )
            if (
                float(np.mean(selected - stable)) < 0.55
                or top_share < 0.65
                or confidence < 0.62
            ):
                continue
            result[start:end] = candidate_state
            restored += 1
        return result, restored

    @classmethod
    def _refined_runs(cls, path, boundary_strength, minimum_frames):
        runs = cls._runs(path)
        if len(runs) <= 1:
            return runs
        raw_boundaries = [run[1] for run in runs[:-1]]
        refined = []
        previous = 0
        radius = max(2, int(round(0.12 * SAMPLE_RATE / HOP_LENGTH)))
        for index, boundary in enumerate(raw_boundaries):
            next_boundary = (
                raw_boundaries[index + 1]
                if index + 1 < len(raw_boundaries)
                else len(path)
            )
            lower = max(previous + minimum_frames, boundary - radius)
            upper = min(next_boundary - minimum_frames, boundary + radius)
            if lower <= upper:
                local = boundary_strength[lower : upper + 1]
                boundary = lower + int(local.argmax())
            refined.append(boundary)
            previous = boundary
        boundaries = [0, *refined, len(path)]
        return [
            (boundaries[index], boundaries[index + 1], runs[index][2])
            for index in range(len(runs))
        ]

    @staticmethod
    def _segment_confidence(emissions, start, end, state):
        import numpy as np

        selected = emissions[start:end, state]
        competing = emissions[start:end].copy()
        competing[:, state] = -np.inf
        margin = selected - np.max(competing, axis=1)
        confidence = 1.0 / (1.0 + np.exp(-np.clip(margin, -12.0, 12.0)))
        return float(np.mean(confidence))

    @staticmethod
    def _robust_scale(values):
        import numpy as np

        values = np.asarray(values, dtype=np.float32)
        high = float(np.percentile(values, 95)) if values.size else 0.0
        return np.clip(values / max(high, 1e-8), 0.0, 1.0)

    @classmethod
    def _event_strength(cls, cqt, onset, frame_count):
        import numpy as np
        from scipy.ndimage import gaussian_filter1d, median_filter

        cqt = np.asarray(cqt[:, :frame_count], dtype=np.float32)
        horizontal = median_filter(cqt, size=(1, 17))
        vertical = median_filter(cqt, size=(17, 1))
        harmonic = cqt * horizontal**2 / np.maximum(
            horizontal**2 + vertical**2,
            1e-12,
        )
        log_harmonic = np.log1p(harmonic * 100.0)
        positive_flux = np.zeros(frame_count, dtype=np.float32)
        if frame_count > 1:
            positive_flux[1:] = np.maximum(
                log_harmonic[:, 1:] - log_harmonic[:, :-1],
                0.0,
            ).mean(axis=0)
        onset = np.asarray(onset[:frame_count], dtype=np.float32)
        score = (
            0.72 * cls._robust_scale(positive_flux)
            + 0.28 * cls._robust_scale(onset)
        )
        return gaussian_filter1d(score, sigma=0.75)

    @staticmethod
    def _tempo_adaptive_minimum_frames(
        beat_frames,
        requested_frames: int,
    ) -> int:
        import numpy as np

        requested_frames = max(1, int(requested_frames))
        beat_frames = np.asarray(beat_frames, dtype=int)
        if beat_frames.size < 2:
            return min(requested_frames, max(4, int(round(0.25 / (
                HOP_LENGTH / SAMPLE_RATE
            )))))
        beat_length = float(np.median(np.diff(beat_frames)))
        return min(
            requested_frames,
            max(4, int(round(0.45 * beat_length))),
        )

    @classmethod
    def _reattack_frames(
        cls,
        event_strength,
        beat_frames,
        active,
        minimum_frames,
    ) -> list[int]:
        import numpy as np

        event_strength = np.asarray(event_strength, dtype=np.float32)
        active = np.asarray(active, dtype=bool)
        beat_frames = np.asarray(beat_frames, dtype=int)
        if event_strength.size == 0:
            return []

        candidates: list[tuple[int, float]] = []
        beat_scores = []
        for beat in beat_frames:
            left = max(1, int(beat) - 2)
            right = min(len(event_strength) - 1, int(beat) + 2)
            if left > right:
                continue
            frame = left + int(np.argmax(event_strength[left : right + 1]))
            if active[frame]:
                beat_scores.append(float(event_strength[frame]))
                candidates.append((frame, float(event_strength[frame])))
        if beat_scores:
            threshold = max(0.50, float(np.percentile(beat_scores, 65)))
            candidates = [
                (frame, score) for frame, score in candidates if score >= threshold
            ]

        strong_threshold = max(
            0.68,
            float(np.percentile(event_strength[active], 90))
            if active.any()
            else 1.0,
        )
        for frame in range(2, len(event_strength) - 2):
            score = float(event_strength[frame])
            if (
                active[frame]
                and score >= strong_threshold
                and score == float(event_strength[frame - 2 : frame + 3].max())
            ):
                candidates.append((frame, score))

        selected = []
        minimum_gap = max(3, int(round(0.55 * minimum_frames)))
        for frame, score in sorted(candidates):
            if frame <= 1 or frame >= len(event_strength) - 1:
                continue
            if selected and frame - selected[-1][0] < minimum_gap:
                if score > selected[-1][1]:
                    selected[-1] = (frame, score)
                continue
            selected.append((frame, score))
        return [frame for frame, _score in selected]

    @staticmethod
    def _split_regions_at_reattacks(
        regions,
        reattack_frames,
        event_strength,
        frame_duration,
        minimum_frames=2,
    ):
        events = []
        for index, region in enumerate(regions):
            start = float(region["start_time"])
            end = start + float(region["duration_sec"])
            if end <= start:
                continue
            start_frame = int(round(start / frame_duration))
            end_frame = int(round(end / frame_duration))
            cuts = []
            minimum_frames = max(2, int(minimum_frames))
            if region.get("chord") not in {"N", "N/A", "", None}:
                for frame in reattack_frames:
                    if (
                        frame - (cuts[-1] if cuts else start_frame)
                        >= minimum_frames
                        and end_frame - frame >= minimum_frames
                    ):
                        cuts.append(frame)
            boundaries = [start_frame, *cuts, end_frame]
            for part, (left, right) in enumerate(
                zip(boundaries[:-1], boundaries[1:])
            ):
                item = dict(region)
                event_start = max(start, left * frame_duration)
                event_end = min(end, right * frame_duration)
                if event_end <= event_start:
                    continue
                item["start_time"] = round(event_start, 3)
                item["duration_sec"] = round(event_end - event_start, 3)
                item["event_boundary"] = (
                    "timeline_start"
                    if not events
                    else "reattack"
                    if part > 0
                    else "label_change"
                )
                if part > 0 and left < len(event_strength):
                    item["onset_confidence"] = round(
                        float(event_strength[left]),
                        3,
                    )
                events.append(item)
        return events

    @staticmethod
    def _normalize_event_timeline(events, duration):
        """Make a sequential event timeline exact after JSON-level rounding."""
        ordered = sorted(
            (dict(item) for item in events),
            key=lambda item: float(item.get("start_time", 0.0)),
        )
        normalized = []
        duration = max(0.0, float(duration))
        for item in ordered:
            start = max(0.0, min(duration, float(item.get("start_time", 0.0))))
            end = max(
                start,
                min(
                    duration,
                    start + max(0.0, float(item.get("duration_sec", 0.0))),
                ),
            )
            if normalized:
                previous = normalized[-1]
                previous_start = float(previous["start_time"])
                previous_end = previous_start + float(previous["duration_sec"])
                if abs(start - previous_end) <= 0.002:
                    start = previous_end
                elif start < previous_end:
                    previous["duration_sec"] = round(
                        max(0.0, start - previous_start),
                        6,
                    )
                    if previous["duration_sec"] <= 0:
                        normalized.pop()
            if end <= start:
                continue
            item["start_time"] = round(start, 6)
            item["duration_sec"] = round(end - start, 6)
            if item["duration_sec"] > 0:
                normalized.append(item)
        return normalized

    @staticmethod
    def _estimate_key(chroma, active):
        import numpy as np

        major = np.asarray(
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        )
        minor = np.asarray(
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        )
        values = chroma[:, active] if active.any() else chroma
        average = np.mean(values, axis=1) if values.size else np.zeros(12)
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
    def _merge_adjacent(progression):
        merged = []
        for segment in progression:
            if merged and merged[-1]["chord"] == segment["chord"]:
                previous_duration = merged[-1]["duration_sec"]
                end = segment["start_time"] + segment["duration_sec"]
                total_duration = previous_duration + segment["duration_sec"]
                merged[-1]["confidence"] = round(
                    (
                        merged[-1]["confidence"] * previous_duration
                        + segment["confidence"] * segment["duration_sec"]
                    ) / max(total_duration, 1e-9), 3
                )
                merged[-1]["duration_sec"] = round(
                    end - merged[-1]["start_time"], 3
                )
                evidence = dict(merged[-1].get("evidence") or {})
                evidence["frame_end"] = (segment.get("evidence") or {}).get("frame_end")
                merged[-1]["evidence"] = evidence
            else:
                merged.append(dict(segment))
        return merged

    def analyze(self, input_path, cancel_event=None):
        import librosa
        import numpy as np

        self._check_cancelled(cancel_event)
        self.progress(14, "正在读取音频")
        audio, sample_rate = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
        if audio.size == 0:
            raise ValueError("音频为空")
        self.progress(18, "正在提取和弦特征")
        cqt = np.abs(
            librosa.hybrid_cqt(
                audio,
                sr=sample_rate,
                bins_per_octave=36,
                fmin=librosa.note_to_hz("F#0"),
                n_bins=CQT_BINS,
                tuning=None,
                hop_length=HOP_LENGTH,
            )
        )
        features = cqt.T[:, FEATURE_OFFSET : FEATURE_OFFSET + FEATURE_BINS]
        features = features.astype(np.float32, copy=False)
        if not len(features):
            raise ValueError("无法提取神经和弦特征")
        self._check_cancelled(cancel_event)
        self.progress(24, "正在准备神经模型推理")

        probabilities, device, device_fallback = self._predict_probabilities(
            features, cancel_event
        )
        self.progress(61, "正在分析演奏事件边界")
        emissions = self._candidate_emissions(probabilities)
        frame_count = len(emissions)
        event_probabilities = probabilities
        event_context_fallback = False
        if frame_count > EVENT_INFERENCE_WINDOW:
            try:
                event_probabilities = self._run_models(
                    features,
                    self._backend,
                    self._device,
                    cancel_event,
                    window_size=EVENT_INFERENCE_WINDOW,
                    overlap=EVENT_INFERENCE_OVERLAP,
                    progress_start=62,
                    progress_end=78,
                    progress_message="正在细化演奏事件",
                )
            except RuntimeError:
                event_probabilities = probabilities
                event_context_fallback = True
        self.progress(80, "正在解码和弦时间线")
        local_event_emissions = self._candidate_emissions(event_probabilities)
        event_emissions = (
            0.75 * emissions + 0.25 * local_event_emissions
            if event_probabilities is not probabilities
            else local_event_emissions
        )

        rms = librosa.feature.rms(
            y=audio,
            frame_length=2048,
            hop_length=HOP_LENGTH,
            center=True,
        )[0]
        if len(rms) < frame_count:
            rms = np.pad(rms, (0, frame_count - len(rms)), mode="edge")
        rms = rms[:frame_count]
        rms_db = (
            librosa.amplitude_to_db(rms, ref=np.max)
            if np.any(rms)
            else np.full(frame_count, -80.0)
        )
        active = (rms_db >= -48.0) & (rms > 1e-7)
        harmony_evidence = measure_harmony(cqt[:, :frame_count], active)
        self._check_cancelled(cancel_event)
        chroma = librosa.feature.chroma_cqt(
            C=cqt,
            sr=sample_rate,
            hop_length=HOP_LENGTH,
            fmin=librosa.note_to_hz("F#0"),
            bins_per_octave=36,
            norm=1,
        )[:, :frame_count]
        onset = librosa.onset.onset_strength(
            y=audio,
            sr=sample_rate,
            hop_length=HOP_LENGTH,
            aggregate=np.median,
        )[:frame_count]
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset,
            sr=sample_rate,
            hop_length=HOP_LENGTH,
            trim=False,
        )
        beat_frames = np.asarray(beat_frames, dtype=int)
        beat_frames = beat_frames[(beat_frames > 0) & (beat_frames < frame_count)]
        boundary_strength = self._boundary_strength(chroma, onset, beat_frames)
        path, adjusted_emissions = self._decode(
            emissions, active, boundary_strength
        )

        frame_duration = HOP_LENGTH / sample_rate
        minimum_frames = max(
            1,
            int(math.ceil(self.min_chord_duration_ms / 1000.0 / frame_duration)),
        )
        path = self._remove_short_runs(
            path, adjusted_emissions, active, minimum_frames
        )
        path[~active] = 0
        path = self._path_from_runs(
            self._refined_runs(
                path,
                boundary_strength,
                minimum_frames,
            ),
            frame_count,
        )
        event_minimum_frames = self._tempo_adaptive_minimum_frames(
            beat_frames,
            minimum_frames,
        )
        event_path, adjusted_event_emissions = self._decode(
            event_emissions,
            active,
            boundary_strength,
        )
        event_path = self._remove_short_runs(
            event_path,
            adjusted_event_emissions,
            active,
            event_minimum_frames,
        )
        event_path[~active] = 0
        event_path = self._path_from_runs(
            self._refined_runs(
                event_path,
                boundary_strength,
                event_minimum_frames,
            ),
            frame_count,
        )
        path, weak_same_root_fragments_absorbed = (
            self._absorb_weak_same_root_fragments(
                path,
                adjusted_emissions,
                boundary_strength,
                beat_frames,
            )
        )
        path, passing_chords_restored = (
            self._restore_supported_passing_chords(
                path,
                event_path,
                adjusted_event_emissions,
                boundary_strength,
                beat_frames,
                minimum_frames,
            )
        )
        runs = self._runs(path)
        event_runs = self._runs(event_path)
        event_strength = self._event_strength(cqt, onset, frame_count)
        reattack_frames = self._reattack_frames(
            event_strength,
            beat_frames,
            active,
            event_minimum_frames,
        )

        duration = len(audio) / sample_rate
        progression = []
        final_chord_frames = np.zeros(frame_count, dtype=bool)
        threshold_rejections = 0
        raw_progression = []
        root_probabilities = probabilities[0]
        entropy = -np.sum(
            root_probabilities * np.log(np.maximum(root_probabilities, 1e-9)), axis=1
        ) / math.log(root_probabilities.shape[1])
        rejection_reason = harmony_evidence["reason"]
        hard_harmony_rejection = harmony_evidence["insufficient"] and (
            rejection_reason
            in {
                "silence",
                "single_pitch_dominant",
                "single_bass_with_percussion",
                "noise_like_spectrum",
            }
        )
        for start, end, state in runs:
            label = model_label(state)
            raw_label = label
            confidence = self._segment_confidence(
                adjusted_emissions, start, end, state
            )
            rejection_reason = None
            if label != "N" and confidence < self.confidence_threshold:
                label = "N"
                rejection_reason = "below_confidence_threshold"
                threshold_rejections += 1
            if hard_harmony_rejection:
                label = "N"
                rejection_reason = harmony_evidence["reason"]
            if label != "N":
                final_chord_frames[start:end] = True
            start_time = min(duration, start * frame_duration)
            end_time = min(duration, end * frame_duration)
            if end_time <= start_time:
                continue
            raw_progression.append({
                "chord": raw_label,
                "start_time": round(start_time, 3),
                "duration_sec": round(end_time - start_time, 3),
                "confidence": round(confidence, 3),
                "rejection_reason": rejection_reason,
                "normalized_root_entropy": round(float(entropy[start:end].mean()), 4),
            })
            progression.append(
                {
                    "chord": label,
                    "start_time": round(start_time, 3),
                    "duration_sec": round(end_time - start_time, 3),
                    "confidence": round(confidence, 3),
                    "evidence": {
                        "engine": MODEL_NAME,
                        "frame_start": start,
                        "frame_end": end,
                        "confidence_kind": "uncalibrated_margin_score",
                    },
                }
            )
        progression = self._merge_adjacent(progression)

        event_regions = []
        for start, end, state in event_runs:
            label = model_label(state)
            confidence = self._segment_confidence(
                adjusted_event_emissions,
                start,
                end,
                state,
            )
            event_rejection_reason = None
            if label != "N" and confidence < self.confidence_threshold:
                label = "N"
                event_rejection_reason = "below_confidence_threshold"
            if hard_harmony_rejection:
                label = "N"
                event_rejection_reason = harmony_evidence["reason"]
            start_time = min(duration, start * frame_duration)
            end_time = min(duration, end * frame_duration)
            if end_time <= start_time:
                continue
            evidence = {
                "engine": MODEL_NAME,
                "frame_start": start,
                "frame_end": end,
                "confidence_kind": "uncalibrated_margin_score",
                "context_window_frames": (
                    EVENT_INFERENCE_WINDOW
                    if event_probabilities is not probabilities
                    else min(frame_count, INFERENCE_WINDOW)
                ),
            }
            if harmony_evidence["insufficient"] and not hard_harmony_rejection:
                evidence["quality_warning"] = harmony_evidence["reason"]
            if event_rejection_reason:
                evidence["rejection_reason"] = event_rejection_reason
            event_regions.append(
                {
                    "chord": label,
                    "start_time": round(start_time, 3),
                    "duration_sec": round(end_time - start_time, 3),
                    "confidence": round(confidence, 3),
                    "evidence": evidence,
                }
            )
        chord_events = self._split_regions_at_reattacks(
            event_regions,
            reattack_frames,
            event_strength,
            frame_duration,
            event_minimum_frames,
        )
        chord_events = self._normalize_event_timeline(chord_events, duration)

        bass_failure = None
        inversion_count = 0
        if self.bass_analyzer is not None:
            try:
                self.progress(86, "正在解析低音与转位")
                bass_result = self.bass_analyzer.analyze(
                    input_path, cancel_event=cancel_event
                )
                progression, inversion_count = apply_bass_inversions(
                    progression,
                    bass_result.get("chord_progression", []),
                )
                chord_events, event_inversions = apply_bass_inversions(
                    chord_events,
                    bass_result.get("chord_progression", []),
                )
                inversion_count += event_inversions
            except Exception as exc:
                bass_failure = str(exc) or type(exc).__name__
        self.progress(90, "正在汇总神经和弦结果")

        harmonic_regions = progression
        active_count = int(np.count_nonzero(active))
        active_chord_count = int(np.count_nonzero(active & (path != 0)))
        raw_active_chord_coverage = (
            active_chord_count / active_count if active_count else 0.0
        )
        active_chord_coverage = (
            float(np.count_nonzero(active & final_chord_frames)) / active_count
            if active_count else 0.0
        )
        quality_gate_passed = active_chord_coverage >= 0.35
        abstained = hard_harmony_rejection or active_chord_coverage < 0.35
        counts = Counter(item["chord"] for item in harmonic_regions)
        confidences = [
            float(item["confidence"]) for item in harmonic_regions
        ]
        recognized_event_count = sum(
            item.get("chord") not in {None, "", "N", "N/A"}
            for item in chord_events
        )
        tempo_value = float(np.asarray(tempo).reshape(-1)[0]) if np.size(tempo) else 0
        return {
            "analysis_mode": self.analysis_mode,
            "key": (
                {"key": "Unknown", "scale": "Unknown", "confidence": 0.0}
                if hard_harmony_rejection
                else self._estimate_key(chroma, active)
            ),
            "tempo_bpm": round(
                tempo_value if math.isfinite(tempo_value) and tempo_value > 0 else 120.0,
                2,
            ),
            "beat_count": int(len(beat_frames)),
            "chord_progression": harmonic_regions,
            "chord_events": chord_events,
            "harmonic_regions": harmonic_regions,
            "raw_chord_progression": raw_progression,
            "recognition_status": (
                "insufficient_harmony" if hard_harmony_rejection
                else "no_confident_chords" if not final_chord_frames.any()
                else "uncertain_harmony"
                if harmony_evidence["insufficient"]
                else "recognized"
            ),
            "stats": {
                "duration_sec": round(duration, 2),
                "total_segments": len(harmonic_regions),
                "unique_chords": len(counts),
                "most_common_chords": [
                    {"chord": chord, "count": count}
                    for chord, count in counts.most_common(5)
                ],
                "avg_confidence": round(
                    sum(confidences) / len(confidences), 3
                )
                if confidences
                else 0.0,
                "event_count": len(chord_events),
                "recognized_event_count": recognized_event_count,
                "unrecognized_event_count": (
                    len(chord_events) - recognized_event_count
                ),
                "harmonic_region_count": len(harmonic_regions),
                "reattack_count": sum(
                    item.get("event_boundary") == "reattack"
                    for item in chord_events
                ),
            },
            "neural": {
                "model": MODEL_NAME,
                "weights": model_provenance('onnx' if device=='onnx-cpu' else 'pt'),
                "architecture": "Structured CNN-BiLSTM ensemble",
                "device": device,
                "runtime": (
                    "onnxruntime"
                    if device == "onnx-cpu"
                    else "pytorch"
                ),
                "mps_fallback": device_fallback,
                "frame_duration_ms": round(frame_duration * 1000.0, 3),
                "ensemble_size": 5,
                "vocabulary_size": len(CANDIDATES),
                "active_chord_coverage": round(active_chord_coverage, 4),
                "raw_active_chord_coverage": round(raw_active_chord_coverage, 4),
                "abstained": bool(abstained),
                "harmony_evidence": harmony_evidence,
                "harmony_warning": (
                    harmony_evidence["reason"]
                    if harmony_evidence["insufficient"]
                    and not hard_harmony_rejection
                    else None
                ),
                "threshold_rejections": threshold_rejections,
                "confidence_kind": "uncalibrated_margin_score",
                "quality_gate_passed": quality_gate_passed,
                "quality_gate_reason": (
                    harmony_evidence["reason"] if harmony_evidence["insufficient"]
                    else "sufficient_active_chord_coverage"
                    if quality_gate_passed
                    else "insufficient_active_chord_coverage"
                ),
                "bass_inversions_added": inversion_count,
                "bass_failure": bass_failure,
                "event_context_window_frames": (
                    EVENT_INFERENCE_WINDOW
                    if event_probabilities is not probabilities
                    else min(frame_count, INFERENCE_WINDOW)
                ),
                "event_context_fallback": event_context_fallback,
                "event_global_context_weight": (
                    0.75 if event_probabilities is not probabilities else 1.0
                ),
                "event_min_duration_ms": round(
                    event_minimum_frames * frame_duration * 1000.0,
                    2,
                ),
                "weak_same_root_fragments_absorbed": (
                    weak_same_root_fragments_absorbed
                ),
                "passing_chords_restored": passing_chords_restored,
            },
        }
