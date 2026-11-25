"""Offline, windowed musical-instrument recognition and evidence fusion."""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.signal import resample_poly

from tools import load_audio
from tools.efficient_at import load_efficientat_mn10
from tools.instrument_roles import (
    extract_role_features,
    fuse_role_predictions,
    needs_role_features,
)

MODEL_NAME = "efficientat_mn10_as"
MODEL_SAMPLE_RATE = 32_000
WINDOW_SECONDS = 10.0
HOP_SECONDS = 5.0
MIN_STEM_RELATIVE_DB = -38.0


@dataclass(frozen=True)
class InstrumentSpec:
    instrument_id: str
    label: str
    indices: tuple[int, ...]
    compatible_stems: tuple[str, ...]
    threshold: float = 0.18


INSTRUMENT_SPECS = (
    InstrumentSpec(
        "vocals",
        "人声",
        (27, 28, 32, 33, 34, 35, 36, 37),
        ("vocals",),
        0.16,
    ),
    InstrumentSpec("choir", "合唱", (28,), ("vocals",), 0.16),
    InstrumentSpec("rap", "说唱", (36,), ("vocals",), 0.16),
    InstrumentSpec(
        "drums",
        "鼓组",
        (161, 162, 164, 165, 166, 167, 168, 171, 172),
        ("drums",),
        0.15,
    ),
    InstrumentSpec("drum_machine", "鼓机", (163,), ("drums", "other"), 0.16),
    InstrumentSpec(
        "percussion",
        "打击乐",
        (161, 169, 170, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183),
        ("drums", "other"),
        0.17,
    ),
    InstrumentSpec("timpani", "定音鼓", (169,), ("drums", "other"), 0.17),
    InstrumentSpec("tabla", "塔布拉鼓", (170,), ("drums", "other"), 0.17),
    InstrumentSpec("tambourine", "铃鼓", (174,), ("drums", "other"), 0.17),
    InstrumentSpec("gong", "锣", (177,), ("drums", "other"), 0.17),
    InstrumentSpec(
        "mallet_percussion",
        "键盘打击乐",
        (179, 180, 181, 182),
        ("other",),
        0.17,
    ),
    InstrumentSpec("steelpan", "钢鼓", (183,), ("other",), 0.17),
    InstrumentSpec("bass_guitar", "贝斯", (142,), ("bass",), 0.14),
    InstrumentSpec(
        "guitar",
        "吉他",
        (139, 140, 141, 143, 144, 145, 146),
        ("guitar", "other"),
        0.15,
    ),
    InstrumentSpec(
        "electric_guitar",
        "电吉他",
        (141,),
        ("guitar", "other"),
        0.16,
    ),
    InstrumentSpec(
        "acoustic_guitar",
        "原声吉他",
        (143,),
        ("guitar", "other"),
        0.16,
    ),
    InstrumentSpec(
        "slide_guitar",
        "滑棒吉他",
        (144,),
        ("guitar", "other"),
        0.18,
    ),
    InstrumentSpec("banjo", "班卓琴", (147,), ("guitar", "other"), 0.17),
    InstrumentSpec("sitar", "西塔琴", (148,), ("guitar", "other"), 0.17),
    InstrumentSpec("mandolin", "曼陀林", (149,), ("guitar", "other"), 0.17),
    InstrumentSpec("zither", "齐特琴", (150,), ("guitar", "other"), 0.17),
    InstrumentSpec("ukulele", "尤克里里", (151,), ("guitar", "other"), 0.17),
    InstrumentSpec(
        "keyboard",
        "键盘",
        (152, 153, 154, 155, 156, 157, 158, 159, 160),
        ("piano", "other"),
        0.14,
    ),
    InstrumentSpec("piano", "钢琴", (153,), ("piano", "other"), 0.16),
    InstrumentSpec(
        "electric_piano",
        "电钢琴",
        (154,),
        ("piano", "other"),
        0.16,
    ),
    InstrumentSpec(
        "organ",
        "风琴",
        (155, 156, 157),
        ("piano", "other"),
        0.16,
    ),
    InstrumentSpec(
        "synthesizer",
        "合成器",
        (158, 159),
        ("piano", "other"),
        0.15,
    ),
    InstrumentSpec("harpsichord", "羽管键琴", (160,), ("piano", "other"), 0.18),
    InstrumentSpec(
        "strings",
        "弦乐组",
        (184, 189, 190, 191, 192, 193, 194),
        ("other",),
        0.16,
    ),
    InstrumentSpec("violin", "小提琴", (191, 192), ("other",), 0.17),
    InstrumentSpec("cello", "大提琴", (193,), ("bass", "other"), 0.17),
    InstrumentSpec(
        "double_bass",
        "低音提琴",
        (194,),
        ("bass", "other"),
        0.17,
    ),
    InstrumentSpec(
        "brass",
        "铜管",
        (185, 186, 187, 188),
        ("other",),
        0.16,
    ),
    InstrumentSpec("french_horn", "圆号", (186,), ("other",), 0.17),
    InstrumentSpec("trumpet", "小号", (187,), ("other",), 0.17),
    InstrumentSpec("trombone", "长号", (188,), ("other",), 0.17),
    InstrumentSpec(
        "woodwind",
        "木管",
        (195, 196, 197, 198),
        ("other",),
        0.16,
    ),
    InstrumentSpec("flute", "长笛", (196,), ("other",), 0.17),
    InstrumentSpec("saxophone", "萨克斯", (197,), ("other",), 0.17),
    InstrumentSpec("clarinet", "单簧管", (198,), ("other",), 0.17),
    InstrumentSpec("harp", "竖琴", (199,), ("other",), 0.18),
    InstrumentSpec("harmonica", "口琴", (208,), ("other",), 0.18),
    InstrumentSpec("accordion", "手风琴", (209,), ("other",), 0.18),
    InstrumentSpec("bagpipes", "风笛", (210,), ("other",), 0.18),
    InstrumentSpec("didgeridoo", "迪吉里杜管", (211,), ("other",), 0.18),
    InstrumentSpec("shofar", "羊角号", (212,), ("other",), 0.18),
    InstrumentSpec("theremin", "特雷门琴", (213,), ("other",), 0.18),
    InstrumentSpec("singing_bowl", "颂钵", (214,), ("other",), 0.18),
)

SPEC_BY_ID = {item.instrument_id: item for item in INSTRUMENT_SPECS}
PARENT_CHILDREN = {
    "vocals": ("choir", "rap"),
    "guitar": (
        "electric_guitar",
        "acoustic_guitar",
        "slide_guitar",
        "banjo",
        "sitar",
        "mandolin",
        "zither",
        "ukulele",
    ),
    "keyboard": (
        "piano",
        "electric_piano",
        "organ",
        "synthesizer",
        "harpsichord",
    ),
    "percussion": (
        "timpani",
        "tabla",
        "tambourine",
        "gong",
        "mallet_percussion",
        "steelpan",
    ),
    "strings": ("violin", "cello", "double_bass"),
    "brass": ("french_horn", "trumpet", "trombone"),
    "woodwind": ("flute", "saxophone", "clarinet"),
}


class EfficientATPreprocessor:
    def __init__(self):
        import torch

        self.torch = torch
        self.window = torch.hann_window(800, periodic=False)
        self.preemphasis = torch.as_tensor([[[-0.97, 1.0]]])
        import torchaudio

        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            128,
            1024,
            MODEL_SAMPLE_RATE,
            0.0,
            15_000.0,
            vtln_low=100.0,
            vtln_high=-500.0,
            vtln_warp_factor=1.0,
        )
        self.mel_basis = torch.nn.functional.pad(mel_basis, (0, 1))

    def __call__(self, waveforms):
        torch = self.torch
        values = torch.nn.functional.conv1d(
            waveforms.unsqueeze(1),
            self.preemphasis,
        ).squeeze(1)
        spectrum = torch.stft(
            values,
            1024,
            hop_length=320,
            win_length=800,
            center=True,
            normalized=False,
            window=self.window,
            return_complex=True,
        )
        power = spectrum.abs().square()
        mel = torch.matmul(self.mel_basis, power)
        return ((mel + 0.00001).log() + 4.5) / 5.0


def _resample(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == MODEL_SAMPLE_RATE:
        return np.asarray(audio, dtype=np.float32)
    divisor = math.gcd(int(sample_rate), MODEL_SAMPLE_RATE)
    return np.asarray(
        resample_poly(
            audio,
            MODEL_SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        ),
        dtype=np.float32,
    )


def _window_starts(sample_count: int, window_size: int, hop_size: int) -> list[int]:
    if sample_count <= window_size:
        return [0]
    starts = list(range(0, sample_count - window_size + 1, hop_size))
    final = sample_count - window_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _load_windows(
    path: Path,
    reference_windows: list[dict[str, float]] | None = None,
) -> tuple[np.ndarray, list[dict[str, float]], float]:
    audio, sample_rate = load_audio(os.fspath(path), mono=True)
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32), copy=False)
    audio = _resample(audio, int(sample_rate))
    window_size = round(WINDOW_SECONDS * MODEL_SAMPLE_RATE)
    hop_size = round(HOP_SECONDS * MODEL_SAMPLE_RATE)
    if reference_windows is not None:
        reference_samples = round(reference_windows[-1]["end_sec"] * MODEL_SAMPLE_RATE)
        # Match the upper bound of validated lossy-codec tail padding.
        if abs(len(audio) - reference_samples) > round(0.2 * MODEL_SAMPLE_RATE):
            raise ValueError("乐器识别分轨时长与原始混音不一致")
    starts = (
        [round(item["start_sec"] * MODEL_SAMPLE_RATE) for item in reference_windows]
        if reference_windows is not None
        else _window_starts(len(audio), window_size, hop_size)
    )
    windows = np.zeros((len(starts), window_size), dtype=np.float32)
    metadata = []
    for index, start in enumerate(starts):
        reference_end = (
            round(reference_windows[index]["end_sec"] * MODEL_SAMPLE_RATE)
            if reference_windows is not None
            else start + window_size
        )
        end = max(start, min(len(audio), start + window_size, reference_end))
        windows[index, : end - start] = audio[start:end]
        metadata.append(
            dict(reference_windows[index])
            if reference_windows is not None
            else {
                "start_sec": start / MODEL_SAMPLE_RATE,
                "end_sec": end / MODEL_SAMPLE_RATE,
            }
        )
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    return windows, metadata, rms


def _predict_file(
    model,
    preprocessor: EfficientATPreprocessor,
    path: Path,
    *,
    device: str,
    cancel_event: threading.Event,
    progress: Callable[[float], None],
    reference_windows: list[dict[str, float]] | None = None,
    view: str = "mix",
) -> dict:
    import torch

    windows, metadata, rms = _load_windows(path, reference_windows)
    predictions = []
    batch_size = 16 if device == "cuda" else 8 if device == "mps" else 4
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            if cancel_event.is_set():
                raise RuntimeError("乐器识别任务已取消")
            batch = torch.from_numpy(windows[start : start + batch_size])
            features = preprocessor(batch).unsqueeze(1).to(device)
            logits, _embeddings = model(features)
            predictions.append(torch.sigmoid(logits.float()).cpu().numpy())
            progress(0.7 * min(1.0, (start + len(batch)) / max(1, len(windows))))
    scores = np.concatenate(predictions, axis=0)
    role_features = []
    window_rms = []
    for index, (window, metadata_item) in enumerate(zip(windows, metadata, strict=True)):
        if cancel_event.is_set():
            raise RuntimeError("乐器识别任务已取消")
        valid_samples = round(
            (metadata_item["end_sec"] - metadata_item["start_sec"]) * MODEL_SAMPLE_RATE
        )
        audio = window[:valid_samples]
        window_rms.append(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if len(audio) else 0.0)
        role_features.append(
            extract_role_features(audio, MODEL_SAMPLE_RATE, cancel_event=cancel_event)
            if needs_role_features(scores[index], view)
            else None
        )
        progress(0.7 + 0.3 * (index + 1) / max(1, len(windows)))
    return {
        "windows": metadata,
        "scores": scores,
        "rms": rms,
        "window_rms": window_rms,
        "role_features": role_features,
    }


def _aggregate_score(values: np.ndarray) -> float:
    if not len(values):
        return 0.0
    peak = float(np.max(values))
    sustained = float(np.quantile(values, 0.9))
    return max(sustained, min(0.75, peak * 0.75))


def _fused_windows(
    analyses: dict[str, dict],
    spec: InstrumentSpec,
    active_stems: set[str],
) -> tuple[np.ndarray, float, float, list[str]]:
    mix = np.max(
        analyses["mix"]["scores"][:, spec.indices],
        axis=1,
    )
    stem_arrays = []
    evidence = ["mix"]
    for stem in spec.compatible_stems:
        if stem not in active_stems or stem not in analyses:
            continue
        stem_arrays.append(
            np.max(analyses[stem]["scores"][:, spec.indices], axis=1)
        )
        evidence.append(stem)
    stem_scores = (
        np.max(np.stack(stem_arrays, axis=0), axis=0)
        if stem_arrays
        else np.zeros_like(mix)
    )
    fused = np.maximum(mix, stem_scores * 0.92)
    agreement = np.minimum(mix, stem_scores)
    fused = np.minimum(1.0, fused + 0.12 * agreement)
    return (
        fused,
        _aggregate_score(mix),
        _aggregate_score(stem_scores),
        evidence,
    )


def _segments(
    windows: list[dict[str, float]],
    scores: np.ndarray,
    threshold: float,
) -> list[dict[str, float]]:
    selected = [
        {
            "start_sec": float(window["start_sec"]),
            "end_sec": float(window["end_sec"]),
            "score": float(score),
        }
        for window, score in zip(windows, scores, strict=True)
        if float(score) >= threshold
    ]
    merged: list[dict[str, float]] = []
    for item in selected:
        if merged and item["start_sec"] <= merged[-1]["end_sec"]:
            merged[-1]["end_sec"] = max(
                merged[-1]["end_sec"],
                item["end_sec"],
            )
            merged[-1]["score"] = max(merged[-1]["score"], item["score"])
        else:
            merged.append(dict(item))
    return [
        {
            "start_sec": round(item["start_sec"], 2),
            "end_sec": round(item["end_sec"], 2),
            "score": round(item["score"], 3),
        }
        for item in merged
    ]


def _confidence_level(
    score: float,
    mix_score: float,
    stem_score: float,
) -> tuple[str, str]:
    if score >= 0.55 or (mix_score >= 0.32 and stem_score >= 0.32):
        return "high", "确定"
    if score >= 0.32:
        return "medium", "很可能"
    return "possible", "可能"


def fuse_instrument_predictions(
    analyses: dict[str, dict],
) -> dict:
    if "mix" not in analyses:
        raise ValueError("乐器识别缺少原始混音证据")
    mix_rms = max(float(analyses["mix"]["rms"]), 1e-9)
    stem_activity = {}
    active_stems = set()
    for stem, analysis in analyses.items():
        if stem == "mix":
            continue
        relative_db = 20.0 * math.log10(
            max(float(analysis["rms"]), 1e-12) / mix_rms
        )
        active = relative_db >= MIN_STEM_RELATIVE_DB
        stem_activity[stem] = {
            "relative_db": round(relative_db, 2),
            "active": active,
        }
        if active:
            active_stems.add(stem)

    candidates = {}
    for spec in INSTRUMENT_SPECS:
        scores, mix_score, stem_score, evidence = _fused_windows(
            analyses,
            spec,
            active_stems,
        )
        score = _aggregate_score(scores)
        peak = float(np.max(scores))
        if score < spec.threshold and peak < max(0.4, spec.threshold * 2):
            continue
        level, level_label = _confidence_level(score, mix_score, stem_score)
        candidates[spec.instrument_id] = {
            "instrument_id": spec.instrument_id,
            "label": spec.label,
            "score": round(score, 3),
            "confidence": level,
            "confidence_label": level_label,
            "evidence": evidence,
            "segments": _segments(
                analyses["mix"]["windows"],
                scores,
                spec.threshold,
            ),
        }

    for parent, children in PARENT_CHILDREN.items():
        parent_item = candidates.get(parent)
        child_items = [
            candidates[child]
            for child in children
            if child in candidates
        ]
        if parent_item and child_items:
            strong_children = [
                item
                for item in child_items
                if item["confidence"] != "possible"
                and item["score"] >= parent_item["score"] * 0.6
            ]
            if strong_children:
                parent_item["details"] = [
                    item["label"]
                    for item in sorted(
                        strong_children,
                        key=lambda value: value["score"],
                        reverse=True,
                    )[:3]
                ]
            for item in child_items:
                candidates.pop(item["instrument_id"], None)

    priority = {"high": 0, "medium": 1, "possible": 2}
    instruments = sorted(
        candidates.values(),
        key=lambda item: (priority[item["confidence"]], -item["score"]),
    )
    return {
        "model": MODEL_NAME,
        "taxonomy": "audioset_instrument_v1",
        "score_semantics": "uncalibrated_evidence",
        "window_sec": WINDOW_SECONDS,
        "hop_sec": HOP_SECONDS,
        "instruments": instruments,
        "stem_activity": stem_activity,
    }


def analyze_instrument_files(
    source: str | Path,
    tracks: dict[str, str | Path],
    weights: str | Path,
    *,
    device: str,
    cancel_event: threading.Event | None = None,
    progress: Callable[[int, str], None] | None = None,
) -> dict:
    if device not in {"cuda", "mps", "cpu"}:
        raise ValueError("乐器识别仅支持 CUDA、MPS 或 CPU")
    cancel_event = cancel_event or threading.Event()
    progress = progress or (lambda _value, _message: None)
    if cancel_event.is_set():
        raise RuntimeError("乐器识别任务已取消")
    model = load_efficientat_mn10(weights).to(device)
    preprocessor = EfficientATPreprocessor()
    sources = [("mix", Path(source)), *[
        (str(stem), Path(path))
        for stem, path in tracks.items()
    ]]
    analyses = {}
    total = len(sources)
    for index, (name, path) in enumerate(sources):
        progress(
            round(100 * index / max(1, total)),
            f"正在识别{name if name != 'mix' else '原始混音'}",
        )
        analyses[name] = _predict_file(
            model,
            preprocessor,
            path,
            device=device,
            cancel_event=cancel_event,
            progress=lambda fraction, offset=index: progress(
                round(100 * (offset + fraction) / max(1, total)),
                "正在分析乐器时间线" if fraction < 0.7 else "正在分析编曲角色",
            ),
            reference_windows=(
                analyses["mix"]["windows"] if index else None
            ),
            view=name,
        )
    result = fuse_instrument_predictions(analyses)
    result["role_analysis"] = fuse_role_predictions(analyses, cancel_event=cancel_event)
    progress(100, "乐器与编曲角色分析完成")
    return result
