"""Dependency-free candidate fusion for optional multi-view audio experts.

The layer intentionally knows nothing about MIDI, chords, librosa, or cloud APIs.
Callers provide quality scores and a similarity function, which keeps routing and
fallback behaviour deterministic and straightforward to unit test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True)
class EnsembleCandidate:
    """One successful expert/view result."""

    name: str
    value: Any
    quality: float
    expert: str
    view: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnsembleDecision:
    """Result plus a JSON-safe observability trace."""

    value: Any
    chosen: str
    fallback: bool
    reason: str
    trace: Mapping[str, Any]


def select_candidate(
    candidates: Iterable[EnsembleCandidate],
    baseline_name: str,
    similarity: Callable[[Any, Any], float],
    *,
    minimum_gain: float = 0.02,
    consensus_weight: float = 0.12,
    conflict_weight: float = 0.10,
    failures: Mapping[str, str] | None = None,
    fallback_on_any_failure: bool = False,
) -> EnsembleDecision:
    """Rerank candidates, with minimum-gain gating and explicit fallback.

    Similarity must return a value in ``[0, 1]``. Agreement above 0.5 earns a
    consensus bonus; disagreement below 0.5 receives a conflict penalty. The
    baseline is always retained unless a candidate clears ``minimum_gain``.
    """
    items = list(candidates)
    by_name = {item.name: item for item in items}
    if baseline_name not in by_name:
        raise ValueError("baseline candidate is required")
    if len(by_name) != len(items):
        raise ValueError("candidate names must be unique")
    failures = dict(failures or {})

    rows = []
    for candidate in items:
        agreements = []
        for other in items:
            if other.name == candidate.name:
                continue
            try:
                score = float(similarity(candidate.value, other.value))
            except Exception:
                score = 0.0
            agreements.append(max(0.0, min(1.0, score)))
        consensus = sum(max(0.0, value - 0.5) * 2.0 for value in agreements)
        conflict = sum(max(0.0, 0.5 - value) * 2.0 for value in agreements)
        divisor = max(1, len(agreements))
        final_score = (float(candidate.quality)
                       + consensus_weight * consensus / divisor
                       - conflict_weight * conflict / divisor)
        rows.append({
            "name": candidate.name,
            "expert": candidate.expert,
            "view": candidate.view,
            "quality": round(float(candidate.quality), 6),
            "consensus": round(consensus / divisor, 6),
            "conflict": round(conflict / divisor, 6),
            "score": round(final_score, 6),
        })

    baseline = by_name[baseline_name]
    baseline_row = next(row for row in rows if row["name"] == baseline_name)
    winner_row = max(rows, key=lambda row: (row["score"], row["quality"]))
    winner = by_name[winner_row["name"]]
    gain = float(winner_row["score"]) - float(baseline_row["score"])

    fallback = False
    if failures and fallback_on_any_failure:
        winner = baseline
        fallback = True
        reason = "expert_failure"
    elif winner.name == baseline_name:
        fallback = len(items) > 1
        reason = "baseline_best" if fallback else "baseline_only"
    elif gain < float(minimum_gain) or winner.quality <= baseline.quality:
        winner = baseline
        fallback = True
        reason = "minimum_gain_not_met"
    else:
        reason = "consensus_rerank"

    trace = {
        "baseline": baseline_name,
        "chosen": winner.name,
        "fallback": fallback,
        "reason": reason,
        "minimum_gain": float(minimum_gain),
        "gain": round(gain, 6),
        "failures": failures,
        "candidates": rows,
    }
    return EnsembleDecision(winner.value, winner.name, fallback, reason, trace)
