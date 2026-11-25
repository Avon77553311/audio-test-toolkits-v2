"""Chord lookup helpers shared by the library and virtual piano."""

from __future__ import annotations

from typing import Any

from tools.chord_studio import (
    BROWSABLE_CHORD_TYPES,
    CHORD_FORMULAS,
    NOTE_NAMES,
    SCALE_FORMULAS,
    get_chord_name,
)


INTERVAL_NAMES = {
    0: "根音",
    1: "小二度",
    2: "大二度",
    3: "小三度",
    4: "大三度",
    5: "纯四度",
    6: "增四/减五度",
    7: "纯五度",
    8: "小六度",
    9: "大六度",
    10: "小七度",
    11: "大七度",
    12: "八度",
    13: "小九度",
    14: "大九度",
    15: "增九度",
    17: "纯十一度",
    18: "增十一度",
    20: "小十三度",
    21: "大十三度",
}


def interval_labels(intervals: list[int]) -> list[str]:
    return [INTERVAL_NAMES.get(value, f"{value} 半音") for value in intervals]


def identify_chords(notes: set[int]) -> list[dict[str, Any]]:
    selected = {int(note) % 12 for note in notes}
    if len(selected) < 2:
        return []
    matches = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    for root_index, root in enumerate(NOTE_NAMES):
        for chord_type in BROWSABLE_CHORD_TYPES:
            formula = CHORD_FORMULAS[chord_type]
            pitch_classes = {
                (root_index + interval) % 12
                for interval in formula["intervals"]
            }
            identity = (root_index, tuple(sorted(pitch_classes)))
            if pitch_classes != selected or identity in seen:
                continue
            seen.add(identity)
            matches.append(
                {
                    "root": root,
                    "type": chord_type,
                    "name": get_chord_name(root, chord_type),
                    "display": formula["display"],
                }
            )
    return sorted(matches, key=lambda item: (len(item["name"]), item["name"]))


def compatible_scales(notes: set[int], limit: int = 12) -> list[str]:
    selected = {int(note) % 12 for note in notes}
    matches = []
    for root_index, root in enumerate(NOTE_NAMES):
        for scale in SCALE_FORMULAS.values():
            pitches = {
                (root_index + interval) % 12
                for interval in scale["intervals"]
            }
            if selected <= pitches:
                matches.append(f"{root} {scale['display']}")
    return matches[:limit]
