"""基于词级时间戳的无损工作区音频局部编辑。"""

import os
import shutil
import threading
import time
import uuid
from copy import deepcopy
from pathlib import Path


class AudioEditError(ValueError):
    pass


def normalize_ranges(ranges, duration):
    """校验、排序并合并重叠时间区间。"""
    import numpy as np

    if not isinstance(ranges, list) or not ranges:
        raise AudioEditError("至少选择一个词或时间区间")
    cleaned = []
    for item in ranges:
        if not isinstance(item, dict):
            raise AudioEditError("时间区间格式无效")
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AudioEditError("时间区间必须包含数字 start/end") from exc
        if (
            not np.isfinite(start)
            or not np.isfinite(end)
            or start < 0
            or end <= start
            or end > duration + 0.05
        ):
            raise AudioEditError("时间区间超出音频范围")
        cleaned.append((max(0.0, start), min(duration, end)))
    cleaned.sort()
    merged = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def edit_audio(input_path, output_path, ranges, operation="mute", fade_ms=12):
    """执行消音或删除；在编辑边界应用短淡入淡出以避免爆音。"""
    import numpy as np
    import soundfile as sf
    from tools import load_audio

    if operation not in {"mute", "delete"}:
        raise AudioEditError("operation 仅支持 mute 或 delete")
    try:
        fade_ms = float(fade_ms)
    except (TypeError, ValueError) as exc:
        raise AudioEditError("fade_ms 必须为数字") from exc
    if not 0 <= fade_ms <= 200:
        raise AudioEditError("fade_ms 必须在 0 到 200 之间")
    audio, sample_rate = load_audio(os.fspath(input_path), mono=False)
    duration = len(audio) / sample_rate
    intervals = normalize_ranges(ranges, duration)
    fade = int(sample_rate * fade_ms / 1000)

    if operation == "mute":
        result = audio.copy()
        for start, end in intervals:
            left, right = (
                int(start * sample_rate),
                min(len(result), int(end * sample_rate)),
            )
            width = min(fade, (right - left) // 2)
            if width:
                result[left : left + width] *= np.linspace(1, 0, width, endpoint=False)[
                    :, None
                ]
                result[right - width : right] *= np.linspace(
                    0, 1, width, endpoint=False
                )[:, None]
            result[left + width : right - width] = 0
    else:
        pieces, cursor = [], 0
        for start, end in intervals:
            left, right = (
                int(start * sample_rate),
                min(len(audio), int(end * sample_rate)),
            )
            piece = audio[cursor:left].copy()
            width = min(fade, len(piece))
            if width:
                piece[-width:] *= np.linspace(1, 0, width, endpoint=False)[:, None]
            pieces.append(piece)
            cursor = right
        tail = audio[cursor:].copy()
        width = min(fade, len(tail))
        if width:
            tail[:width] *= np.linspace(0, 1, width, endpoint=False)[:, None]
        pieces.append(tail)
        result = (
            np.concatenate(pieces)
            if pieces
            else np.empty((0, audio.shape[1]), dtype=np.float32)
        )
        if not len(result):
            raise AudioEditError("不能删除整段音频")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp.wav")
    try:
        sf.write(temporary, result, sample_rate, subtype="PCM_16")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "duration_sec": round(len(result) / sample_rate, 3),
        "sample_rate": sample_rate,
    }


class EditSessionStore:
    """线程安全的进程内编辑会话；文件限定在输出根目录。"""

    def __init__(self, root, ttl_seconds=24 * 3600):
        self.root = Path(root).resolve()
        self.ttl_seconds = ttl_seconds
        self._sessions = {}
        self._lock = threading.RLock()

    def create(self, source_path, words, user_id=None):
        import soundfile as sf
        from tools import load_audio

        user_id = str(user_id or "local")
        session_id = uuid.uuid4().hex
        folder = self.root / "audio_sessions" / user_id / session_id
        folder.mkdir(parents=True, exist_ok=False)
        target = folder / "original.wav"
        temporary = folder / f".original.{uuid.uuid4().hex}.tmp.wav"
        try:
            audio, sr = load_audio(os.fspath(source_path), mono=False)
            sf.write(temporary, audio, sr, subtype="PCM_16")
            os.replace(temporary, target)
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        finally:
            temporary.unlink(missing_ok=True)
        with self._lock:
            self._sessions[session_id] = {
                "folder": folder,
                "history": [target],
                "word_history": [deepcopy(words)],
                "user_id": user_id,
                "updated": time.time(),
                "lock": threading.Lock(),
            }
        return session_id, target

    def is_owner(self, session_id, user_id):
        try:
            session = self._get(session_id)
            return session.get("user_id") == user_id
        except AudioEditError:
            return False

    def _get(self, session_id):
        if (
            not isinstance(session_id, str)
            or len(session_id) != 32
            or not session_id.isalnum()
        ):
            raise AudioEditError("会话 ID 无效")
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise AudioEditError("编辑会话不存在或已过期")
            session["updated"] = time.time()
            return session

    def apply(self, session_id, ranges, operation, fade_ms):
        session = self._get(session_id)
        with session["lock"]:
            with self._lock:
                source = session["history"][-1]
                output = session["folder"] / f"edit_{len(session['history']):03d}.wav"
            metadata = edit_audio(source, output, ranges, operation, fade_ms)
            words = deepcopy(session["word_history"][-1])
            if operation == "delete":
                import soundfile as sf

                info = sf.info(source)
                intervals = normalize_ranges(ranges, info.duration)
                intervals = [(int(a * info.samplerate) / info.samplerate,
                              int(b * info.samplerate) / info.samplerate)
                             for a, b in intervals]

                def mapped(t):
                    return t - sum(max(0, min(t, b) - a) for a, b in intervals)

                retained = []
                for word in words:
                    word.setdefault("original_start", word["start"])
                    word.setdefault("original_end", word["end"])
                    start, end = mapped(word["start"]), mapped(word["end"])
                    if end > start + 1 / info.samplerate:
                        word.update(start=start, end=end)
                        retained.append(word)
                words = retained
            with self._lock:
                session["history"].append(output)
                session["word_history"].append(words)
                session["updated"] = time.time()
            metadata["words"] = deepcopy(words)
            return output, metadata

    def undo(self, session_id):
        session = self._get(session_id)
        with session["lock"]:
            with self._lock:
                if len(session["history"]) <= 1:
                    raise AudioEditError("没有可撤销的编辑")
                removed = session["history"].pop()
                session["word_history"].pop()
                current = session["history"][-1]
            removed.unlink(missing_ok=True)
            return current

    def current(self, session_id):
        return self._get(session_id)["history"][-1]

    def words(self, session_id):
        session = self._get(session_id)
        with session["lock"]:
            return deepcopy(session["word_history"][-1])

    def cleanup(self):
        cutoff = time.time() - self.ttl_seconds
        with self._lock:
            expired = [
                key
                for key, value in self._sessions.items()
                if value["updated"] < cutoff
            ]
            for key in expired:
                shutil.rmtree(self._sessions.pop(key)["folder"], ignore_errors=True)
