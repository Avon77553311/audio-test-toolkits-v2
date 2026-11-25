"""Demucs 二进制结果的一次性、跨进程回调登记与安全落盘。"""
from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

TASK_ID_RE = re.compile(r"^[0-9a-f]{32}$")


@dataclass
class CallbackEntry:
    token: str
    path: Path
    expires_at: float
    event: threading.Event
    source_path: Path | None = None
    received: bool = False
    claimed: bool = False
    error: str | None = None


class StemCallbackStore:
    def __init__(self):
        self._entries: dict[str, CallbackEntry] = {}
        self._lock = threading.Lock()
        self._output_root: Path | None = None

    def configure(self, output_root: str | Path) -> None:
        self._output_root = Path(output_root).resolve()
        (self._output_root / ".stem-callbacks").mkdir(parents=True, exist_ok=True)

    def _directory(self, root: str | Path | None = None) -> Path:
        if root is not None:
            directory = Path(root).resolve() / ".stem-callbacks"
        elif self._output_root is not None:
            directory = self._output_root / ".stem-callbacks"
        else:
            raise RuntimeError("Stem callback store 尚未配置输出目录")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def _metadata_path(entry_path: Path) -> Path:
        return entry_path.with_suffix(".json")

    def _persist(self, entry: CallbackEntry) -> None:
        metadata = self._metadata_path(entry.path)
        temporary = metadata.with_suffix(f".tmp-{secrets.token_hex(8)}")
        payload = {
            "token": entry.token,
            "path": os.fspath(entry.path),
            "expires_at": entry.expires_at,
            "source_path": os.fspath(entry.source_path) if entry.source_path else None,
            "received": entry.received,
            "error": entry.error,
        }
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(payload, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, metadata)

    def _load(self, task_id: str) -> CallbackEntry | None:
        if self._output_root is None:
            return None
        metadata = self._directory() / f"{task_id}.json"
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            path = Path(payload["path"]).resolve()
            root = self._output_root
            if os.path.commonpath([root, path]) != os.fspath(root):
                return None
            source_value = payload.get("source_path")
            source_path = Path(source_value).resolve() if source_value else None
            if (
                source_path is not None
                and os.path.commonpath([root, source_path]) != os.fspath(root)
            ):
                return None
            return CallbackEntry(
                token=str(payload["token"]),
                path=path,
                expires_at=float(payload["expires_at"]),
                event=threading.Event(),
                source_path=source_path,
                received=bool(payload.get("received")),
                error=payload.get("error"),
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    def cleanup_expired(self, now: float | None = None) -> int:
        now = time.time() if now is None else now
        with self._lock:
            expired = {
                task_id: entry
                for task_id, entry in self._entries.items()
                if now > entry.expires_at
            }
            metadata_files = (
                self._directory().glob("*.json")
                if self._output_root is not None
                else ()
            )
            for metadata in metadata_files:
                task_id = metadata.stem
                if task_id in expired or not TASK_ID_RE.fullmatch(task_id):
                    continue
                entry = self._load(task_id)
                if entry is not None and now > entry.expires_at:
                    expired[task_id] = entry
            for task_id in expired:
                self._entries.pop(task_id, None)
        for entry in expired.values():
            for path in (
                entry.path,
                self._metadata_path(entry.path),
                entry.path.with_suffix(".claim"),
            ):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        return len(expired)

    def register(
        self, root, public_base_url: str, ttl_seconds: float, source_path=None
    ) -> tuple[str, str, CallbackEntry]:
        self._output_root = Path(root).resolve()
        self.cleanup_expired()
        base = public_base_url.strip().rstrip("/")
        if not base.startswith("https://") and not base.startswith("http://localhost"):
            raise ValueError("STEM_CALLBACK_PUBLIC_BASE_URL 必须是明确配置的 HTTPS 公共基址")
        task_id = secrets.token_hex(16)
        token = secrets.token_urlsafe(32)
        directory = self._directory(root)
        resolved_source = Path(source_path).resolve() if source_path else None
        if (
            resolved_source is not None
            and os.path.commonpath([self._output_root, resolved_source])
            != os.fspath(self._output_root)
        ):
            raise ValueError("Stem 回调输入文件必须位于输出根目录内")
        entry = CallbackEntry(
            token, directory / f"{task_id}.zip", time.time() + ttl_seconds,
            threading.Event(), source_path=resolved_source)
        self._persist(entry)
        with self._lock:
            self._entries[task_id] = entry
        url = f"{base}/api/stems/callback/{task_id}?token={quote(token, safe='')}"
        return task_id, url, entry

    def input_url(self, task_id: str, public_base_url: str, entry: CallbackEntry) -> str:
        base = public_base_url.strip().rstrip("/")
        return (
            f"{base}/api/stems/input/{task_id}"
            f"?token={quote(entry.token, safe='')}"
        )

    def get_input(self, task_id: str, token: str) -> tuple[int, str, Path | None]:
        if not TASK_ID_RE.fullmatch(task_id):
            return 404, "输入任务不存在", None
        with self._lock:
            entry = self._entries.get(task_id) or self._load(task_id)
            expected = entry.token if entry else secrets.token_urlsafe(32)
            if not entry or not hmac.compare_digest(
                expected.encode(), (token or "").encode()
            ):
                return 403, "输入凭证无效", None
            if time.time() > entry.expires_at:
                return 410, "输入凭证已过期", None
            if not entry.source_path or not entry.source_path.is_file():
                return 404, "输入文件不存在", None
            source_path = entry.source_path.resolve()
            if (
                self._output_root is None
                or os.path.commonpath([self._output_root, source_path])
                != os.fspath(self._output_root)
            ):
                return 404, "输入文件不存在", None
            entry.source_path = source_path
            self._entries[task_id] = entry
            return 200, "", source_path

    def receive(self, task_id: str, token: str, stream, content_length: int | None,
                max_bytes: int) -> tuple[int, str]:
        if not TASK_ID_RE.fullmatch(task_id):
            return 404, "回调任务不存在"
        with self._lock:
            entry = self._entries.get(task_id) or self._load(task_id)
            if entry is not None:
                self._entries[task_id] = entry
            expected = entry.token if entry else secrets.token_urlsafe(32)
            token_ok = hmac.compare_digest(expected.encode(), (token or "").encode())
            if not entry or not token_ok:
                return 403, "回调凭证无效"
            if time.time() > entry.expires_at:
                return 410, "回调凭证已过期"
            if entry.received:
                return 409, "回调结果已上传"
            if content_length is not None and content_length > max_bytes:
                return 413, "回调文件超过大小限制"

        claim = entry.path.with_suffix(".claim")
        claim_fd = None
        claim_acquired = False
        temporary = entry.path.with_suffix(f".tmp-{secrets.token_hex(8)}")
        try:
            try:
                claim_fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                return 409, "回调结果正在处理"
            claim_acquired = True
            os.close(claim_fd)
            claim_fd = None

            # 另一进程可能已在本进程读取缓存后完成上传；抢到跨进程锁后必须复核磁盘。
            disk_entry = self._load(task_id)
            if disk_entry is None:
                return 404, "回调任务不存在"
            if not hmac.compare_digest(
                disk_entry.token.encode(), (token or "").encode()
            ):
                return 403, "回调凭证无效"
            if disk_entry.received:
                return 409, "回调结果已上传"
            if time.time() > disk_entry.expires_at:
                return 410, "回调凭证已过期"
            with self._lock:
                cached_entry = self._entries.get(task_id)
                if cached_entry is None:
                    entry = disk_entry
                    self._entries[task_id] = entry
                else:
                    cached_entry.token = disk_entry.token
                    cached_entry.path = disk_entry.path
                    cached_entry.expires_at = disk_entry.expires_at
                    cached_entry.source_path = disk_entry.source_path
                    cached_entry.received = disk_entry.received
                    cached_entry.error = disk_entry.error
                    entry = cached_entry
                entry.claimed = True

            total = 0
            with temporary.open("xb") as output:
                while True:
                    chunk = stream.read(min(1024 * 1024, max_bytes + 1 - total))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise OverflowError
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if total == 0:
                raise ValueError("回调文件为空")
            os.replace(temporary, entry.path)
            entry.received = True
            entry.claimed = True
            entry.error = None
            self._persist(entry)
            entry.event.set()
            return 204, ""
        except OverflowError:
            entry.error = "回调文件超过大小限制"
            self._persist(entry)
            entry.event.set()
            return 413, entry.error
        except Exception as exc:
            entry.claimed = False
            entry.error = f"回调文件写入失败：{exc}"
            self._persist(entry)
            return 400, entry.error
        finally:
            if claim_fd is not None:
                os.close(claim_fd)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            if claim_acquired:
                try:
                    claim.unlink()
                except FileNotFoundError:
                    pass

    def check_received(self, task_id: str) -> tuple[bool, str | None, Path | None]:
        with self._lock:
            entry = self._entries.get(task_id)
            disk_entry = self._load(task_id)
            if disk_entry is not None:
                if entry is not None:
                    entry.received = disk_entry.received
                    entry.error = disk_entry.error
                    if entry.received or entry.error:
                        entry.event.set()
                else:
                    entry = disk_entry
                    self._entries[task_id] = entry
            if entry is None:
                return False, "回调任务不存在", None
            return entry.received, entry.error, entry.path

    def discard(self, task_id: str) -> None:
        with self._lock:
            entry = self._entries.pop(task_id, None) or self._load(task_id)
        if entry:
            for path in (entry.path, self._metadata_path(entry.path), entry.path.with_suffix(".claim")):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


stem_callback_store = StemCallbackStore()
