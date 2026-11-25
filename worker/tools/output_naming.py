"""Helpers for stable, user-facing output filenames."""
import re
from pathlib import PurePath


def safe_output_stem(filename, fallback="audio", max_length=80):
    """Return a filesystem-safe Unicode stem without upload IDs or path segments."""
    raw_name = str(filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    stem = PurePath(raw_name).stem
    stem = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._-")
    return (stem[:max_length].rstrip("._-") or fallback)
