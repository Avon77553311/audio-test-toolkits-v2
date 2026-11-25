"""Configured RunPod Demucs view adapter with explicit temporary lifetime."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class CloudStemView:
    """A selected, locally materialized stem returned by CloudStemClient."""

    path: Path
    stem: str
    model: str


class CloudStemViewProvider:
    """Context-managed ``input_path, **kwargs`` adapter for CloudStemClient.

    ``CloudStemClient.separate`` returns a directory containing WAV stems. This
    adapter selects one useful harmonic view while keeping that directory alive
    until the owning Tool finishes analysis/export.
    """

    def __init__(self, client_factory: Callable | None = None, temp_root=None):
        if client_factory is None:
            from cloud_inline import LocalStemClient
            client_factory = LocalStemClient
        self.client_factory = client_factory
        self.temp_root = temp_root
        self._temporary = None
        self._output_root = None

    def __enter__(self):
        self._temporary = tempfile.TemporaryDirectory(
            prefix='hybrid-cloud-stems-', dir=self.temp_root)
        self._output_root = Path(self._temporary.name)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._temporary is not None:
            self._temporary.cleanup()
        self._temporary = None
        self._output_root = None

    def __call__(self, input_path, **kwargs):
        if self._output_root is None:
            raise RuntimeError('CloudStemViewProvider 必须在 with 上下文中使用')
        purpose = kwargs.get('purpose', 'harmony')
        model = kwargs.get('model', 'htdemucs_6s')
        target = self.client_factory().separate(
            Path(input_path), self._output_root, model)
        target = Path(target)
        # MIDI 优先明确的乐器 Stem；和弦优先较完整的伴奏和声 Stem。
        preferred = (('piano', 'guitar', 'other') if purpose == 'midi'
                     else ('other', 'piano', 'guitar'))
        for stem in preferred:
            path = target / f'{stem}.wav'
            if path.is_file():
                return CloudStemView(path=path, stem=stem, model=model)
        available = sorted(path.name for path in target.glob('*.wav'))
        raise RuntimeError(
            f'RunPod Demucs 未返回可用的 harmonic/other/piano/guitar 视图：{available}')


def create_cloud_stem_view_provider(**kwargs):
    """Return an adapter only for an explicitly enabled, matching endpoint."""
    from tools import config
    if not config.RUNPOD_STEM_ENABLED or config.RUNPOD_STEM_PROTOCOL != 'demucs-v1':
        return None
    return CloudStemViewProvider(**kwargs)
