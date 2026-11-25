"""Resolve all project-local resources before importing legacy tools."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / 'worker'
RUNTIME = ROOT / 'runtime'

def configure():
    RUNTIME.mkdir(exist_ok=True)
    config = ROOT / '.env.local'
    allowed = {'OPENAI_API_KEY', 'RUNPOD_API_KEY', 'RUNPOD_RVC_ENABLED', 'RUNPOD_RVC_ENDPOINT_ID',
               'RUNPOD_SEED_VC_ENABLED', 'RUNPOD_SEED_VC_ENDPOINT_ID', 'RUNPOD_STEM_ENABLED',
               'RUNPOD_STEM_ENDPOINT_ID', 'RUNPOD_STEM_PROTOCOL', 'RUNPOD_STRUCTURE_ENDPOINT_ID',
               'RUNPOD_STRUCTURE_PROTOCOL', 'CHORD_MODEL_DIR', 'CHORD_ONNX_MODEL_DIR',
               'RUBBERBAND_R3_BIN', 'STUDIO_ENABLE_PAID', 'STUDIO_USE_NATIVE_SONG_INFO'}
    allowed |= {'INSTRUMENT_MODEL_PATH','RUNPOD_CHORD_ENABLED','RUNPOD_CHORD_PROTOCOL','RUNPOD_STRUCTURE_ENABLED',
                'RUNPOD_MAX_AUDIO_SECONDS','RUNPOD_MAX_AUDIO_BYTES','RUNPOD_TIMEOUT',
                'RUNPOD_POLL_INTERVAL','RUNPOD_MAX_POLL'}
    if config.is_file():
        for line in config.read_text(encoding='utf-8-sig').splitlines():
            key, separator, value = line.strip().partition('=')
            if separator and key in allowed and key not in os.environ:
                os.environ[key] = value.strip().strip('"').strip("'")
    os.environ['PATH'] = str(RUNTIME / 'bin') + os.pathsep + os.environ.get('PATH', '')
    os.environ.setdefault('TORCH_HOME', str(RUNTIME / 'torch'))
    os.environ.setdefault('NUMBA_CACHE_DIR', str(RUNTIME / 'numba'))
    os.environ.setdefault('AUDIO_TOOLKIT_RUNTIME_DIR', str(RUNTIME / 'scratch'))
    rubber = RUNTIME / 'bin' / ('rubberband-r3.exe' if os.name == 'nt' else 'rubberband-r3')
    if rubber.is_file():
        os.environ.setdefault('RUBBERBAND_R3_BIN', str(rubber))
    os.environ.setdefault('OMP_NUM_THREADS', '2')

configure()
