"""Shared model discovery for capability reporting and actual backend selection."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

WORKER=Path(__file__).resolve().parents[1]
RELEASE='music-x-lab-ismir2019-v1'

def _files(folder,extension):
    return tuple(folder/f'structured_chord_fold{i}.{extension}' for i in range(5))

@lru_cache(maxsize=16)
def _verify(paths,signature,manifest_path,manifest_signature):
    try:
        manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8')) if manifest_signature else None
        for path in map(Path,paths):
            with path.open('rb') as stream:header=stream.read(128)
            if header.startswith(b'version https://git-lfs.github.com/spec/v1'):return False
            if path.suffix=='.pt' and not header.startswith(b'PK'):return False
            if manifest is not None:
                key=path.parent.name+'/'+path.name
                if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['files'].get(key):return False
        return True
    except (OSError,ValueError,KeyError,TypeError):return False

def complete(paths):
    try:
        stats=tuple((p.stat().st_size,p.stat().st_mtime_ns) for p in paths)
        if any(size<=1024 for size,_ in stats):return False
        manifest=paths[0].parent.parent/'manifest.json'
        metadata=(manifest.stat().st_size,manifest.stat().st_mtime_ns) if manifest.is_file() else ()
        return _verify(tuple(map(str,paths)),stats,str(manifest),metadata)
    except OSError:return False

def model_paths(extension):
    env='CHORD_ONNX_MODEL_DIR' if extension=='onnx' else 'CHORD_MODEL_DIR'
    configured=os.environ.get(env,'').strip()
    kind='chord_onnx' if extension=='onnx' else 'chord'
    if configured:return _files(Path(configured).expanduser().resolve(),extension)
    installed=_files(WORKER.parent/'runtime/models'/RELEASE/kind,extension)
    if complete(installed):return installed
    return _files(WORKER/'resources/models'/kind,extension)

def model_available(extension):return complete(model_paths(extension))

def model_provenance(extension):
    manifest=model_paths(extension)[0].parent.parent/'manifest.json'
    try:
        data=json.loads(manifest.read_text(encoding='utf-8'))
        return {key:data[key] for key in ('model','upstream','revision','original_lfs_weights_recovered') if key in data}
    except (OSError,ValueError,TypeError):return {'model':'external-five-fold','original_lfs_weights_recovered':False}
