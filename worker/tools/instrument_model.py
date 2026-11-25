"""Locate and verify the exact EfficientAT MN10 model referenced by original Git LFS."""
from functools import lru_cache
import hashlib
import os
from pathlib import Path

NAME='mn10_as_mAP_471.pt'
RELEASE='efficientat-mn10-as-v1'
SHA256='0bd7dc2443af498c289a2e739f02ebb515d6aa3fd3ab9db539c86123ae368a4e'
SIZE=19708753
WORKER=Path(__file__).resolve().parents[1]

@lru_cache(maxsize=8)
def _matches(path,size,modified):
    try:return size==SIZE and hashlib.sha256(Path(path).read_bytes()).hexdigest()==SHA256
    except OSError:return False

def valid_model(path):
    try:
        info=path.stat()
        return _matches(str(path),info.st_size,info.st_mtime_ns)
    except OSError:return False

def model_path():
    configured=os.environ.get('INSTRUMENT_MODEL_PATH','').strip()
    if configured:return Path(configured).expanduser().resolve()
    installed=WORKER.parent/'runtime/models'/RELEASE/NAME
    if valid_model(installed):return installed
    return WORKER/'resources/models/instrument'/NAME

def model_available():return valid_model(model_path())
