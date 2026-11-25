from pathlib import Path
import hashlib,sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'worker'))
from tools import instrument_model

@pytest.fixture
def isolated(tmp_path,monkeypatch):
    worker=tmp_path/'worker';worker.mkdir();monkeypatch.setattr(instrument_model,'WORKER',worker)
    monkeypatch.delenv('INSTRUMENT_MODEL_PATH',raising=False);instrument_model._matches.cache_clear()
    return worker

def test_lfs_pointer_is_not_available(isolated):
    path=isolated/'resources/models/instrument'/instrument_model.NAME;path.parent.mkdir(parents=True)
    path.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:'+instrument_model.SHA256+'\nsize '+str(instrument_model.SIZE)+'\n')
    assert not instrument_model.model_available()

def test_runtime_discovery_and_corruption(isolated,monkeypatch):
    content=b'valid-test-model';monkeypatch.setattr(instrument_model,'SIZE',len(content));monkeypatch.setattr(instrument_model,'SHA256',hashlib.sha256(content).hexdigest())
    path=isolated.parent/'runtime/models'/instrument_model.RELEASE/instrument_model.NAME;path.parent.mkdir(parents=True);path.write_bytes(content)
    assert instrument_model.model_path()==path and instrument_model.model_available()
    path.write_bytes(b'bad-model-value!')
    assert not instrument_model.model_available()

def test_explicit_path_not_ignored(isolated,monkeypatch,tmp_path):
    monkeypatch.setenv('INSTRUMENT_MODEL_PATH',str(tmp_path/'custom.pt'))
    assert instrument_model.model_path()==tmp_path/'custom.pt'
    assert not instrument_model.model_available()

def test_loader_rejects_missing_weight(isolated,monkeypatch):
    from tools.runtime_resources import instrument_model_path
    with pytest.raises(RuntimeError,match='SHA256'):instrument_model_path()
