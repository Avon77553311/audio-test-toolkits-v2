from pathlib import Path
import hashlib,json,sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'worker'))
from tools import chord_models

def folds(folder,extension):
    folder.mkdir(parents=True)
    for i in range(5):(folder/f'structured_chord_fold{i}.{extension}').write_bytes((b'PK' if extension=='pt' else b'\x08\x08')+bytes([i])*2048)

@pytest.fixture
def isolated(tmp_path,monkeypatch):
    worker=tmp_path/'worker';worker.mkdir();monkeypatch.setattr(chord_models,'WORKER',worker)
    monkeypatch.delenv('CHORD_MODEL_DIR',raising=False);monkeypatch.delenv('CHORD_ONNX_MODEL_DIR',raising=False)
    chord_models._verify.cache_clear()
    return worker

def test_lfs_pointers_never_report_available(isolated):
    path=isolated/'resources/models/chord';folds(path,'pt')
    (path/'structured_chord_fold3.pt').write_text('version https://git-lfs.github.com/spec/v1\n'+'x'*2048)
    assert not chord_models.model_available('pt')

def test_managed_models_and_hash_changes(isolated):
    root=isolated.parent/'runtime/models'/chord_models.RELEASE;folder=root/'chord_onnx';folds(folder,'onnx')
    manifest={'files':{str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}}
    (root/'manifest.json').write_text(json.dumps(manifest))
    assert chord_models.model_available('onnx')
    assert chord_models.model_paths('onnx')[0].parent==folder
    path=folder/'structured_chord_fold4.onnx';path.write_bytes(path.read_bytes()+b'corrupt')
    assert not chord_models.model_available('onnx')

def test_explicit_configuration_not_silently_ignored(isolated,tmp_path,monkeypatch):
    folder=isolated/'resources/models/chord';folds(folder,'pt')
    assert chord_models.model_available('pt')
    monkeypatch.setenv('CHORD_MODEL_DIR',str(tmp_path/'missing'))
    assert not chord_models.model_available('pt')

def test_backend_uses_pt_on_cpu_when_onnx_files_are_missing(isolated,monkeypatch):
    import torch
    from tools.neural_chord_analysis import NeuralChordAnalyzer
    folds(isolated/'resources/models/chord','pt')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
    assert NeuralChordAnalyzer._preferred_backend()[2]=='cpu'

def test_missing_folds_fail_before_inference(isolated):
    from tools.runtime_resources import chord_onnx_model_files
    with pytest.raises(RuntimeError,match='ONNX'):chord_onnx_model_files()
