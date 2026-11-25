"""Recover the original LFS-matching EfficientAT model from its official release."""
from pathlib import Path
import hashlib,json,shutil,sys,tempfile,urllib.request
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'worker'))
from tools.instrument_model import NAME,RELEASE,SHA256,SIZE
URL='https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/'+NAME

def main():
    cache=ROOT/'.local-tools/efficientat-recovery';cache.mkdir(exist_ok=True)
    source=cache/NAME
    if not source.is_file() or source.stat().st_size!=SIZE or hashlib.sha256(source.read_bytes()).hexdigest()!=SHA256:
        pending=source.with_suffix('.download')
        try:
            with urllib.request.urlopen(URL,timeout=120) as response,pending.open('wb') as target:shutil.copyfileobj(response,target)
            if pending.stat().st_size!=SIZE or hashlib.sha256(pending.read_bytes()).hexdigest()!=SHA256:raise ValueError('Official model does not match original LFS hash')
            pending.replace(source)
        finally:pending.unlink(missing_ok=True)
    import torch
    from tools.efficient_at import load_efficientat_mn10
    from tools.instrument_analysis import EfficientATPreprocessor
    torch.set_num_threads(1);model=load_efficientat_mn10(source);preprocess=EfficientATPreprocessor()
    with torch.inference_mode():
        logits,embedding=model(preprocess(torch.zeros(1,320000)).unsqueeze(1))
    assert list(logits.shape)==[1,527] and list(embedding.shape)==[1,960]
    assert torch.isfinite(logits).all() and torch.isfinite(embedding).all()
    target=ROOT/'runtime/models'/RELEASE;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError('Model release already exists; do not overwrite a released model')
    with tempfile.TemporaryDirectory(prefix='efficientat-',dir=target.parent) as temp:
        staging=Path(temp);shutil.copy2(source,staging/NAME)
        shutil.copy2(ROOT/'worker/resources/licenses/EFFICIENTAT_LICENSE.txt',staging/'LICENSE.txt')
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in staging.iterdir()}
        (staging/'manifest.json').write_text(json.dumps({'schema':1,'release':RELEASE,'model':'EfficientAT MN10 AudioSet','upstream':'https://github.com/fschmid56/EfficientAT','url':URL,'original_lfs_weights_recovered':True,'sha256':SHA256,'bytes':SIZE,'input':{'sample_rate':32000,'window_sec':10,'hop_sec':5,'n_mels':128},'outputs':{'audioset_classes':527,'embedding':960},'files':files},indent=2)+'\n',encoding='utf-8')
        staging.rename(target)
    print('PASS original LFS hash/size, strict weights-only load, 527 logits and 960-dimensional embedding; installed:',target)

if __name__=='__main__':main()
