"""Download pinned official folds, export TorchScript/ONNX and verify numeric parity.

Build tool only: needs CPU PyTorch 2.5.1, onnx 1.17.0 and onnxruntime.
No remote Python code is executed; checkpoints use torch.load(weights_only=True).
Generated models live under ignored runtime/models, never in source/Git.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,sys,tempfile,urllib.parse,urllib.request
import numpy as np
import torch
from chord_model_network import StructuredChordNet

ROOT=Path(__file__).resolve().parents[1]
REVISION='481f4ce703f8822b99f4037e9104ba1760e21ea3'
REPOSITORY='https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition'
CHECKPOINTS=[
 ('921b42d5d1cf9ce1c0c0e45a74d409b8066e0acec46058ef74e24ee0fb540761',5746183),
 ('bcb75859e0efa256696cf5da396b320093317b9b1d9560c304f46c25fe1f8b17',5746175),
 ('acddf85c3fff29954c4877021177d72e2cba9f729ce80c1010f054c477bf3f61',5746179),
 ('65d81a3ab73435aaaade586981b4cabdf57b8953d76052703e6968c32ef8421c',5746175),
 ('5ff6b0ec85640e17a09a9b3de68c93fdd45adc24488e8fa9be5715c28d561122',5746227),
]
OUTPUTS=['triad','bass','seventh','ninth','eleventh','thirteenth']
SIZES=[73,13,4,4,3,3]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def checkpoint(fold,cache):
    name=f'joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s{fold}.best.sdict'
    target=cache/name;sha,size=CHECKPOINTS[fold]
    if not target.exists() or target.stat().st_size!=size or digest(target)!=sha:
        url=f'https://raw.githubusercontent.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition/{REVISION}/cache_data/'+urllib.parse.quote(name)
        temporary=target.with_suffix('.download')
        try:
            with urllib.request.urlopen(url,timeout=120) as response,temporary.open('wb') as output:
                shutil.copyfileobj(response,output)
            if temporary.stat().st_size!=size or digest(temporary)!=sha:raise ValueError('Official checkpoint checksum mismatch: '+name)
            temporary.replace(target)
        finally:temporary.unlink(missing_ok=True)
    return target

def export_fold(fold,source,target):
    import onnx
    import onnxruntime as ort
    model=StructuredChordNet().eval()
    state=torch.load(source,map_location='cpu',weights_only=True)
    model.load_state_dict(state['net'],strict=True)
    pt=target/'chord'/f'structured_chord_fold{fold}.pt'
    onnx_path=target/'chord_onnx'/f'structured_chord_fold{fold}.onnx'
    scripted=torch.jit.script(model);scripted.save(str(pt))
    torch.manual_seed(20261124+fold)
    example=torch.rand(1,64,252)
    torch.onnx.export(model,example,str(onnx_path),input_names=['features'],output_names=OUTPUTS,
        dynamic_axes={'features':{1:'frames'},**{name:{1:'frames'} for name in OUTPUTS}},
        opset_version=17,do_constant_folding=True,dynamo=False)
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(onnx_path),sess_options=opts,providers=['CPUExecutionProvider'])
    peak=0.0
    with torch.inference_mode():
        for frames in (31,64,129):
            features=torch.rand(1,frames,252)
            expected=model(features);loaded=scripted(features);actual=session.run(None,{'features':features.numpy()})
            for size,e,j,o in zip(SIZES,expected,loaded,actual):
                assert list(o.shape)==[1,frames,size]
                np.testing.assert_allclose(j.numpy(),e.numpy(),rtol=1e-5,atol=1e-5)
                np.testing.assert_allclose(o,e.numpy(),rtol=2e-4,atol=5e-5)
                peak=max(peak,float(np.max(np.abs(o-e.numpy()))))
    print(f'Fold {fold}: strict state load, TorchScript + ONNX dynamic frames verified; max error {peak:.6g}',flush=True)
    return {'fold':fold,'source_sha256':digest(source),'source_bytes':source.stat().st_size,'max_onnx_logit_error':peak}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,default=ROOT/'.local-tools/chord-model-recovery');parser.add_argument('--output',type=Path,default=ROOT/'runtime/models/music-x-lab-ismir2019-v1');args=parser.parse_args()
    args.cache.mkdir(parents=True,exist_ok=True);args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError('Do not overwrite a model release; choose a new --output directory')
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='chord-export-',dir=args.output.parent) as staging:
        target=Path(staging);(target/'chord').mkdir();(target/'chord_onnx').mkdir()
        folds=[export_fold(i,checkpoint(i,args.cache),target) for i in range(5)]
        shutil.copy2(ROOT/'worker/resources/licenses/MUSIC_X_LAB_CHORD_LICENSE.txt',target/'LICENSE.txt')
        files={str(p.relative_to(target)).replace('\\','/'):digest(p) for p in sorted(target.rglob('*')) if p.is_file()}
        manifest={'schema':1,'model':'music-x-lab-ismir2019-unweighted-five-fold','upstream':REPOSITORY,'revision':REVISION,
            'original_lfs_weights_recovered':False,'format':'compatible re-export, not byte-identical to original LFS objects',
            'input':{'name':'features','shape':[1,'frames',252],'sample_rate':22050,'hop_length':512,'cqt_bins':288,'bins_per_octave':36,'feature_offset':18,'tuning':None},
            'outputs':dict(zip(OUTPUTS,SIZES)),'torch':torch.__version__,'folds':folds,'files':files}
        (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        target.rename(args.output)
    print('Verified model release:',args.output,flush=True)

if __name__=='__main__':main()
