"""Opt-in original EfficientAT weights + actual CPU Demucs/Java visitor pipeline.
Synthetic local fixture only; no outbound inference or paid API.
"""
from pathlib import Path
import argparse,io,json,os,shutil,socket,subprocess,sys,time,uuid,zipfile
import numpy as np
import requests
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'worker'))

def source_audio(folder):
    rate=32000;t=np.arange(rate*3)/rate;audio=np.zeros_like(t)
    for freq in (261.6256,329.6276,391.9954):
        audio+=sum(np.sin(2*np.pi*freq*h*t)/h**1.8 for h in range(1,6))*.09*np.exp(-t*1.1)
    source=folder/'synthetic-keyboard.wav';sf.write(source,np.column_stack([audio,audio*.9]),rate)
    return source

def gateway(base,source,folder):
    session=requests.Session();session.trust_env=False
    info=session.get(base+'/api/me',timeout=20).json();session.headers['X-CSRF-Token']=info['csrf']
    caps=session.get(base+'/api/capabilities',timeout=30).json();assert caps['features']['instrument_analysis'] and not caps['features']['paid_enabled']
    with source.open('rb') as stream:
        r=session.post(base+'/api/tasks',headers={'X-Submission-ID':str(uuid.uuid4())},data={'tool':'stems','options':json.dumps({'model':'htdemucs_ft','execution':'cpu','instrument_analysis':True})},files={'file':(source.name,stream,'audio/wav')},timeout=60)
    assert r.status_code==202,r.text;task=r.json()['task'];deadline=time.monotonic()+600
    while task['status'] in ('uploading','queued','running'):
        assert time.monotonic()<deadline,'Actual CPU stems+instrument recognition timeout'
        time.sleep(.5);r=session.get(base+'/api/tasks/'+task['id'],timeout=30);r.raise_for_status();task=r.json()['task']
    assert task['status']=='completed',task
    report=task['result']['metadata']['instrument_analysis']
    assert report['model']=='efficientat_mn10_as' and report['score_semantics']=='uncalibrated_evidence'
    assert set(report['stem_activity'])=={'vocals','drums','bass','other'}
    assert report['instruments'],'Synthetic keyboard should produce instrument evidence'
    for item in report['instruments']:assert 0<=item['score']<=1 and item['confidence'] in ('high','medium','possible')
    file=next(f for f in task['result']['files'] if f['name']=='instrument_analysis.json')
    assert session.get(base+file['url'],timeout=20).json()['model']=='efficientat_mn10_as'
    assert requests.get(base+file['url'],timeout=20).status_code==401
    archive=next(f for f in task['result']['files'] if f['name']=='results.zip')
    with zipfile.ZipFile(io.BytesIO(session.get(base+archive['url'],timeout=30).content)) as z:
        assert 'instrument_analysis.json' in z.namelist() and all(name+'.wav' in z.namelist() for name in ('vocals','drums','bass','other'))
    (folder/'instrument-task.json').write_text(json.dumps(task,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS real LAN visitor CPU Demucs four stems + original EfficientAT classification, finite evidence levels, private JSON and ZIP; NO cloud calls',flush=True)
    return session,task

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base-url');args=parser.parse_args()
    folder=ROOT/'.local-tools'/('instrument-full-qa-'+uuid.uuid4().hex[:8]);folder.mkdir();source=source_audio(folder)
    if args.base_url:gateway(args.base_url,source,folder);return
    import bootstrap,torch
    from tools.instrument_analysis import analyze_instrument_files
    from tools.runtime_resources import instrument_model_path
    torch.set_num_threads(1)
    result=analyze_instrument_files(source,{'piano':source},instrument_model_path(),device='cpu')
    assert any(i['instrument_id'] in ('keyboard','piano','electric_piano') for i in result['instruments']),result
    silence=folder/'silence.wav';sf.write(silence,np.zeros(32000*3),32000)
    assert analyze_instrument_files(silence,{'piano':silence},instrument_model_path(),device='cpu')['instruments']==[]
    shutil.copytree(ROOT/'dist',folder/'dist');shutil.copytree(ROOT/'worker',folder/'worker',ignore=shutil.ignore_patterns('models','__pycache__','*.pyc'))
    release=ROOT/'runtime/models/efficientat-mn10-as-v1';shutil.copytree(release,folder/'runtime/models'/release.name)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env={**os.environ,'STUDIO_PYTHON':str(ROOT/'.venv/Scripts/python.exe'),'STUDIO_ACCESS_MODE':'lan-public','STUDIO_ENABLE_PAID':'false','STUDIO_SETUP_TOKEN':'test-only','PATH':str(ROOT/'runtime/bin')+os.pathsep+os.environ['PATH'],'TORCH_HOME':str(ROOT/'runtime/torch')}
    for key in ('STUDIO_PUBLIC_URL','INSTRUMENT_MODEL_PATH'):env.pop(key,None)
    with (folder/'java.log').open('wb') as log:
        process=subprocess.Popen(['java','-jar',str(ROOT/'server/audio-studio-server.jar'),'--root',str(folder/'dist'),'--port',str(port)],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(100):
                if process.poll() is not None:raise RuntimeError((folder/'java.log').read_text(errors='replace'))
                try:
                    if requests.get(f'http://127.0.0.1:{port}/api/health',timeout=1).status_code==200:break
                except requests.RequestException:pass
                time.sleep(.1)
            gateway(f'http://127.0.0.1:{port}',source,folder)
        finally:
            if os.name=='nt':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            else:process.terminate()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait()
    print('PASS original model recognizes synthetic keyboard evidence and abstains on silence; QA:',folder,flush=True)

if __name__=='__main__':main()
