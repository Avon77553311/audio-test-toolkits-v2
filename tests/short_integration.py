"""Short edition gateway tests. Real local algorithms, isolated data, no paid calls."""
from pathlib import Path
import io,json,os,shutil,socket,subprocess,sys,time,uuid,zipfile
import numpy as np
import requests
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from chord_models_integration import source_audio

def stop(process):
    if os.name=='nt':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    else:process.terminate()
    try:process.wait(timeout=15)
    except subprocess.TimeoutExpired:process.kill();process.wait()

def main():
    root=ROOT/'.local-tools'/('short-qa-'+uuid.uuid4().hex[:8]);root.mkdir(parents=True)
    shutil.copytree(ROOT/'dist',root/'dist')
    shutil.copytree(ROOT/'worker',root/'worker',ignore=shutil.ignore_patterns('models','__pycache__','*.pyc'))
    shutil.copytree(ROOT/'runtime/models',root/'runtime/models')
    source=source_audio(root)
    rate=22050;t=np.arange(rate*2)/rate;melody=root/'melody.wav'
    sf.write(melody,.18*np.sin(2*np.pi*np.where(t<1,440,523.25)*t)*np.minimum(1,t*20),rate)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env={k:v for k,v in os.environ.items() if not k.startswith(('RUNPOD_','OPENAI_','STUDIO_','CHORD_','INSTRUMENT_'))}
    env.update(STUDIO_PYTHON=str(ROOT/'.venv/Scripts/python.exe'),STUDIO_ACCESS_MODE='lan-public',STUDIO_ENABLE_PAID='false',STUDIO_SETUP_TOKEN='short-test-only-token',TORCH_HOME=str(ROOT/'runtime/torch'),PATH=str(ROOT/'runtime/bin')+os.pathsep+os.environ['PATH'])
    log=(root/'java.log').open('wb');process=subprocess.Popen(['java','-jar',str(ROOT/'server/audio-studio-server.jar'),'--root',str(root/'dist'),'--port',str(port)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    base=f'http://127.0.0.1:{port}'
    def session():s=requests.Session();s.trust_env=False;return s
    a,b,raw=session(),session(),session()
    def req(s,method,path,status=200,**kwargs):
        r=s.request(method,base+path,timeout=40,**kwargs);assert r.status_code==status,(path,r.status_code,r.text[:700]);return r.json()
    def upload(tool,options,file,status=202,key=None,headers=None):
        with file.open('rb') as audio:
            return req(a,'POST','/api/tasks',status,headers=headers or {'X-Submission-ID':key or str(uuid.uuid4())},files={'file':(file.name,audio,'audio/wav')},data={'tool':tool,'options':json.dumps(options)})
    def finish(task):
        deadline=time.monotonic()+600;message=None
        while task['status'] in ('uploading','queued','running'):
            assert time.monotonic()<deadline,task
            if task['message']!=message:print(task['tool'],task['progress'],task['message'],flush=True);message=task['message']
            time.sleep(.6);task=req(a,'GET','/api/tasks/'+task['id'])['task']
        assert task['status']=='completed',task
        for f in task['result']['files']:
            r=a.get(base+f['url'],timeout=20);assert r.status_code==200 and len(r.content)==f['size']
            assert b.get(base+f['url'],timeout=20).status_code==404
            assert raw.get(base+f['url'],timeout=20).status_code==401
            if f['name'].endswith('.zip'):assert zipfile.ZipFile(io.BytesIO(r.content)).testzip() is None
        req(b,'GET','/api/tasks/'+task['id'],404)
        (root/(task['tool']+'-result.json')).write_text(json.dumps(task,ensure_ascii=False,indent=2),encoding='utf-8')
        print('PASS',task['tool'],'completed, own files downloadable, other visitors denied',flush=True)
        return task
    try:
        for _ in range(200):
            if process.poll() is not None:raise RuntimeError((root/'java.log').read_text(errors='replace'))
            try:
                if raw.get(base+'/api/health',timeout=1).status_code==200:break
            except requests.RequestException:pass
            time.sleep(.1)
        else:raise RuntimeError('Java startup timeout')
        req(raw,'GET','/api/tasks',401)
        for s in (a,b):
            data=req(s,'GET','/api/me');assert data['accessMode']=='lan-public' and data['visitor']['role']=='guest';s.headers['X-CSRF-Token']=data['csrf']
        caps=req(a,'GET','/api/capabilities');assert set(caps['tools'])=={'stems','midi','chords'} and caps['edition']=='short' and all(t['available'] for t in caps['tools'].values()),caps
        assert caps['features']['neural_chords'] and caps['features']['instrument_analysis'] and not caps['features']['paid_enabled'],caps
        req(a,'GET','/api/tasks',403,headers={'Host':'attacker.example'})
        req(a,'GET','/api/tasks',403,headers={'Origin':'http://attacker.example'})
        req(a,'GET','/api/admin/tasks',401)
        upload('midi',{},melody,403,headers={'X-CSRF-Token':'wrong','X-Submission-ID':str(uuid.uuid4())})
        for tool in ('trim','speech','convert','voice','pitch','song','piano','studio','practice'):
            upload(tool,{},melody,400)
        for path in ('/api/studio/data','/api/studio/render','/api/piano/render','/api/tasks/'+uuid.uuid4().hex+'/edit'):
            req(a,'GET' if path.endswith('/data') else 'POST',path,404,json={} if not path.endswith('/data') else None)
        print('PASS exact three-tool capability, other nine tools rejected, retired routes 404, Host/Origin/CSRF and admin protection',flush=True)
        key=str(uuid.uuid4());midi=upload('midi',{'transcription_mode':'mono','transcription_profile':'balanced','quality_mode':'balanced'},melody,key=key)['task']
        assert upload('midi',{},melody,200,key=key)['task']['id']==midi['id']
        upload('midi',{},melody,409)
        midi=finish(midi);assert midi['result']['metadata']['notes'];assert any(f['name'].endswith('.mid') for f in midi['result']['files']);assert any(f['name']=='midi_preview.wav' for f in midi['result']['files'])
        chords=finish(upload('chords',{'analysis_mode':'harmony','quality_mode':'high','confidence_threshold':.1,'min_chord_duration_ms':700},source)['task'])
        meta=chords['result']['metadata'];assert meta['neural']['ensemble_size']==5 and meta['neural']['runtime'] in ('onnxruntime','pytorch'),meta
        recognized=[x['chord'] for x in meta['chord_progression']];assert recognized==['C','F','G','Am'],recognized
        print('PASS high-quality chord analysis: all five local models, C-F-G-Am, no cloud',flush=True)
        stems=finish(upload('stems',{'model':'htdemucs_6s','execution':'cpu','instrument_analysis':True},source)['task'])
        names={f['name'] for f in stems['result']['files']};assert all(x+'.wav' in names for x in ('vocals','drums','bass','guitar','piano','other')),names
        assert stems['result']['metadata']['instrument_analysis']['model'] and stems['result']['metadata']['instrument_analysis']['score_semantics']=='uncalibrated_evidence'
        assert req(b,'GET','/api/tasks')['tasks']==[]
        print('PASS six-track local CPU Demucs and EfficientAT; MIDI preview; private sessions; NO paid calls',flush=True)
        (root/'summary.json').write_text(json.dumps({'passed':True,'tools':['stems','midi','chords'],'source':str(source)},indent=2),encoding='utf-8')
        print('QA evidence:',root,flush=True)
    finally:stop(process);log.close()
if __name__=='__main__':main()
