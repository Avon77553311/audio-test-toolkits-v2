"""Opt-in actual five-fold inference and authenticated LAN gateway task.
--base-url verifies an already running Docker QA instance; no cloud calls.
"""
from pathlib import Path
import argparse,json,os,shutil,socket,subprocess,sys,time,uuid
import numpy as np
import requests
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'worker'))

def source_audio(folder):
    import bootstrap
    from tools.chord_studio import CHORD_FORMULAS,get_chord_midi_notes,synthesize_progression
    major=next(k for k,v in CHORD_FORMULAS.items() if v['intervals']==[0,4,7]);minor=next(k for k,v in CHORD_FORMULAS.items() if v['intervals']==[0,3,7])
    notes=[get_chord_midi_notes(root,minor if root=='A' else major,4) for root in ['C','F','G','A']]
    audio,rate=synthesize_progression(notes,'block',90)
    source=folder/'C-F-G-Am.wav';sf.write(source,audio,rate)
    return source

def gateway(base,source,folder):
    session=requests.Session();session.trust_env=False
    info=session.get(base+'/api/me',timeout=20);info.raise_for_status();session.headers['X-CSRF-Token']=info.json()['csrf']
    caps=session.get(base+'/api/capabilities',timeout=30).json();assert caps['features']['neural_chords'] and not caps['features']['paid_enabled']
    with source.open('rb') as audio:
        r=session.post(base+'/api/tasks',headers={'X-Submission-ID':str(uuid.uuid4())},data={'tool':'chords','options':json.dumps({'analysis_mode':'harmony','quality_mode':'high','confidence_threshold':.1,'min_chord_duration_ms':700})},files={'file':(source.name,audio,'audio/wav')},timeout=60)
    assert r.status_code==202,r.text;task=r.json()['task']
    deadline=time.monotonic()+180
    while task['status'] in ('queued','running','uploading'):
        assert time.monotonic()<deadline,'Chord inference timeout'
        time.sleep(.4);r=session.get(base+'/api/tasks/'+task['id'],timeout=30);r.raise_for_status();task=r.json()['task']
    assert task['status']=='completed',task
    (folder/'gateway-result.json').write_text(json.dumps(task,ensure_ascii=False,indent=2),encoding='utf-8')
    metadata=task['result']['metadata'];neural=metadata.get('neural',{})
    assert neural.get('ensemble_size')==5 and neural.get('runtime') in ('onnxruntime','pytorch'),metadata
    assert metadata.get('ensemble',{}).get('chosen')!='dsp_fallback',metadata
    recognized=[x['chord'] for x in metadata['chord_progression']]
    assert recognized==['C','F','G','Am'],recognized
    report=next(f for f in task['result']['files'] if f['name'].endswith('.json'))
    assert session.get(base+report['url'],timeout=10).status_code==200
    assert requests.get(base+report['url'],timeout=10).status_code==401
    print('PASS real LAN visitor high-quality chord task, five local models, C-F-G-Am recognized, no DSP fallback, private report download; NO cloud calls',flush=True)
    return session,task

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base-url');args=parser.parse_args()
    folder=ROOT/'.local-tools'/('chord-full-qa-'+uuid.uuid4().hex[:8]);folder.mkdir()
    source=source_audio(folder)
    if args.base_url:gateway(args.base_url,source,folder);return
    import bootstrap
    from tools.neural_chord_analysis import NeuralChordAnalyzer
    from tools.chord_models import model_paths
    # Confirm actual PT fallback and ONNX outputs from the same audio features.
    analyzer=NeuralChordAnalyzer(confidence_threshold=.1,min_chord_duration_ms=700)
    result=analyzer.analyze(str(source));(folder/'neural-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    assert [x['chord'] for x in result['chord_progression']]==['C','F','G','Am']
    assert result['neural']['ensemble_size']==5 and result['neural']['quality_gate_passed']
    # Verify overlap/event windows on a longer real waveform, not only a tensor shape.
    audio,rate=sf.read(source);long_source=folder/'long-progression.wav';sf.write(long_source,np.tile(audio,6),rate)
    long_result=analyzer.analyze(str(long_source))
    assert long_result['neural']['event_context_window_frames']==384
    assert not long_result['neural']['event_context_fallback']
    for name,audio in [('silence',np.zeros(22050*3,dtype=np.float32)),('single-tone',.15*np.sin(2*np.pi*440*np.arange(22050*3)/22050)),('noise',np.random.default_rng(1).normal(0,.05,22050*3))]:
        path=folder/(name+'.wav');sf.write(path,audio,22050)
        value=analyzer.analyze(str(path));assert all(x['chord']=='N' for x in value['chord_progression']),(name,value['chord_progression'])
    shutil.copytree(ROOT/'dist',folder/'dist');shutil.copytree(ROOT/'worker',folder/'worker',ignore=shutil.ignore_patterns('models','__pycache__','*.pyc'))
    model_release=ROOT/'runtime/models/music-x-lab-ismir2019-v1'
    shutil.copytree(model_release,folder/'runtime/models'/model_release.name)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env={**os.environ,'STUDIO_PYTHON':str(ROOT/'.venv/Scripts/python.exe'),'STUDIO_ACCESS_MODE':'lan-public','STUDIO_ENABLE_PAID':'false','STUDIO_SETUP_TOKEN':'test-only','PATH':str(ROOT/'runtime/bin')+os.pathsep+os.environ['PATH']}
    for key in ('STUDIO_PUBLIC_URL','CHORD_MODEL_DIR','CHORD_ONNX_MODEL_DIR'):env.pop(key,None)
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
    print('PASS actual four-chord labels, overlap/event windows, silence/single-tone/noise abstention; QA:',folder,flush=True)

if __name__=='__main__':main()
