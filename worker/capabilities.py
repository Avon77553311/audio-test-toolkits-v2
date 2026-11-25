import importlib.util
import os
import shutil

def installed(name):
    try:return importlib.util.find_spec(name) is not None
    except (ValueError,ModuleNotFoundError):return False

def enabled(name):return os.environ.get(name,'').lower() in {'true','1','yes','on'}

def capabilities():
    media=bool(shutil.which('ffmpeg') and shutil.which('ffprobe'))
    dsp=media and all(installed(name) for name in ('numpy','scipy','librosa','soundfile','mido'))
    paid=enabled('STUDIO_ENABLE_PAID')
    states={'stems':(media and installed('torch') and installed('demucs'),'需要 CPU PyTorch、Demucs 和 FFmpeg'),
            'midi':(dsp,'需要 Python 音频依赖及 FFmpeg'),'chords':(dsp,'需要 Python 音频依赖及 FFmpeg')}
    from tools.chord_models import model_available
    from tools.instrument_model import model_available as instrument_available
    neural=bool(installed('onnxruntime') and model_available('onnx') or installed('torch') and model_available('pt'))
    return {'success':True,'connected':True,'edition':'short',
            'tools':{name:{'available':available,'reason':'' if available else reason} for name,(available,reason) in states.items()},
            'features':{'neural_chords':neural,'instrument_analysis':installed('torch') and installed('torchaudio') and instrument_available(),
                        'paid_enabled':paid,
                        'cloud_stems':paid and enabled('RUNPOD_STEM_ENABLED') and bool(os.environ.get('RUNPOD_API_KEY')) and bool(os.environ.get('RUNPOD_STEM_ENDPOINT_ID')) and os.environ.get('RUNPOD_STEM_PROTOCOL')=='demucs-v1',
                        'cloud_chords':paid and enabled('RUNPOD_CHORD_ENABLED') and bool(os.environ.get('RUNPOD_API_KEY')) and bool(os.environ.get('RUNPOD_STEM_ENDPOINT_ID')) and os.environ.get('RUNPOD_CHORD_PROTOCOL')=='structured-chord-v1'},
            'limits':{'upload_bytes':100*1024*1024,'task_concurrency':2}}
