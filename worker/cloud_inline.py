"""RunPod inline transport for a private LAN: no public callback or automatic retry."""
import base64
import json
import os
from pathlib import Path
import re
import time
import requests

def invoke(source, payload, progress=None):
    from audio_admission import probe_audio_duration
    source = Path(source)
    limit = min(6 * 1024 * 1024, int(os.environ.get('RUNPOD_MAX_AUDIO_BYTES', 6 * 1024 * 1024)))
    if source.stat().st_size > limit:
        raise ValueError('局域网云端传输最多 6 MiB，请先压缩音频；无需开放公网回调。')
    duration = probe_audio_duration(str(source))
    if not 0 < duration <= float(os.environ.get('RUNPOD_MAX_AUDIO_SECONDS', '600')):
        raise ValueError('云端输入超过时长限制')
    endpoint = os.environ.get('RUNPOD_STEM_ENDPOINT_ID', '')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', endpoint):
        raise ValueError('RunPod 端点 ID 无效')
    base = 'https://api.runpod.ai/v2/' + endpoint
    headers = {'Authorization': 'Bearer ' + os.environ['RUNPOD_API_KEY']}
    request = {**payload, 'filename': source.name, 'audio_base64': base64.b64encode(source.read_bytes()).decode('ascii')}
    receipt = Path(os.environ.get('STUDIO_TASK_FOLDER', str(source.parent))) / 'remote-receipt.json'
    receipt.write_text(json.dumps({'endpoint': endpoint, 'state': 'submitting', 'task': payload['task']}), encoding='utf-8')
    try:
        response = requests.post(base + '/run', headers=headers, json={'input': request}, timeout=60)
    except requests.RequestException as error:
        raise RuntimeError('云端提交结果不明，请先核对服务商控制台；不会自动重试。') from error
    if response.status_code >= 400:
        raise RuntimeError('RunPod 拒绝提交，请核对服务商控制台。')
    job_id = response.json().get('id', '')
    if not isinstance(job_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,160}', job_id):
        raise RuntimeError('云端回执无效，请先核对服务商控制台。')
    receipt.write_text(json.dumps({'endpoint': endpoint, 'job_id': job_id, 'task': payload['task'], 'state': 'accepted'}), encoding='utf-8')
    for attempt in range(int(os.environ.get('RUNPOD_MAX_POLL', '120'))):
        if progress:
            progress(min(90, 18 + attempt // 2), '等待 GPU 处理结果')
        response = requests.get(base + '/status/' + job_id, headers=headers, timeout=45)
        if response.status_code >= 400:
            raise RuntimeError('云端状态查询失败，请核对控制台，避免重复提交。')
        job = response.json()
        if job.get('status') == 'COMPLETED':
            return job.get('output') or {}
        if job.get('status') in {'FAILED', 'CANCELLED', 'TIMED_OUT'}:
            raise RuntimeError('云端任务失败，请核对服务商控制台。')
        time.sleep(float(os.environ.get('RUNPOD_POLL_INTERVAL', '5')))
    raise TimeoutError('云端等待超时，请先核对控制台，避免重复提交。')

class LocalStemClient:
    def separate(self, source, output_root, model):
        from capabilities import capabilities
        from cloud_local import extract_stems
        from tools.audio_result_validation import validate_stem_outputs
        if not capabilities()['features']['cloud_stems']:
            raise ValueError('云端分轨未启用')
        output = invoke(source, {'task': 'separate', 'model': model, 'instrument_analysis': False})
        if output.get('status') != 'success':
            raise RuntimeError('云端分轨结果无效')
        paths = extract_stems(output.get('archive_base64'), Path(output_root), model)
        validate_stem_outputs(paths, str(source))
        return Path(output_root)

def analyze_chords(source, options, progress=None):
    from capabilities import capabilities
    if not capabilities()['features']['cloud_chords']:
        raise ValueError('云端神经和弦未启用')
    result = invoke(source, {'task': 'chord_analysis', 'protocol': 'structured-chord-v1',
        'analysis_mode': options.get('analysis_mode', 'harmony'),
        'confidence_threshold': options.get('confidence_threshold', .18),
        'min_chord_duration_ms': options.get('min_chord_duration_ms', 1100),
        'enhance_with_stems': True}, progress)
    if result.get('protocol') != 'structured-chord-v1' or not isinstance(result.get('metadata'), dict):
        raise RuntimeError('云端神经和弦协议响应无效')
    return result['metadata']
