"""Opt-in local delivery for the original RunPod Demucs handler (no public callback)."""
import base64
import io
import os
from pathlib import Path
import re
import time
import zipfile
import requests

def extract_stems(encoded, output, model):
    if not isinstance(encoded, str) or len(encoded) > 200 * 1024 * 1024:
        raise ValueError('云端结果超过本地接收上限')
    stems = {'vocals','drums','bass','other'} | ({'piano','guitar'} if model == 'htdemucs_6s' else set())
    found = {}
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded, validate=True))) as archive:
        if sum(item.file_size for item in archive.infolist()) > 512 * 1024 * 1024:
            raise ValueError('分轨解压后超过上限')
        for item in archive.infolist():
            # Flatten only expected audio names; never extract remote paths or links.
            name = Path(item.filename.replace('\\','/')).name
            if name in {stem+'.wav' for stem in stems}:
                if name in found:
                    raise ValueError('云端返回了重复轨道')
                path = output / name
                with archive.open(item) as source, path.open('wb') as target:
                    import shutil
                    shutil.copyfileobj(source,target)
                found[name] = str(path)
    if set(found) != {stem+'.wav' for stem in stems}:
        raise ValueError('云端结果缺少所需音轨')
    return list(found.values())

def separate(source, output, model, progress):
    from capabilities import capabilities
    if not capabilities()['features']['cloud_stems']:
        raise ValueError('云端分轨尚未显式启用并配置')
    from cloud_inline import invoke
    from tools.audio_result_validation import validate_stem_outputs
    source=Path(source);output=Path(output)
    result=invoke(source,{'task':'separate','model':model,'instrument_analysis':False},progress)
    if not isinstance(result,dict) or result.get('status')!='success':
        raise RuntimeError('云端未返回有效分轨结果')
    files=extract_stems(result.get('archive_base64'),output,model)
    validate_stem_outputs(files,str(source))
    return {'success':True,'message':'云端分轨完成','output_files':files,'metadata':{'model':model,'execution':'gpu'}}
