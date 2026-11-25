"""One cancellable process per job; calls migrated algorithms, never web-auth bypasses."""
from __future__ import annotations
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile
from bootstrap import configure
configure()

PREFIX = '__STUDIO__'

def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items() if not str(k).endswith(('_path', '_token')) and str(k) not in {'payload', 'api_key', 'runpod_job_id'}}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, 'tolist'):
        return safe(value.tolist())
    if hasattr(value, 'item'):
        return safe(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

def progress(value, message):
    print(PREFIX + json.dumps({'progress': value, 'message': message}, ensure_ascii=False), flush=True)

def execute(folder):
    os.environ['STUDIO_TASK_FOLDER'] = str(folder.resolve())
    job = json.loads((folder / 'input.json').read_text(encoding='utf-8'))
    tool, options = job['tool'], dict(job['options'])
    if tool not in {'stems','midi','chords'}:
        raise ValueError('Tool is not available in the short edition')
    source, output = str(folder / job['source']), folder / 'results'
    output.mkdir(exist_ok=True)
    options['original_filename'] = job['name']
    progress(5, '正在加载处理引擎')
    if tool == 'stems' and options.get('execution') != 'gpu':
        from tools.stem_separation import StemTaskManager
        manager = StemTaskManager(output)
        legacy_id = manager.submit(source, model=options.get('model', 'htdemucs_6s'), instrument_analysis=False,
                                   combined=options.get('combined', False), ranges=options.get('ranges', []), operation=options.get('operation', 'mute'), fade_ms=options.get('fade_ms',12))
        last_event=None
        while True:
            task = manager.get(legacy_id)
            event=(task['progress'],task['message'])
            if event != last_event:
                progress(min(84,round(event[0]*.84)) if options.get('instrument_analysis') else event[0],event[1])
                last_event=event
            if task['status'] in {'completed', 'failed', 'cancelled'}:
                break
            time.sleep(.5)
        manager.executor.shutdown(wait=True)
        if task['status'] != 'completed':
            raise RuntimeError(task['message'])
        paths = [str(output / path) for path in task['outputs']]
        result = {'success': True, 'message': '音轨分离完成', 'output_files': paths, 'metadata': {'model': task['model'], 'stems': task['stems']}}
        if options.get('instrument_analysis'):
            from tools.instrument_analysis import analyze_instrument_files
            from tools.runtime_resources import instrument_model_path
            report = analyze_instrument_files(source, {Path(path).stem:path for path in paths if Path(path).stem in {'vocals','drums','bass','other','piano','guitar'}}, instrument_model_path(), device='cpu', progress=lambda value,message:progress(85+round(value*.14),message))
            report_path=output/'instrument_analysis.json'
            report_path.write_text(json.dumps(safe(report),ensure_ascii=False,allow_nan=False),encoding='utf-8')
            result['output_files'].append(str(report_path))
            result['metadata']['instrument_analysis']=report

    elif tool == 'stems':
        from cloud_local import separate
        progress(12, '正在提交显式选择的云端分轨任务')
        result = separate(source, output, options.get('model', 'htdemucs_6s'), progress)
    else:
        modules = {'midi': ('audio_to_midi','AudioToMidiTool'), 'chords': ('chord_analysis','ChordAnalysisTool')}
        module, name = modules[tool]
        engine = getattr(importlib.import_module('tools.' + module), name)()
        if job.get('reference'):
            options['reference_path'] = str(folder / job['reference'])
        if tool == 'chords':
            options['_progress_callback'] = progress
        # This runner is already the disposable, cancellable process boundary.
        # Avoid the upstream MIDI wrapper's locale-dependent nested subprocess on Windows.
        result = engine._process_direct(source, str(output), **options) if tool == 'midi' else engine.process(source, str(output), **options)
    if not isinstance(result, dict) or not result.get('success'):
        raise RuntimeError('工具没有返回有效结果')
    paths = [Path(path).resolve() for path in result.get('output_files', [])]
    for path in paths:
        if not path.is_relative_to(output.resolve()) or not path.is_file():
            raise ValueError('结果文件校验失败')
    if tool == 'midi':
        from midi_preview import render_preview
        midi = next((path for path in paths if path.suffix == '.mid'), None)
        if midi:
            preview = output / 'midi_preview.wav'
            note_events = render_preview(midi,preview)
            if note_events:
                paths.append(preview)
                result.setdefault('metadata',{})['notes'] = note_events
    if tool in {'chords', 'midi'}:
        report = output / 'analysis.json'
        report.write_text(json.dumps(safe(result.get('metadata', {})), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        paths.append(report)
    if len(paths) > 1:
        bundle = output / 'results.zip'
        with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as zipped:
            for index, path in enumerate(paths):
                zipped.write(path, path.name if sum(p.name == path.name for p in paths) == 1 else f'{index}_{path.name}')
        paths.append(bundle)
    files = [{'name': path.name, 'relative': str(path.relative_to(folder)), 'size': path.stat().st_size} for path in paths]
    public = safe({key: value for key, value in result.items() if key != 'output_files'})
    public['files'] = files
    (folder / 'result.json').write_text(json.dumps(public, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    progress(100, '处理完成')

if __name__ == '__main__':
    folder = Path(sys.argv[1]).resolve()
    try:
        execute(folder)
    except Exception as error:
        import traceback
        traceback.print_exc()
        message = str(error)[:1000] or type(error).__name__
        # Detailed messages stay in the private job log; the API returns a short safe error.
        (folder / 'error.json').write_text(json.dumps({'error_type': type(error).__name__, 'message': message}, ensure_ascii=False), encoding='utf-8')
        sys.exit(1)
