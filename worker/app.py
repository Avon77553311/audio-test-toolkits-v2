"""Loopback algorithm service owned by the Java process, authenticated with an ephemeral token."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hmac
import json
import logging
import math
import mimetypes
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import uuid
from flask import Flask, jsonify, request, send_file
from werkzeug.exceptions import HTTPException
from bootstrap import RUNTIME, WORKER, configure
from capabilities import capabilities, enabled
from run_task import safe

def now():
    return datetime.now(timezone.utc).isoformat()

class TaskStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / 'tasks.sqlite3'
        self.pool = ThreadPoolExecutor(max_workers=max(1,min(32,int(os.environ.get('STUDIO_WORKER_CONCURRENCY','2')))), thread_name_prefix='audio-task')
        self.processes = {}
        self.lock = threading.RLock()
        with self.connection() as database:
            database.execute('CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, tool TEXT NOT NULL, name TEXT NOT NULL, status TEXT NOT NULL, progress INTEGER NOT NULL, message TEXT NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL, result TEXT, paid INTEGER NOT NULL DEFAULT 0)')
            database.execute("UPDATE tasks SET status='failed', message='服务已重启，请重新提交未完成任务', updated=? WHERE status IN ('queued','running')", (now(),))

    def connection(self):
        database = sqlite3.connect(self.database, timeout=10)
        database.row_factory = sqlite3.Row
        return database

    def update(self, task_id, **values):
        values['updated'] = now()
        with self.connection() as database:
            database.execute('UPDATE tasks SET ' + ', '.join(key + '=?' for key in values) + ' WHERE id=?', (*values.values(), task_id))

    def create(self, task_id, tool, name, paid=False):
        with self.connection() as database:
            database.execute('INSERT INTO tasks(id,tool,name,status,progress,message,created,updated,paid) VALUES(?,?,?,?,?,?,?,?,?)',
                             (task_id, tool, name, 'queued', 0, '等待处理', now(), now(), int(paid)))

    def get(self, task_id):
        if not re.fullmatch(r'[0-9a-f]{32}', task_id):
            return None
        with self.connection() as database:
            row = database.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
        if not row:
            return None
        value = dict(row)
        value['result'] = json.loads(value['result']) if value['result'] else None
        value['cancelable'] = value['status'] in {'queued','running'} and not (value['paid'] and value['status'] == 'running')
        if value['result']:
            for file in value['result'].get('files', []):
                file['url'] = f'/api/files/{task_id}/{file["name"]}'
                file.pop('relative', None)
        return value

    def list(self):
        with self.connection() as database:
            rows = database.execute('SELECT id FROM tasks ORDER BY created DESC LIMIT 100').fetchall()
        return [self.get(row['id']) for row in rows]

    def execute(self, task_id):
        folder = self.root / task_id
        with self.lock:
            if self.get(task_id)['status'] == 'cancelled':
                return
            process = subprocess.Popen([sys.executable, '-u', str(WORKER / 'run_task.py'), str(folder)], cwd=WORKER,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                env={**os.environ, 'PYTHONIOENCODING':'utf-8'}, start_new_session=os.name != 'nt',
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
            self.processes[task_id] = process
            self.update(task_id, status='running', progress=2, message='正在启动音频引擎')
        def read_output():
            with (folder / 'task.log').open('w', encoding='utf-8') as log:
                for line in process.stdout:
                    log.write(line)
                    prefix=next((prefix for prefix in ('__STUDIO__','AUDIO_TOOLKIT_PROGRESS=') if line.startswith(prefix)),None)
                    if prefix:
                        try:
                            event = json.loads(line[len(prefix):])
                            if self.get(task_id)['status'] == 'running':
                                self.update(task_id, progress=max(2, min(99, int(event['progress']))), message=str(event['message'])[:250])
                        except (ValueError, KeyError, TypeError):
                            pass
        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        try:
            process.wait(timeout=1800)
            reader.join(timeout=3)
            with self.lock:
                if self.get(task_id)['status'] == 'cancelled':
                    return
                if process.returncode or not (folder / 'result.json').is_file():
                    detail = json.loads((folder / 'error.json').read_text(encoding='utf-8')) if (folder / 'error.json').is_file() else {}
                    kind = detail.get('error_type', '')
                    message = '处理失败，请检查音频内容和运行环境；可查看 runtime/jobs 中对应任务日志。'
                    if kind in {'ModuleNotFoundError', 'ImportError'}:
                        message = '算法依赖缺失，请运行 Setup-Tools.cmd。'
                    elif kind == 'FileNotFoundError':
                        message = '模型或媒体组件缺失，请运行环境安装脚本或检查模型路径。'
                    elif kind in {'ValueError', 'AudioAdmissionError'}:
                        text = detail.get('message', '')
                        if re.search('[\u4e00-\u9fff]', text) and not re.search(r'https?://|[A-Za-z]:[\\/]|token|api.key', text, re.I):
                            message = text[:400]
                    self.update(task_id, status='failed', message=message)
                    return
                result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
                self.update(task_id, status='completed', progress=100, message=result.get('message', '处理完成'), result=json.dumps(result, ensure_ascii=False, allow_nan=False))
        except subprocess.TimeoutExpired:
            self.terminate(process)
            self.update(task_id, status='failed', message='处理超过 30 分钟，已停止本地任务。云端任务请核对服务商状态。')
        except Exception:
            logging.exception('Task execution failed: %s', task_id)
            self.update(task_id, status='failed', message='任务服务异常，请查看本地运行日志。')
        finally:
            with self.lock:
                self.processes.pop(task_id, None)

    @staticmethod
    def terminate(process):
        if process.poll() is not None:
            return
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True, timeout=15)
        else:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    def cancel(self, task_id):
        with self.lock:
            task = self.get(task_id)
            if not task:
                return None
            if not task['cancelable']:
                raise ValueError('此任务已结束或已经提交付费服务，不能在本地取消。')
            self.update(task_id, status='cancelled', message='任务已取消')
            if task_id in self.processes:
                self.terminate(self.processes[task_id])
        return self.get(task_id)

    def close(self):
        with self.lock:
            for process in list(self.processes.values()):
                self.terminate(process)
        self.pool.shutdown(wait=False, cancel_futures=True)

def validated_options(tool, values):
    if tool not in {'stems','midi','chords'}:
        raise ValueError('简化版仅支持音轨分离、音频转 MIDI 和和弦分析')
    if not isinstance(values, dict):
        raise ValueError('参数必须为对象')
    fields = {
        'stems': {'model','execution','instrument_analysis','paid_consent'},
        'midi': {'transcription_mode','transcription_profile','quality_mode','time_window_ms','activation_level','fmin','fmax','min_note_duration_ms','bpm','onset_semitone','use_runpod','paid_consent'},
        'chords': {'analysis_mode','quality_mode','time_window_ms','confidence_threshold','min_chord_duration_ms','min_frequency','max_frequency','expected_chord_count','experimental','use_runpod','paid_consent'}}
    if set(values) - fields[tool]:
        raise ValueError('包含未知参数')
    for key in {'use_runpod','paid_consent','experimental','instrument_analysis'} & values.keys():
        if not isinstance(values[key], bool):
            raise ValueError(key + ' 必须是布尔值')
    if tool == 'midi':
        from tools.audio_to_midi_options import validate_audio_to_midi_options
        values = {**values, **validate_audio_to_midi_options(values)}
        if values.get('use_runpod') and values.get('quality_mode') != 'high':
            raise ValueError('GPU 辅助转录需要质量优先模式')
    elif tool == 'chords':
        from tools.chord_analysis_options import validate_chord_analysis_options
        values = {**values, **validate_chord_analysis_options(values)}
        if values.get('use_runpod') and (values.get('quality_mode') != 'high' or values.get('analysis_mode') == 'bass'):
            raise ValueError('GPU 神经和弦需要质量优先的和声或混合模式')
        if values.get('quality_mode') == 'high' and not values.get('use_runpod') and not capabilities()['features']['neural_chords']:
            raise ValueError('高质量神经和弦模型尚未安装，请先使用均衡模式。')
    elif tool == 'stems':
        if values.get('model','htdemucs_6s') not in {'htdemucs_ft','htdemucs_6s'} or values.get('execution','cpu') not in {'cpu','gpu'}:
            raise ValueError('分轨模型或执行方式无效')
        if values.get('instrument_analysis') and (values.get('execution') == 'gpu' or not capabilities()['features']['instrument_analysis']):
            raise ValueError('乐器识别需要安装完整 EfficientAT 权重，当前仅在本地 CPU 开放')
    return values


def check_paid(tool, options):
    paid = options.get('use_runpod') or options.get('execution') == 'gpu'
    if paid and (not enabled('STUDIO_ENABLE_PAID') or options.get('paid_consent') is not True):
        raise PermissionError('付费能力默认关闭；管理员启用后仍需明确勾选本次调用确认。')
    features = capabilities()['features']
    if (options.get('execution') == 'gpu' or tool == 'midi' and options.get('use_runpod')) and not features['cloud_stems']:
        raise ValueError('云端分轨未配置')
    if tool == 'chords' and options.get('use_runpod') and not features['cloud_chords']:
        raise ValueError('云端神经和弦尚未配置')
    return bool(paid)


def create_app(data_root=None, token=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 102 * 1024 * 1024
    app.json.ensure_ascii = False
    app.json.sort_keys = False
    secret = token or os.environ.get('STUDIO_WORKER_TOKEN', '')
    if not secret:
        raise RuntimeError('Java worker token is required')
    store = TaskStore(data_root or RUNTIME / 'jobs')
    app.extensions['task_store'] = store

    @app.before_request
    def authenticate():
        if not hmac.compare_digest(request.headers.get('X-Studio-Token',''), secret):
            return jsonify(success=False, message='Unauthorized'), 401

    @app.errorhandler(Exception)
    def error_response(error):
        if isinstance(error, HTTPException):
            return jsonify(success=False, message='文件过大' if error.code == 413 else error.name), error.code
        if isinstance(error, PermissionError):
            return jsonify(success=False, message=str(error)), 403
        if isinstance(error, (ValueError, KeyError, TypeError)):
            return jsonify(success=False, message=str(error) if isinstance(error,ValueError) else '参数格式无效'), 400
        app.logger.exception('Local service error')
        return jsonify(success=False, message='本地服务错误，请查看 runtime/worker.log'), 500

    @app.get('/api/health')
    def health():
        return jsonify(status='ok', application='audio-toolkit', mode='local-tools', audioBackendConnected=True)

    @app.post('/internal/shutdown')
    def shutdown_jobs():
        store.close()
        return jsonify(success=True)

    def assigned_task_id():
        value=request.headers.get('X-Studio-Task-Id') or uuid.uuid4().hex
        if not re.fullmatch(r'[0-9a-f]{32}',value):
            raise ValueError('任务编号无效')
        return value

    def schedule(task_id):
        if request.headers.get('X-Studio-Prepare') != 'true':
            store.pool.submit(store.execute,task_id)

    @app.post('/internal/tasks/<task_id>/start')
    def start_prepared(task_id):
        with store.lock:
            task=store.get(task_id)
            if not task:
                return jsonify(success=False,message='任务不存在'),404
            if task['status']=='queued':
                # Mark claimed before submitting so a repeated start cannot run twice.
                store.update(task_id,status='running',message='业务调度已领取任务')
                store.pool.submit(store.execute,task_id)
            return jsonify(success=True,task=store.get(task_id))

    @app.get('/internal/tasks/<task_id>/intent')
    def task_intent(task_id):
        task=store.get(task_id)
        if not task:
            return jsonify(success=False,message='任务不存在'),404
        job=json.loads((store.root/task_id/'input.json').read_text(encoding='utf-8'))
        options=job['options']
        gpu=options.get('execution')=='gpu' or bool(options.get('use_runpod'))
        return jsonify(success=True,tool=task['tool'],resource='gpu' if gpu else 'cpu',paid=bool(task['paid']),cloudSpeech=False)

    @app.get('/api/capabilities')
    def get_capabilities():
        return jsonify(capabilities())

    @app.get('/api/tasks')
    def list_tasks():
        return jsonify(success=True, tasks=store.list())

    @app.post('/api/tasks')
    def submit_task():
        tool = request.form.get('tool','')
        options = validated_options(tool, json.loads(request.form.get('options','{}')))
        paid = check_paid(tool, options)
        capability = capabilities()['tools'].get(tool, {'available':False,'reason':'工具不可用'})
        if not capability['available'] and not (tool == 'stems' and options.get('execution') == 'gpu'):
            return jsonify(success=False, message=capability['reason']), 503
        uploaded = request.files.get('file')
        if not uploaded or not uploaded.filename:
            raise ValueError('请先导入音频')
        extension = Path(uploaded.filename.replace('\\','/')).suffix.lower()
        if extension not in {'.wav','.wave','.mp3','.flac','.m4a','.aac','.ogg','.oga','.aif','.aiff','.mp4'}:
            raise ValueError('不支持的音频格式')
        task_id = assigned_task_id()
        folder = store.root / task_id
        folder.mkdir()
        try:
            source = 'input' + extension
            uploaded.save(folder / source)
            from audio_admission import task_mode, validate_audio_file
            tool_names = {'midi':'audio-to-midi','chords':'chord-analysis','stems':'stems'}
            mode_options = {**options, 'resource_type': 'gpu' if options.get('execution') == 'gpu' else 'cpu'}
            duration = validate_audio_file(str(folder / source), task_mode(tool_names[tool], mode_options))
            reference = None
            name = Path(uploaded.filename.replace('\\','/')).name[:120]
            (folder / 'input.json').write_text(json.dumps({'tool':tool,'options':options,'name':name,'source':source,'reference':reference}, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            store.create(task_id, tool, name, paid)
            schedule(task_id)
            return jsonify(success=True, task=store.get(task_id)), 202
        except Exception:
            import shutil
            shutil.rmtree(folder)
            raise

    @app.get('/api/tasks/<task_id>')
    def get_task(task_id):
        task = store.get(task_id)
        return (jsonify(success=True, task=task), 200) if task else (jsonify(success=False,message='任务不存在'),404)

    @app.delete('/api/tasks/<task_id>')
    def cancel_task(task_id):
        try:
            task = store.cancel(task_id)
            return (jsonify(success=True,task=task),200) if task else (jsonify(success=False,message='任务不存在'),404)
        except ValueError as error:
            return jsonify(success=False,message=str(error)),409

    @app.get('/api/files/<task_id>/<name>')
    def get_file(task_id, name):
        task = store.get(task_id)
        if not task or task['status'] != 'completed':
            return jsonify(success=False,message='结果不可用'),404
        with store.connection() as database:
            result = json.loads(database.execute('SELECT result FROM tasks WHERE id=?',(task_id,)).fetchone()['result'])
        files = [item for item in result.get('files',[]) if item['name'] == name]
        if not files:
            return jsonify(success=False,message='文件不存在'),404
        path = (store.root / task_id / files[0]['relative']).resolve()
        if not path.is_relative_to((store.root / task_id / 'results').resolve()) or not path.is_file():
            return jsonify(success=False,message='文件不存在'),404
        return send_file(path, as_attachment=request.args.get('download') == '1', download_name=name, conditional=True,
                         mimetype='audio/midi' if path.suffix == '.mid' else mimetypes.guess_type(name)[0])

    return app

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    application = create_app()
    try:
        application.run(host='127.0.0.1',port=args.port,threaded=True,use_reloader=False)
    finally:
        application.extensions['task_store'].close()
