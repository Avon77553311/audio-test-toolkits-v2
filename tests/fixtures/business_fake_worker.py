"""Test-only worker: deterministic CPU/GPU results, no audio engines or outbound calls."""
import argparse,json,os
from pathlib import Path
from flask import Flask,request,jsonify

app=Flask(__name__);tasks={};root=Path(__file__).resolve().parents[1]/'runtime'
@app.before_request
def auth():
    if request.headers.get('X-Studio-Token')!=os.environ['STUDIO_WORKER_TOKEN']:return jsonify(message='Unauthorized'),401
@app.get('/api/health')
def health():return jsonify(audioBackendConnected=True)
@app.get('/api/capabilities')
def capabilities():return jsonify(connected=True,features={k:True for k in ['paid_enabled','speech_paid','seed_vc','rvc','cloud_stems','cloud_chords','cloud_structure']},tools={'voice':{'available':True,'reason':''}})
@app.post('/api/tasks')
def create():
    assert request.headers.get('X-Studio-Prepare')=='true'
    id=request.headers['X-Studio-Task-Id'];options=json.loads(request.form.get('options','{}'))
    tasks[id]={'id':id,'name':'simulation','tool':request.form['tool'],'status':'queued','progress':0,'result':None,'options':options}
    return jsonify(success=True,task=tasks[id]),202
@app.get('/internal/tasks/<id>/intent')
def intent(id):
    t=tasks[id];return jsonify(tool=t['tool'],paid=t['options'].get('paid',True),resource=t['options'].get('resource','gpu'))
@app.post('/internal/tasks/<id>/start')
def start(id):
    t=tasks[id];t['status']='running' if t['options'].get('hold') else ('failed' if t['options'].get('fail') else 'completed')
    return jsonify(success=True,task=t)
@app.route('/api/tasks/<id>',methods=['GET','DELETE'])
def task(id):
    t=tasks[id]
    if request.method=='DELETE':t['status']='cancelled'
    elif t['status']=='running' and (root/'release-test').exists():t['status']='completed'
    return jsonify(success=True,task=t)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,required=True);args=parser.parse_args()
    app.run(host='127.0.0.1',port=args.port,threaded=True,use_reloader=False)
