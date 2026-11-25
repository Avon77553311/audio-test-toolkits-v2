from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'worker'))
from app import validated_options,create_app

@pytest.mark.parametrize('tool',['trim','speech','convert','voice','pitch','song','piano','studio','practice','edit','anything'])
def test_other_tools_rejected(tool):
    with pytest.raises(ValueError):validated_options(tool,{})

@pytest.mark.parametrize('options',[{'combined':True},{'ranges':[{'start':0,'end':1}]},{'_progress_callback':'inject'},{'execution':'remote'},{'instrument_analysis':'true'}])
def test_stems_rejects_retired_or_invalid_options(options):
    with pytest.raises(ValueError):validated_options('stems',options)

def test_private_worker_boundary_and_retired_routes(tmp_path):
    app=create_app(tmp_path/'jobs',token='short-private-test')
    try:
        client=app.test_client()
        assert client.get('/api/capabilities').status_code==401
        headers={'X-Studio-Token':'short-private-test'}
        caps=client.get('/api/capabilities',headers=headers).json
        assert set(caps['tools'])=={'stems','midi','chords'} and caps['edition']=='short'
        for path in ('/api/studio/data','/api/studio/render','/api/piano/render','/api/tasks/'+'a'*32+'/edit'):
            assert client.post(path,headers=headers,json={}).status_code==404
    finally:app.extensions['task_store'].close()
