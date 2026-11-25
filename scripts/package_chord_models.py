"""Package verified model payload separately from source and the runtime image."""
from pathlib import Path
import argparse,hashlib,json,re,zipfile
ROOT=Path(__file__).resolve().parents[1]
RELEASE='music-x-lab-ismir2019-v1'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',default=RELEASE);args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]+',args.version):parser.error('Use ASCII letters, numbers, hyphens or underscores')
    source=ROOT/'runtime/models'/RELEASE
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
    output=ROOT/'releases'/('audio-toolkit-chord-models-'+args.version+'.zip')
    if output.exists():raise FileExistsError('Do not overwrite an existing model release archive')
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name in [*manifest['files'],'manifest.json']:archive.write(source/name,'models/'+name)
        for name in ['Install-ChordModels.ps1','Install-ChordModels.cmd','CHORD-MODELS-README.md']:
            value=(ROOT/'deploy/windows'/name).read_text(encoding='utf-8-sig').replace('\r\n','\n')
            if name.endswith('.cmd'):value=value.replace('\n','\r\n')
            archive.writestr(name,value.encode('utf-8-sig' if name.endswith('.ps1') else 'utf-8'))
    with zipfile.ZipFile(output) as archive:assert archive.testzip() is None
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    Path(str(output)+'.sha256').write_text(digest+'  '+output.name+'\n',encoding='ascii')
    print(json.dumps({'archive':str(output),'bytes':output.stat().st_size,'sha256':digest},ensure_ascii=False))

if __name__=='__main__':main()
