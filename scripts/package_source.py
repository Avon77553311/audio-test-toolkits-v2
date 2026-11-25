"""Create a reproducible source archive; no models, secrets or business runtime data."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'node_modules','.git','.venv','.local-tools','runtime','dist','build','__pycache__','.pytest_cache','.openai','.sites-runtime'}
ROOT_FILES={'.env.local.example','.gitignore','.gitattributes','AGENTS.md','README.md','package.json','pnpm-lock.yaml','pnpm-workspace.yaml','index.html','tsconfig.json','vite.config.ts','compose.yaml','Start-Local.cmd','Start-LAN.cmd','Start-Docker.cmd','Setup-Tools.cmd','Allow-LAN-Firewall.cmd','start-local.sh'}
FOLDERS={'src','worker','server','scripts','docs','public','deploy','tests'}
selected=[]
for p in ROOT.rglob('*'):
    rel=p.relative_to(ROOT)
    if not p.is_file() or set(rel.parts)&EXCLUDED or rel.parts[:2]==('worker','models'):continue
    if p.suffix in ('.pyc','.log','.zip','.jar','.tmp') or p.name in ('.env','.env.local','deploy.env'):continue
    if len(rel.parts)==1 and p.name not in ROOT_FILES:continue
    if len(rel.parts)>1 and rel.parts[0] not in FOLDERS:continue
    selected.append((p,rel.as_posix()))
out=ROOT/'releases';out.mkdir(exist_ok=True);archive=out/'audio-toolkit-short-source.zip'
manifest={name:hashlib.sha256(p.read_bytes()).hexdigest() for p,name in sorted(selected)}
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p,name in sorted(selected):z.write(p,name)
    z.writestr('SOURCE-MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
sha=hashlib.sha256(archive.read_bytes()).hexdigest();archive.with_suffix('.zip.sha256').write_text(sha+'  '+archive.name+'\n',encoding='ascii')
print(f'Packaged {len(selected)} source files: {archive} ({archive.stat().st_size/1048576:.1f} MiB)\nSHA256 {sha}')
