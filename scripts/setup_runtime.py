"""Install upstream media executables locally; no changes to system PATH."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'runtime'

def download(url, destination):
    with urllib.request.urlopen(url, timeout=60) as response, destination.open('wb') as output:
        shutil.copyfileobj(response, output)

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    binary = RUNTIME / 'bin'
    binary.mkdir(parents=True, exist_ok=True)
    if os.name == 'nt':
        with tempfile.TemporaryDirectory(dir=RUNTIME) as temp:
            temp = Path(temp)
            if not (binary / 'ffprobe.exe').is_file():
                url = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
                archive = temp / 'ffmpeg.zip'
                print('Downloading FFmpeg and ffprobe...', flush=True)
                download(url, archive)
                with urllib.request.urlopen(url + '.sha256', timeout=30) as response:
                    expected = response.read().decode().strip().split()[0].lower()
                actual = hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest()
                if expected != actual:
                    raise RuntimeError('FFmpeg checksum mismatch')
                with zipfile.ZipFile(archive) as zipped:
                    for item in zipped.infolist():
                        name = Path(item.filename).name
                        if name in {'ffmpeg.exe', 'ffprobe.exe', 'LICENSE', 'LICENSE.txt'}:
                            with zipped.open(item) as source, (binary / name).open('wb') as output:
                                shutil.copyfileobj(source, output)
                (binary / 'ffmpeg-source.json').write_text(json.dumps({'url': url, 'sha256': actual}), encoding='utf-8')
    print('Runtime setup complete.', flush=True)

if __name__ == '__main__':
    main()
