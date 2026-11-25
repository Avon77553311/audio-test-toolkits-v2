"""Prefetch only the two supported Demucs models. No paid services."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'worker'))
import bootstrap
from demucs.pretrained import get_model
for name in ('htdemucs_ft','htdemucs_6s'):
    print('Prefetch Demucs:',name,flush=True)
    get_model(name)
print('Demucs models are ready. Install optional chord/EfficientAT weights separately.',flush=True)
