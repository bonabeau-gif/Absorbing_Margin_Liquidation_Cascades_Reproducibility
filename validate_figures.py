#!/usr/bin/env python3
"""Check regenerated figure files against bundled expected figures."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parent
for n in range(1,10):
    a=next((ROOT/'figures').glob(f'fig{n}_*.png'))
    b=next((ROOT/'expected_figures').glob(f'fig{n}_*.png'))
    ha=hashlib.sha256(a.read_bytes()).hexdigest(); hb=hashlib.sha256(b.read_bytes()).hexdigest()
    if ha != hb: raise SystemExit(f'figure {n} mismatch')
print('figure checks: PASS (9/9)')
