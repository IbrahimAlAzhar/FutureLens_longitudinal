"""Serialization, provenance, and deterministic text units."""
import hashlib, json, re
from pathlib import Path
import numpy as np
import pandas as pd

def clean(x):
    if x is None or (not isinstance(x, str) and pd.isna(x)): return ''
    s = str(x).strip()
    return '' if s.lower() in {'nan', 'none', 'null', '<na>'} else s

def sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', clean(text)) if s.strip()]

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()

def file_hash(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()

def safe(x):
    if isinstance(x, dict): return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)): return [safe(v) for v in x]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x, Path): return str(x)
    return x

def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(safe(obj), indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def table(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(path.with_suffix('.csv'), index=False)
    write_json(path.with_suffix('.json'), df.to_dict('records'))
    return df

def bh(p):
    p = np.asarray(p, float); out = np.full(len(p), np.nan)
    ix = np.where(np.isfinite(p))[0]; order = ix[np.argsort(p[ix])]
    if len(order): out[order] = np.minimum(1, np.minimum.accumulate((p[order]*len(order)/np.arange(1,len(order)+1))[::-1])[::-1])
    return out
