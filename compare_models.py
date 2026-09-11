#!/usr/bin/env python3
"""Collect results from exactly three completed, compatible model runs."""
import argparse,json
from pathlib import Path
import pandas as pd
from common import table
p=argparse.ArgumentParser();p.add_argument('--runs',nargs=3,required=True);p.add_argument('--out-dir',required=True)
a=p.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
rows=[];spec=None;models=set()
for root in map(Path,a.runs):
    if not (root/'completion.json').exists(): raise ValueError(f'Incomplete run: {root}')
    manifest=json.loads((root/'manifest.json').read_text());cfg=manifest['config'];models.add(cfg['model'])
    comparable={k:v for k,v in cfg.items() if k not in ['out_dir','model','model_dir','chat_template']}
    comparable['files']=manifest['files'];comparable['code']=manifest['code_sha256']
    selected=pd.read_csv(root/'intermediate/selected_inputs.csv');comparable['ids']=sorted(selected.id)
    if spec is not None and comparable!=spec: raise ValueError('Runs use different data/config/code/samples')
    spec=comparable
    for scope in ['acl','neurips','combined']:
        for filename in ['forecast/summary.csv','forecast/paired_improvements.csv','topics/quality_summary.csv']:
            path=root/scope/filename
            if path.exists():
                try: df=pd.read_csv(path)
                except pd.errors.EmptyDataError: continue
                for r in df.to_dict('records'): rows.append(dict(llm_model=cfg['model'],scope=scope,result_table=filename,**r))
if models!={'qwen','gemma','mistral'}: raise ValueError('Provide one completed run for each model')
df=table(out/'model_comparison',rows)
(out/'FINAL_COMPARISON.txt').write_text('MODEL COMPARISON\nReference strata must be interpreted separately. All jobs use identical supplied references. Only topic labels depend on the selected LLM.\n\n'+df.to_string(index=False)+'\n')
