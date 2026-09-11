#!/usr/bin/env python3
import argparse,importlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--model-dir',required=True)
p.add_argument('--methods',nargs='+',default=['nmf','lda','bertopic']);p.add_argument('--embedding-model',required=True)
p.add_argument('--no-topic-labels',action='store_true');a=p.parse_args()
for mod in ['numpy','pandas','scipy','sklearn']: importlib.import_module(mod)
if not a.no_topic_labels:
    for mod in ['transformers','vllm']: importlib.import_module(mod)
    if not Path(a.model_dir,'config.json').exists(): raise SystemExit('Missing LLM model: '+a.model_dir)
if 'bertopic' in a.methods:
    for mod in ['bertopic','sentence_transformers','umap','hdbscan']: importlib.import_module(mod)
    from sentence_transformers import SentenceTransformer
    SentenceTransformer(a.embedding_model,device='cpu',local_files_only=True)
from run_experiments import parser
from data import load_data
args=parser().parse_args(['--out-dir','unused','--model','qwen','--model-dir',a.model_dir,'--data-dir',a.data_dir,'--num-samples','1'])
df,files,audit=load_data(args)
print(json.dumps(dict(status='preflight passed',pipeline='existing_references_only',files=len(files),smoke_rows=len(df),reference_rows=int(df.reference.ne('').sum()),unknown_year_rows=int(df.year.isna().sum())),indent=2))
