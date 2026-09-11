#!/usr/bin/env python3
import argparse,datetime,json,platform,sys
from pathlib import Path
import importlib.metadata
from common import write_json,table,file_hash,digest

ROOT='/lstr/sahara/datalab-ml/ibrahim/limagents_update/futureScope'
def parser():
    p=argparse.ArgumentParser(description='FutureScope existing-reference topic and temporal experiments; no extraction')
    p.add_argument('--data-dir',default=ROOT+'/data');p.add_argument('--out-dir',required=True)
    p.add_argument('--model',required=True,choices=['gemma','qwen','mistral']);p.add_argument('--model-dir',required=True)
    p.add_argument('--chat-template',default=None,help='Same resolved Jinja file passed to the vLLM server')
    p.add_argument('--vllm-url',default='http://127.0.0.1:8000/v1')
    p.add_argument('--experiment',choices=['all','topics','forecast'],default='all')
    p.add_argument('--dataset',choices=['all','acl','neurips'],default='all')
    p.add_argument('--num-samples',type=int,default=10,help='Papers PER venue/year; -1=all; NeurIPS without year is one group.')
    p.add_argument('--neurips-year-column',default=None)
    p.add_argument('--activity-source',choices=['input','abstract'],default='input')
    p.add_argument('--methods',nargs='+',choices=['nmf','lda','bertopic'],default=['nmf','lda','bertopic'])
    p.add_argument('--embedding-model',default='/lstr/sahara/datalab-ml/ibrahim/models/all-MiniLM-L6-v2')
    p.add_argument('--num-topics',type=int,default=20);p.add_argument('--max-features',type=int,default=20000)
    p.add_argument('--min-cluster-size',type=int,default=10)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--horizons',type=int,nargs='+',default=[1,2,3]);p.add_argument('--warmup-years',type=int,default=3)
    p.add_argument('--trend-window',type=int,default=3)
    p.add_argument('--min-train-origins',type=int,default=3);p.add_argument('--ridge-alpha',type=float,default=.01)
    p.add_argument('--chunk-tokens',type=int,default=6000);p.add_argument('--context-length',type=int,default=16384)
    p.add_argument('--max-tokens',type=int,default=1024)
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--no-topic-labels',action='store_true')
    return p

def run(args):
    if args.num_samples==0 or args.num_samples < -1: raise ValueError('--num-samples must be positive or -1')
    for key in ['num_topics','max_features','min_cluster_size','warmup_years','trend_window','min_train_origins','chunk_tokens','context_length','max_tokens','bootstrap']:
        if getattr(args,key)<=0: raise ValueError(f'{key} must be positive')
    if any(h<1 for h in args.horizons): raise ValueError('Invalid sample count or horizon')
    if args.chunk_tokens+args.max_tokens+1024>args.context_length: raise ValueError('Chunk and completion budget exceed context')
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    from data import load_data,activity
    df,files,audit=load_data(args)
    if df.empty: raise ValueError('No nonempty input rows')
    cfg=vars(args).copy();cfg.pop('vllm_url',None)
    code={p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')}
    identity=digest(dict(config=cfg,files=files,code=code));manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity']!=identity:
        raise ValueError('Output directory belongs to a different configuration/data/code. Choose a new --out-dir.')
    # Small model/tokenizer config hashes plus weight inventory; avoids rehashing tens of GB per job.
    model_root=Path(args.model_dir)
    model_identity={'path':args.model_dir,'configuration_sha256':{},'weight_inventory':[]}
    if model_root.exists():
        for p in model_root.glob('*.json'):
            model_identity['configuration_sha256'][p.name]=file_hash(p)
        for pattern in ['*.safetensors','*.bin']:
            for p in sorted(model_root.glob(pattern)):
                stat=p.stat();model_identity['weight_inventory'].append(dict(name=p.name,size=stat.st_size,mtime_ns=stat.st_mtime_ns))
    model_record=out/'model_identity.json'
    if model_record.exists() and json.loads(model_record.read_text())!=model_identity:
        raise ValueError('Local model files changed; use a new output directory')
    write_json(model_record,model_identity)
    for marker in ['completion.json','failure.json']:
        if (out/marker).exists(): (out/marker).unlink()
    versions={}
    for name in ['numpy','pandas','scipy','scikit-learn','transformers','vllm','bertopic','sentence-transformers','umap-learn','hdbscan','bitsandbytes']:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]='not installed'
    write_json(manifest,dict(identity=identity,config=cfg,files=files,code_sha256=code,versions=versions,python=sys.version,platform=platform.platform()))
    table(out/'intermediate/input_audit',audit);table(out/'intermediate/selected_inputs',df)
    print(f'[load] {len(df)} selected papers; {int(df.reference.ne("").sum())} existing references; no extraction.',flush=True)
    df['reference_verbatim_in_input']=[bool(r.reference) and all(s in r.input for s in __import__('common').sentences(r.reference)) for _,r in df.iterrows()]
    df['activity']=[activity(r,args.activity_source) for _,r in df.iterrows()]
    table(out/'intermediate/analysis_inputs',df)
    client=None
    if not args.no_topic_labels and args.experiment in ['all','topics']:
        from llm import Client
        client=Client(args)
    from report import report
    scopes={'acl':df[df.venue=='acl'],'neurips':df[df.venue=='neurips'],'combined':df}
    for name,subset in scopes.items():
        if subset.empty: continue
        dest=out/name;dest.mkdir(parents=True,exist_ok=True)
        coverage=[]
        for (venue,year),g in subset.groupby(['venue','year'],dropna=False):
            coverage.append(dict(venue=venue,year=year,n=len(g),reference_available=int(g.reference.ne('').sum()),reference_missing=int(g.reference.eq('').sum()),
                activity_nonempty=int(g.activity.ne('').sum())))
        table(dest/'coverage',coverage)
        write_json(dest/'reference_status.json',dict(available=int(subset.reference.ne('').sum()),missing=int(subset.reference.eq('').sum()),fallback_used=False))
        if args.experiment in ['all','topics']:
            from topics import descriptive
            descriptive(subset,dest/'topics',args,None if args.no_topic_labels else client)
        if args.experiment in ['all','forecast']:
            from temporal import forecast
            forecast(subset,dest/'forecast',args)
        report(subset,dest,args)
    write_json(out/'completion.json',dict(status='complete',timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),identity=identity))
    (out/'FINAL_REPORT.txt').write_text('\n\n'.join((out/name/'FINAL_REPORT.txt').read_text() for name in scopes if (out/name/'FINAL_REPORT.txt').exists()))
    print(f'DONE: {out}/FINAL_REPORT.txt',flush=True)

if __name__=='__main__':
    args=parser().parse_args()
    try: run(args)
    except Exception as e:
        import traceback
        write_json(Path(args.out_dir)/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise
