#!/usr/bin/env python3
"""Build final.txt from incomplete FutureScope results. Standard library only.
Never runs extraction, LLMs, embeddings, topic fitting, or forecasting.
"""
import argparse,csv,json,math,re,sys,itertools,hashlib,statistics,datetime
from pathlib import Path
from collections import Counter,defaultdict

csv.field_size_limit(min(sys.maxsize,2147483647))
DEFAULT='/lstr/sahara/datalab-ml/ibrahim/limagents_update/futureScope/exp_curr_without_extract/results'
WARNINGS=[]

def number(x):
    try:
        n=float(x)
        return n if math.isfinite(n) else None
    except (TypeError,ValueError): return None

def fmt(x,d=4):
    if x is None:return 'NA'
    if isinstance(x,(float,int)):
        return f'{x:.{d}f}' if math.isfinite(x) else 'NA'
    return str(x).replace('\n',' ').replace('\r',' ')

def mean(values):
    a=[v for v in values if v is not None]
    return statistics.mean(a) if a else None

def key(x):
    n=number(x)
    return str(int(n)) if n is not None and n.is_integer() else str(x)

def warn(p,message):WARNINGS.append(f'{p}: {message}')

def read_json(p):
    p=Path(p)
    if not p.exists():return None
    try:
        before=p.stat();obj=json.loads(p.read_text(encoding='utf-8-sig'));after=p.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            warn(p,'Changed while being read; skipped.');return None
        return obj
    except Exception as e:warn(p,f'Unreadable JSON: {e}');return None

def rows(base,columns=None):
    """Use one representation, CSV then JSON; never double-count paired files."""
    base=Path(base)
    if base.suffix in ('.csv','.json'):base=base.with_suffix('')
    p=base.with_suffix('.csv')
    if p.exists():
        try:
            before=p.stat();out=[]
            with p.open(encoding='utf-8-sig',newline='') as f:
                reader=csv.DictReader(f,strict=True)
                for row in reader:
                    if None in row or any(v is None for v in row.values()):
                        raise ValueError('Incomplete CSV row; file may still be writing')
                    out.append({k:v for k,v in row.items() if columns is None or k in columns})
            after=p.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise ValueError('File changed during reading')
            if out:return out,str(p)
        except Exception as e:warn(p,str(e)+'; trying JSON counterpart.')
    j=base.with_suffix('.json');obj=read_json(j)
    if isinstance(obj,list) and all(isinstance(r,dict) for r in obj):
        return [{k:v for k,v in r.items() if columns is None or k in columns} for r in obj],str(j)
    if obj is not None:warn(j,'Expected list of records; skipped.')
    return [],None

def table(lines,title,data,columns):
    lines.extend(['',title])
    if not data:lines.append('Unavailable: no usable saved records.');return
    matrix=[[fmt(row.get(k)) for k,label in columns] for row in data]
    widths=[max(len(label),max(len(row[i]) for row in matrix)) for i,(k,label) in enumerate(columns)]
    lines.append(' | '.join(label.ljust(widths[i]) for i,(k,label) in enumerate(columns)))
    lines.append('-+-'.join('-'*w for w in widths))
    lines.extend(' | '.join(v.ljust(widths[i]) for i,v in enumerate(row)) for row in matrix)

def ari(a,b):
    if len(a)!=len(b) or not a:return None
    n=len(a)
    if n<2:return 1.
    comb=lambda x:x*(x-1)/2
    cell=sum(comb(v) for v in Counter(zip(a,b)).values())
    ra=sum(comb(v) for v in Counter(a).values());rb=sum(comb(v) for v in Counter(b).values())
    expected=ra*rb/comb(n);den=(ra+rb)/2-expected
    return (cell-expected)/den if den else 1.

def training_diagnostics(seed,topic_rows):
    """Recover only metrics possible without saved embedding matrices."""
    words={key(r.get('topic')):[s.strip() for s in str(r.get('keywords','')).split(';') if s.strip()] for r in topic_rows}
    flat=[w for ws in words.values() for w in ws]
    diversity=len(set(flat))/len(flat) if flat else None
    records,source=rows(seed/'training_assignments',{'unit','text','topic'})
    if not records:return dict(diversity=diversity,npmi=None,outlier_rate=None),None,source
    if any(not {'unit','text','topic'}<=set(r) or not isinstance(r.get('text'),str) for r in records):
        warn(seed,'Training assignments lack unit/text/topic; reconstruction skipped.')
        return dict(diversity=diversity,npmi=None,outlier_rate=None),None,source
    records.sort(key=lambda r:(number(r['unit']) if number(r['unit']) is not None else math.inf,str(r['unit'])))
    if len({r['unit'] for r in records})!=len(records):
        warn(seed,'Duplicate training-unit IDs; reconstruction skipped.')
        return dict(diversity=diversity,npmi=None,outlier_rate=None),None,source
    vocabulary=set(flat);df=Counter();pairs=Counter();sig=hashlib.sha256();labels=[]
    wanted=set(tuple(sorted((a,b))) for ws in words.values() for a,b in itertools.combinations(ws,2))
    for row in records:
        sig.update(json.dumps([key(row['unit']),row['text']],ensure_ascii=False).encode());sig.update(b'\n')
        labels.append(key(row['topic']))
        present=set(re.findall(r'(?u)\b\w\w+\b',row['text'].lower()))&vocabulary
        df.update(present)
        pairs.update(pair for pair in itertools.combinations(sorted(present),2) if pair in wanted)
    n=len(records);values=[]
    for ws in words.values():
        for a,b in itertools.combinations(ws,2):
            pa=df[a]/n;pb=df[b]/n;pab=pairs[tuple(sorted((a,b)))]/n
            values.append(math.log(pab/(pa*pb))/-math.log(pab) if 0<pab<1 and pa*pb>0 else 1. if pab==1 else -1.)
    return dict(diversity=diversity,npmi=mean(values),outlier_rate=labels.count('-1')/n),dict(signature=sig.hexdigest(),labels=labels,n=n),source

def summarize_prevalence(records):
    per=defaultdict(list)
    for r in records:per[key(r.get('topic'))].append(r)
    result={}
    for topic,group in per.items():
        valid=[r for r in group if number(r.get('year')) is not None and number(r.get('prevalence')) is not None]
        valid.sort(key=lambda r:number(r['year']))
        if not valid:continue
        if len({r['year'] for r in valid})!=len(valid):
            warn('prevalence','Duplicate topic/year rows; topic '+topic+' skipped.');continue
        first,last=valid[0],valid[-1]
        result[topic]=dict(first_year=int(number(first['year'])),last_year=int(number(last['year'])),
            first_pct=100*number(first['prevalence']),last_pct=100*number(last['prevalence']),
            change_pp=100*(number(last['prevalence'])-number(first['prevalence'])),
            years=len(valid),latest_count=number(last.get('count')),latest_denominator=number(last.get('denominator')))
    return result

def year_coverage(run,scope):
    result,source=rows(run/scope/'coverage')
    if result:return result,source
    records,source=rows(run/'intermediate'/'analysis_inputs',{'id','venue','year','reference'})
    if not records:records,source=rows(run/'intermediate'/'selected_inputs',{'id','venue','year','reference'})
    groups=defaultdict(lambda:dict(n=0,reference_available=0,reference_missing=0))
    for r in records:
        if scope!='combined' and r.get('venue')!=scope:continue
        g=groups[(r.get('venue','unknown'),r.get('year',''))];g['n']+=1
        present=str(r.get('reference') or '').strip().lower() not in ('','nan','none','null','<na>')
        g['reference_available']+=int(present);g['reference_missing']+=int(not present)
    return [dict(venue=v,year=y or 'unknown',**g) for (v,y),g in sorted(groups.items(),key=lambda item:tuple(str(v) for v in item[0]))],source

def find_runs(root,explicit):
    if explicit: candidates=[Path(p).expanduser().resolve() for p in explicit]
    else:
        root=Path(root).expanduser().resolve();candidates=[]
        if (root/'manifest.json').exists() or (root/'intermediate').is_dir():candidates.append(root)
        if root.exists():
            candidates.extend(p.parent for p in root.rglob('manifest.json') if 'requests' not in p.parts)
            candidates.extend(p.parent for p in root.rglob('intermediate') if p.is_dir() and 'requests' not in p.parts)
    unique=sorted(set(candidates),key=str)
    return [p for p in unique if p.is_dir()]

def method_summary(run,scope,method,method_dir,quality,lines,args):
    seeds=sorted([p for p in method_dir.glob('seed_*') if p.is_dir()],key=lambda p:(number(p.name[5:]) if number(p.name[5:]) is not None else math.inf,p.name))
    diagnostics=[];assignments={};available=[]
    for seed in seeds:
        seed_id=key(seed.name[5:]);topics,topic_source=rows(seed/'topics')
        files=[p.stem for p in seed.glob('*.csv')]+[p.stem for p in seed.glob('*.json')]
        if not topics:
            diagnostics.append(dict(seed=seed_id,topics=None,npmi=None,diversity=None,silhouette=None,outlier_rate=None,status='No usable topics table'))
            continue
        available.append(seed)
        saved=next((r for r in quality if str(r.get('method'))==method and key(r.get('seed'))==seed_id),{})
        metric={k:number(saved.get(k)) for k in ['npmi','diversity','silhouette','outlier_rate']}
        # Even when quality.csv was never reached, training outputs can recover these metrics.
        recovered,labels,train_source=training_diagnostics(seed,topics)
        recovered_names=[]
        for k in ['npmi','diversity','outlier_rate']:
            if metric[k] is None and recovered[k] is not None:metric[k]=recovered[k];recovered_names.append(k)
        if labels:assignments[seed_id]=labels
        labels_rows,label_source=rows(seed/'llm_topic_labels_and_human_template')
        diagnostics.append(dict(seed=seed_id,topics=len(topics),**metric,
            status=f'{len(set(files))} tables; titles {sum(bool(str(r.get("label") or "").strip()) for r in labels_rows)}',
            reconstructed=','.join(recovered_names) or 'none'))
        lines.append(f'Seed {seed_id} evidence: {topic_source}; training: {train_source or "absent"}; labels: {label_source or "absent"}.')
    table(lines,f'{method.upper()} — available seeds (no missing seed is counted as zero)',diagnostics,
          [('seed','Seed'),('topics','K'),('npmi','NPMI'),('diversity','Diversity'),('silhouette','Silhouette'),('outlier_rate','Outlier rate'),('reconstructed','Reconstructed'),('status','Availability')])
    aggregates=[]
    for metric in ['topics','npmi','diversity','silhouette','outlier_rate']:
        a=[number(r.get(metric)) for r in diagnostics];a=[v for v in a if v is not None]
        aggregates.append(dict(metric=metric,n=len(a),mean=mean(a),sd=statistics.stdev(a) if len(a)>1 else None))
    table(lines,'Available-seed summaries (sample SD; NA for fewer than two observations)',aggregates,[('metric','Metric'),('n','Observed seeds'),('mean','Mean'),('sd','SD')])
    stability=[]
    for a,b in itertools.combinations(sorted(assignments),2):
        left,right=assignments[a],assignments[b]
        if left['signature']!=right['signature']:
            warn(method_dir,f'Seeds {a}/{b}: different training units; ARI not computed.');continue
        stability.append(dict(seed_a=a,seed_b=b,n=left['n'],ari=ari(left['labels'],right['labels']),source='reconstructed'))
    saved_stability,_=rows(run/scope/'topics'/'stability')
    known={tuple(sorted((key(r['seed_a']),key(r['seed_b'])))) for r in stability}
    for record in saved_stability:
        pair=tuple(sorted((key(record.get('seed_a')),key(record.get('seed_b')))))
        if record.get('method')==method and pair not in known and number(record.get('ari')) is not None:
            stability.append(dict(seed_a=pair[0],seed_b=pair[1],n=None,ari=number(record['ari']),source='saved'))
            known.add(pair)
    table(lines,'Assignment stability (reconstructed only on identical training units; includes outlier label)',stability,[('seed_a','Seed A'),('seed_b','Seed B'),('n','Units'),('ari','ARI'),('source','Source')])
    if not available:return dict(method=method,seeds='',primary_seed=None)
    with_details=[p for p in available if any((p/(name+ext)).exists() for name in ['reference_prevalence','activity_prevalence','input_reference_topic_overlap'] for ext in ['.csv','.json'])]
    primary=(with_details or available)[0] # fixed availability rule; never choose by score
    lines.extend(['',f'Detailed tables use {primary.name}: lowest seed with descriptive tables, or lowest available seed if none have them. No best-seed selection.'])
    topics,_=rows(primary/'topics');titles,_=rows(primary/'llm_topic_labels_and_human_template')
    titles={key(r.get('topic')):str(r.get('label') or '').strip() for r in titles}
    labels={key(r.get('topic')):titles.get(key(r.get('topic'))) or str(r.get('keywords') or 'Unlabeled') for r in topics}
    dictionary=[dict(topic=key(r.get('topic')),label=labels[key(r.get('topic'))],keywords=r.get('keywords','')) for r in topics]
    table(lines,'Topic dictionary (keyword fallback where no LLM title exists)',dictionary[:args.max_topics],[('topic','Topic'),('label','Title / fallback'),('keywords','Keywords')])
    if len(dictionary)>args.max_topics:lines.append(f'{len(dictionary)-args.max_topics} topic rows omitted by --max-topics; full table: {primary}/topics.csv or .json')
    for signal in ['reference','activity']:
        prev,source=rows(primary/f'{signal}_prevalence');summ=summarize_prevalence(prev)
        trends,trend_source=rows(primary/f'{signal}_trends');trends={key(r.get('topic')):r for r in trends}
        detail=[]
        for topic,item in summ.items():
            t=trends.get(topic,{})
            slope=number(t.get('sen_slope'));lo=number(t.get('ci_low'));hi=number(t.get('ci_high'))
            detail.append(dict(topic=topic,label=labels.get(topic,'Unlabeled'),**item,
                slope_pp_year=100*slope if slope is not None else None,
                ci_low_pp=100*lo if lo is not None else None,ci_high_pp=100*hi if hi is not None else None,
                p=number(t.get('p_value')),q=number(t.get('q_value_bh'))))
        detail.sort(key=lambda r:(-abs(r['change_pp']),r['topic']))
        table(lines,f'{signal.upper()} trends — largest absolute first/last prevalence changes',detail[:args.max_topics],
              [('topic','Topic'),('label','Title'),('first_year','First year'),('last_year','Last year'),('first_pct','First %'),('last_pct','Last %'),('latest_count','Last n'),('latest_denominator','Last denominator'),('change_pp','Change pp'),('slope_pp_year','Sen pp/year'),('ci_low_pp','CI low'),('ci_high_pp','CI high'),('p','p'),('q','BH q')])
        lines.append(f'Sources: prevalence {source or "absent"}; saved statistical trends {trend_source or "absent"}. No missing significance test is invented.')
    overlap,source=rows(primary/'input_reference_topic_overlap')
    if overlap:
        groups=defaultdict(list)
        for r in overlap:groups[str(r.get('reference_source','unspecified'))].append(r)
        data=[dict(source=src,n=len(g),scored=sum(number(r.get('jaccard')) is not None for r in g),mean_jaccard=mean([number(r.get('jaccard')) for r in g])) for src,g in groups.items()]
        table(lines,'Input/reference topic overlap (descriptive; NOT extraction precision/recall)',data,[('source','Reference provenance'),('n','Rows'),('scored','Scored'),('mean_jaccard','Mean Jaccard')])
    trajectory,source=rows(primary/'input_reference_trajectory_agreement')
    table(lines,'Input/reference temporal agreement',trajectory[:args.max_topics],[('topic','Topic'),('years','Years'),('spearman','Spearman'),('mean_absolute_gap','Mean absolute gap (fraction)')])
    return dict(method=method,seeds=','.join(p.name[5:] for p in available),primary_seed=primary.name[5:])

def forecast_report(scope_dir,lines):
    summary,source=rows(scope_dir/'forecast'/'summary');rebuilt=False
    if not summary:
        folds,source=rows(scope_dir/'forecast'/'fold_metrics');groups=defaultdict(list)
        for r in folds:groups[(r.get('method'),r.get('horizon'),r.get('model'))].append(r)
        for (method,horizon,model),group in groups.items():
            # Abort duplicate origins within a forecast model rather than inflate N.
            if len({r.get('origin') for r in group})!=len(group):
                warn(scope_dir,f'Duplicate forecast origins for {method}/{horizon}/{model}; skipped.');continue
            summary.append(dict(method=method,horizon=horizon,model=model,n_test_origins=len(group),
                **{k:mean([number(r.get(k)) for r in group]) for k in ['mae','rmse','spearman','precision_at_5','recall_at_5','ndcg_at_5']}))
        rebuilt=bool(summary)
    table(lines,'FORECASTING'+(' (means recovered from saved fold metrics)' if rebuilt else ''),summary,
          [('method','Method'),('horizon','Horizon'),('model','Predictor'),('n_test_origins','Test origins'),('mae','MAE'),('rmse','RMSE'),('spearman','Spearman'),('precision_at_5','P@5'),('recall_at_5','R@5'),('ndcg_at_5','nDCG@5')])
    if not summary:lines.append('No completed forecast summary/fold results found. Do not claim predictive improvement from these artifacts.')
    else:lines.append(f'Source: {source}. Different methods may have different topic targets; assess added-signal benefit within each method.')
    paired,source=rows(scope_dir/'forecast'/'paired_improvements')
    if paired:table(lines,'Saved paired forecast improvements (positive = lower MAE than history baseline)',paired,[('method','Method'),('horizon','Horizon'),('contrast','Contrast'),('n_test_origins','Origins'),('mae_improvement','MAE improvement'),('ci_low','CI low'),('ci_high','CI high')])
    status=read_json(scope_dir/'forecast'/'status.json')
    if status is not None:lines.extend(['Forecast status:',json.dumps(status,ensure_ascii=False)])
    return bool(summary)

def build(args):
    WARNINGS.clear()
    runs=find_runs(args.root,args.runs)
    lines=['FUTURESCOPE — REPORT FROM EXISTING ARTIFACTS',
        'Generated (UTC): '+datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'This is a recovered-results report, not a declaration that the original jobs completed.',
        'No extraction, LLM calls, topic fitting, embeddings, or forecasts were executed by this script.',
        'CSV is preferred; JSON is used when CSV is missing/empty/unreadable. They are never counted twice.',
        'NA means unavailable/undefined, never zero. Partially available seeds are reported with their actual count.',
        'Different runs/scopes/models are kept separate; results are NOT pooled across configurations.',
        'Gemma/Mistral/Qwen only title topics in the no-extraction pipeline: matching numeric results are not independent LLM experiments.',
        'Recovered NPMI uses binary word co-occurrence on saved training text units; diversity uses saved keywords.',
        'Silhouette is reported only when saved. No embedding model is loaded to reconstruct it.',
        'Missing human judgments, reference text, forecasts, confidence intervals or p-values are never invented.',
        'Trends are descriptive; exploratory p/q values do not establish causality or account for all temporal dependence.',
        'Input/reference overlap is not extraction accuracy. Supplied references have unverified provenance.',
        'Repeated seeds do not add papers to the dataset. ACL and NeurIPS papers also recur in the combined scope.',
        'Files written concurrently can produce incomplete snapshots. Stable readable tables are used with provenance.',
        '']
    inventory=[]
    for run in runs:
        print('Reading:',run,flush=True)
        manifest=read_json(run/'manifest.json') or {};cfg=manifest.get('config',{}) if isinstance(manifest,dict) else {}
        if not isinstance(cfg,dict): cfg={}
        model=cfg.get('model') or next((x for x in reversed(run.parts) if x in ['gemma','mistral','qwen']),'unknown')
        completion=read_json(run/'completion.json');failure=read_json(run/'failure.json')
        completed=isinstance(completion,dict) and completion.get('status')=='complete'
        lines.extend(['\n'+'='*100,f'MODEL: {model} | RUN: {run.name}',f'Path: {run}',
            'Original run status: '+('completion marker present' if completed else 'NO completion marker; partial snapshot'),
            'Configuration: '+json.dumps(cfg,ensure_ascii=False,sort_keys=True)])
        if failure:lines.append('Saved failure: '+str(failure.get('error',failure) if isinstance(failure,dict) else failure))
        any_scope=False
        for scope in ['acl','neurips','combined']:
            scope_dir=run/scope
            if not scope_dir.is_dir():continue
            any_scope=True;lines.extend(['\n'+'-'*80,'SCOPE: '+scope.upper()])
            coverage,source=year_coverage(run,scope)
            table(lines,'Dataset coverage',coverage,[('venue','Venue'),('year','Year'),('n','Papers'),('reference_available','With reference'),('reference_missing','Missing reference')])
            if coverage:lines.append(f'Coverage source: {source}; total analyzed papers in this scope: {sum(number(r.get("n")) or 0 for r in coverage):.0f}.')
            refstatus=read_json(scope_dir/'reference_status.json')
            if refstatus is not None:lines.append('Reference status: '+json.dumps(refstatus,ensure_ascii=False))
            quality,_=rows(scope_dir/'topics'/'quality')
            found=[]
            for method in ['nmf','lda','bertopic']:
                folder=scope_dir/'topics'/method
                if not folder.is_dir():
                    lines.append(method.upper()+': no saved method directory; unavailable, not a failed score.');continue
                try:
                    info=method_summary(run,scope,method,folder,quality,lines,args);found.append(info)
                except Exception as e:
                    warn(folder,f'Could not finish this method summary: {type(e).__name__}: {e}')
                    lines.append(f'{method}: remaining tables skipped after an unreadable/incompatible artifact; see warnings.')
            forecast=forecast_report(scope_dir,lines)
            inventory.append(dict(model=model,run=run.name,scope=scope,complete=completed,
                methods='; '.join(f'{r["method"]} seeds[{r["seeds"]}]' for r in found) or 'none',forecast='available' if forecast else 'unavailable'))
            lines.extend(['','PAPER/APPENDIX STATUS NOTE:',
                f'This {scope} snapshot contains '+('; '.join(f'{r["method"].upper()} seeds {r["seeds"] or "none"}' for r in found) or 'no readable topic methods')+'.',
                'Only the observed seeds contribute to reported summaries. Missing seeds/methods were not imputed.',
                'Saved forecasting results are included.' if forecast else 'These artifacts support descriptive analysis only; no forecasting result is available.'])
        if not any_scope:lines.append('No acl/neurips/combined result directories found.')
    if not runs:lines.append('No existing run directories were found. Check --root or pass explicit --runs paths.')
    inventory_lines=[]
    table(inventory_lines,'AVAILABLE RESULT INVENTORY',inventory,[('model','Model'),('run','Run'),('scope','Scope'),('complete','Completed marker'),('methods','Methods/seeds'),('forecast','Forecast')])
    lines[16:16]=inventory_lines
    lines.extend(['','READ WARNINGS / SKIPPED FILES']+(WARNINGS or ['None detected.']))
    output=Path(args.output).expanduser().resolve();output.parent.mkdir(parents=True,exist_ok=True)
    if any(output==r/'FINAL_REPORT.txt' or output==r/'final.txt' and (output.exists()) for r in runs):
        print('Note: writing the requested report path; original CSV/JSON artifacts are untouched.',flush=True)
    temporary=output.with_name(output.name+'.tmp')
    temporary.write_text('\n'.join(lines)+'\n',encoding='utf-8');temporary.replace(output)
    print(f'Saved {output} | runs={len(runs)} | warnings={len(WARNINGS)}',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=DEFAULT,help='Results root; scans all model runs separately')
    p.add_argument('--runs',nargs='+',help='Optional explicit run directories instead of discovery')
    p.add_argument('--output',default='final.txt')
    p.add_argument('--max-topics',type=int,default=100,help='Maximum displayed topics per detailed table')
    args=p.parse_args()
    if args.max_topics<1:p.error('--max-topics must be positive')
    build(args)
if __name__=='__main__':main()
