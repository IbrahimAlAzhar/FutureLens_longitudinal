"""Descriptive trends and chronological holdout forecasts in a warmup-only topic space."""
import numpy as np
import pandas as pd
from scipy.stats import kendalltau,theilslopes,spearmanr
from sklearn.linear_model import Ridge
from common import bh,table,write_json
from statistics_utils import bootstrap

def trend_table(prev):
    rows=[]
    for topic,g in prev.groupby('topic'):
        g=g.sort_values('year');x=g.year.to_numpy();y=g.prevalence.to_numpy()
        if len(g)<3: continue
        slope,intercept,low,high=theilslopes(y,x,alpha=.95)
        tau,p=kendalltau(x,y)
        rows.append(dict(topic=int(topic),years=len(g),sen_slope=slope,ci_low=low,ci_high=high,kendall_tau=tau,p_value=p))
    if rows:
        q=bh([r['p_value'] for r in rows])
        for r,v in zip(rows,q): r['q_value_bh']=v
    return rows

def rank_metrics(actual,pred,current,k=5):
    true_gain=actual-current;pred_gain=pred-current;k=min(k,len(actual))
    # Relevance requires positive observed growth. Ties broken by fixed topic index.
    actual_order=np.argsort(-true_gain,kind='stable')
    relevant=set(i for i in actual_order[:k] if true_gain[i]>0)
    retrieved=set(np.argsort(-pred_gain,kind='stable')[:k]);hits=len(relevant&retrieved)
    rel=np.maximum(true_gain,0)
    discounts=np.log2(np.arange(2,k+2))
    dcg=(rel[np.argsort(-pred_gain,kind='stable')[:k]]/discounts).sum()
    ideal=(np.sort(rel)[::-1][:k]/discounts).sum()
    return dict(precision_at_5=hits/k,recall_at_5=hits/len(relevant) if relevant else np.nan,
                ndcg_at_5=dcg/ideal if ideal else np.nan,k_used=k)

def forecast(df,out,args):
    from topics import TopicSpace,units,assign,prevalence
    out.mkdir(parents=True,exist_ok=True)
    missing=int(df.year.isna().sum())
    # Do not silently relabel ACL-only evidence as a combined longitudinal study.
    if missing:
        write_json(out/'status.json',{'status':'skipped','reason':'Publication year missing; no years inferred from text.',
                                     'missing_year_rows':missing,'total_rows':len(df)})
        return
    years=sorted(int(y) for y in df.year.unique())
    if len(years)<args.warmup_years+args.min_train_origins+2:
        write_json(out/'status.json',{'status':'skipped','reason':'Too few distinct years for warmup and rolling training','years':years});return
    cutoff=years[args.warmup_years-1];warm=df[df.year<=cutoff]
    docs=list(dict.fromkeys(units(warm,'activity')+units(warm,'reference')))
    data=df.copy()
    signals=['reference']
    detail=[];folds=[];coefs=[];status=[]
    # Do not present an absent reference series as a measured all-zero gold signal.
    for signal in list(signals):
        if signal.startswith('reference'):
            available=data.groupby('year')[signal].apply(lambda x:x.ne('').any())
            if not available.all():
                signals.remove(signal)
                status.append(dict(signal=signal,status='skipped',reason='Reference signal absent in one or more analyzed years',
                    missing_years=[int(y) for y in available.index[~available]]))
    for method in args.methods:
        dest=out/method;dest.mkdir(parents=True,exist_ok=True)
        try: space=TopicSpace(method,args,args.seeds[0]).fit(docs)
        except ValueError as e:
            status.append(dict(method=method,status='skipped',reason=str(e)));continue
        write_json(dest/'topic_fit_manifest.json',dict(max_fit_year=cutoff,fit_ids=warm.id.tolist(),fit_units=len(docs),seed=args.seeds[0],method=method))
        table(dest/'frozen_topics',[dict(topic=k,keywords='; '.join(space.words[k])) for k in space.ids])
        mats={}
        for column in ['activity']+signals:
            ass=assign(space,data,column)
            table(dest/f'{column}_assignments',[dict(id=i,topics=sorted(k)) for i,k in ass.items()])
            p=table(dest/f'{column}_prevalence',prevalence(data,ass,space.ids))
            mats[column]=p.pivot(index='year',columns='topic',values='prevalence').reindex(index=years,columns=space.ids)
        a=mats['activity']
        for h in args.horizons:
            eligible=[t for t in years if t>=cutoff and t-1 in years and t+h in years]
            for t in eligible:
                train=[s for s in eligible if s+h<=t]
                if len(train)<args.min_train_origins: continue
                def slope(s):
                    observed=[y for y in years if s-args.trend_window+1<=y<=s]
                    x=np.array(observed,dtype=float);x-=x.mean()
                    return (x[:,None]*a.loc[observed].values).sum(axis=0)/(x*x).sum() if len(x)>1 else np.zeros(len(space.ids))
                current=a.loc[t].values;delta=slope(t);truth=a.loc[t+h].values
                targets=np.concatenate([a.loc[s+h].values for s in train])
                predictions={'persistence':current,'linear_trend':np.clip(current+h*delta,0,1)}
                specs=[('current_only',None,False),('activity_history',None,True)]+[(f'history_plus_{sig}',sig,True) for sig in signals]
                for name,sig,history in specs:
                    def features(s):
                        cols=[a.loc[s].values]
                        if history: cols.append(slope(s))
                        if sig: cols.append(mats[sig].loc[s].values)
                        return np.column_stack(cols)
                    x=np.vstack([features(s) for s in train]);xt=features(t)
                    model=Ridge(alpha=args.ridge_alpha).fit(x,targets)
                    predictions[name]=np.clip(model.predict(xt),0,1)
                    coefs.append(dict(method=method,horizon=h,origin=t,model=name,n_train_origins=len(train),
                                      max_train_target_year=max(s+h for s in train),fw_beta=model.coef_[-1] if sig else np.nan,
                                      intercept=model.intercept_,coefficients=model.coef_.tolist()))
                for name,pred in predictions.items():
                    sp=spearmanr(truth,pred).statistic if np.std(truth)>0 and np.std(pred)>0 else np.nan
                    folds.append(dict(method=method,horizon=h,origin=t,target=t+h,model=name,n_topics=len(space.ids),
                        mae=float(np.mean(np.abs(truth-pred))),rmse=float(np.sqrt(np.mean((truth-pred)**2))),spearman=sp,
                        **rank_metrics(truth,pred,current)))
                    detail.extend(dict(method=method,horizon=h,origin=t,target=t+h,model=name,topic=k,
                                       actual=float(y),predicted=float(p),current=float(c)) for k,y,p,c in zip(space.ids,truth,pred,current))
        status.append(dict(method=method,status='completed',warmup_cutoff=cutoff))
    f=table(out/'fold_metrics',folds);table(out/'predictions',detail);table(out/'coefficients',coefs)
    summary=[];paired=[]
    if not f.empty:
        for (method,h,name),g in f.groupby(['method','horizon','model']):
            row=dict(method=method,horizon=h,model=name,n_test_origins=len(g))
            for metric in ['mae','rmse','spearman','precision_at_5','recall_at_5','ndcg_at_5']:
                row[metric]=g[metric].mean()
            summary.append(row)
        for (method,h),g in f.groupby(['method','horizon']):
            pivot=g.pivot(index='origin',columns='model',values='mae')
            for name in [c for c in pivot if c.startswith('history_plus_')]:
                delta=pivot['activity_history']-pivot[name]
                lo,hi=bootstrap(delta,args.seed,args.bootstrap) if len(delta)>=3 else (np.nan,np.nan)
                paired.append(dict(method=method,horizon=h,contrast=name+' vs activity_history',n_test_origins=len(delta),
                    mae_improvement=delta.mean(),ci_low=lo,ci_high=hi,
                    uncertainty_note='Exploratory origin bootstrap; dependent overlapping horizons, not a significance test.' if len(delta)>=3 else 'Too few origins for interval.'))
    else: status.append(dict(status='skipped',reason='No eligible origin/target pairs after chronological training requirements'))
    table(out/'summary',summary);table(out/'paired_improvements',paired);write_json(out/'status.json',status)
