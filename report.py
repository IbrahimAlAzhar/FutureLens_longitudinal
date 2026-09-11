"""Existing-reference results only; no extraction scores or invented references."""
import pandas as pd

def report(df,out,args):
    lines=['FUTURESCOPE: EXISTING REFERENCES ONLY',f'Model used for topic titles: {args.model}',
        f'Scope: {out.name}',f'Analyzed papers: {len(df)}',
        f'Existing references: {int(df.reference.ne("").sum())}',f'Missing references: {int(df.reference.eq("").sum())}',
        f'Methods: {", ".join(args.methods)}',f'Seeds: {args.seeds}',
        'No rule extraction, LLM extraction, missing-reference fallback, or extraction scoring was run.',
        'References: ACL Future_Work; NeurIPS LLM_extracted_future_work. These columns are used as supplied.',
        'Missing reference text stays missing; it is never replaced by generated text.',
        'The LLM only labels topics. Numerical clustering/forecast results do not depend on Gemma/Mistral/Qwen.',
        'Repeated numerical results across those jobs are not independent evidence of LLM performance.',
        'Topic vocabulary is global for description and fitted on warmup years only for forecasting.',
        'Activity uses the selected input source with exact reference sentences and future-work cues removed.',
        'Activity is a text proxy for research activity, not evidence of actual research adoption.',
        'Annual denominator is all analyzed papers; partial reference coverage biases mention prevalence downward.',
        'Reference years with no available text are omitted from descriptive reference tables.',
        'Reference forecast arms are skipped if any analyzed year lacks a reference signal.',
        'Input/reference trajectory comparisons use matched reference-available papers.',
        'Missing-year scopes (NeurIPS and combined) skip temporal analyses.',
        'Forecast baselines: persistence, linear trend, current-only Ridge, history Ridge; added signal: existing reference.',
        'Positive paired MAE improvement indicates improvement over the history baseline.',
        'NMF/LDA default to 20 topics (reduced for small data); BERTopic learns a variable count.',
        'NPMI, diversity and ARI are diagnostic; silhouette spaces differ across methods.',
        'Trend tests and origin-bootstrap intervals are exploratory and do not correct serial dependence.',
        'Pretrained embedding knowledge can postdate historical papers. This is a retrospective backtest.',
        'Supplied references are not independently verified human gold; NeurIPS column is LLM-extracted.',
        'Human topic-label rating fields are blank templates, not completed evaluations.','']
    for rel in ['coverage.csv','topics/quality_summary.csv','topics/stability.csv','forecast/summary.csv','forecast/paired_improvements.csv']:
        path=out/rel
        if path.exists():
            try: frame=pd.read_csv(path);body=frame.to_string(index=False)
            except pd.errors.EmptyDataError: body='No eligible results.'
            lines.extend([rel,body,''])
    for path in sorted((out/'topics').glob('*/seed_*/*trends.csv')):
        try: frame=pd.read_csv(path)
        except pd.errors.EmptyDataError: continue
        if not frame.empty:
            frame=frame.iloc[frame.sen_slope.abs().argsort()[::-1]].head(10)
            lines.extend([str(path.relative_to(out))+' (10 largest absolute slopes)',frame.to_string(index=False),''])
    for path in sorted(out.rglob('*status.json')): lines.extend([str(path.relative_to(out)),path.read_text(),''])
    (out/'FINAL_REPORT.txt').write_text('\n'.join(lines)+'\n')
