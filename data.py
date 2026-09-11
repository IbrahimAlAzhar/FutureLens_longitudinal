"""Exact requested input mapping; no publication years inferred from prose."""
import re
from pathlib import Path
import pandas as pd
from common import clean, sentences, digest, file_hash

def year_value(x):
    s = clean(x)
    if re.fullmatch(r'(19|20)\d{2}(\.0)?', s): return int(float(s))
    return None

def load_data(args):
    root = Path(args.data_dir); records=[]; files=[]
    paths = [root/f'ACL_{y}_updated.csv' for y in range(12,23)]
    paths += [root/'df_neurips_rag_gen_fw_from_paper.csv']
    if args.dataset == 'acl': paths=paths[:-1]
    if args.dataset == 'neurips': paths=paths[-1:]
    for p in paths:
        if not p.exists(): raise FileNotFoundError(p)
        df = pd.read_csv(p, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        acl=p.name.startswith('ACL_'); venue='acl' if acl else 'neurips'
        cols=['abstract']+[f'col_{i}' for i in range(1,7)] if acl else ['df_Concatenated Text']
        refcol='Future_Work' if acl else 'LLM_extracted_future_work'
        required=cols+[refcol]
        missing=set(required)-set(df.columns)
        if missing: raise ValueError(f'{p}: missing columns {sorted(missing)}')
        yc = args.neurips_year_column
        if not acl and not yc:
            yc=next((c for c in df if c.lower() in {'year','publication_year','publication year','pub_year'}), None)
        if not acl and yc and yc not in df: raise ValueError(f'Year column {yc!r} missing')
        files.append({'path':str(p),'sha256':file_hash(p),'rows':len(df),'year_column':yc if not acl else 'filename'})
        for i,row in df.iterrows():
            text='\n'.join(clean(row[c]) for c in cols if clean(row[c]))
            ref=clean(row[refcol])
            records.append(dict(id=f'{p.stem}:{i}',venue=venue,year=2000+int(p.stem.split('_')[1]) if acl else year_value(row[yc]) if yc else None,
                input=text,abstract=clean(row.get('abstract','')),reference=ref,
                reference_source=('acl_supplied' if acl else 'neurips_llm_reference') if ref else 'missing',
                content_hash=digest(text),source_file=p.name,source_row=int(i)))
    df=pd.DataFrame(records)
    # Deduplicate exact normalized inputs within venue, before sampling, preserving audit.
    df['duplicate']=df.duplicated(['venue','content_hash']) & df.input.ne('')
    audit=df[['id','venue','year','duplicate']].copy()
    df=df[~df.duplicate & df.input.ne('')].copy()
    # Cap per venue/year, preserving all years for a temporal smoke run.
    if args.num_samples != -1:
        selected=[]
        for _,g in df.groupby(['venue','year'],dropna=False):
            selected.extend(g.sample(min(len(g),args.num_samples),random_state=args.seed).index)
        df=df.loc[selected]
    return df.sort_values(['venue','year','id']).reset_index(drop=True),files,audit

def activity(row, source='input'):
    text=clean(row.get('abstract')) if source=='abstract' and row['venue']=='acl' else clean(row['input'])
    # Deterministic activity sanitization only; no extraction or model calls.
    excluded={s.casefold() for k in ['reference'] for s in sentences(row.get(k,''))}
    CUE=re.compile(r'\bfuture[\s_]+work\b|\bfuture research\b|\bfuture direction|\bwe (?:plan|intend|aim) to\b|\bremains? to be\b|\bfurther (?:work|research|investigation)\b',re.I)
    return '\n'.join(s for s in sentences(text) if s.casefold() not in excluded and not CUE.search(s))
