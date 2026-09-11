#!/usr/bin/env python3
"""One live topic-title probe only; no sentence extraction calls."""
import traceback
from pathlib import Path
from run_experiments import parser
from llm import Client,ask_validated,topic_label
from common import write_json
if __name__=='__main__':
    args=parser().parse_args()
    try:
        client=Client(args)
        title='Label this research topic in at most eight words. Return JSON {"label":"..."}. Keywords: translation, language, multilingual.'
        label=ask_validated(client,title,topic_label)
        write_json(Path(args.out_dir)/'schema_smoke.json',dict(status='passed',stage='topic_label_only',label=label))
    except Exception as e:
        record=dict(status='failed',error=repr(e),traceback=traceback.format_exc())
        write_json(Path(args.out_dir)/'failure.json',record);raise
