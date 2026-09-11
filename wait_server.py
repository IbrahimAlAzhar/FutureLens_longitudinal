#!/usr/bin/env python3
import argparse,json,os,time,urllib.request,urllib.error
p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--pid',type=int,required=True);p.add_argument('--model',required=True);p.add_argument('--timeout',type=int,default=1800)
a=p.parse_args();end=time.monotonic()+a.timeout
while time.monotonic()<end:
    try: os.kill(a.pid,0)
    except ProcessLookupError: raise SystemExit('vLLM exited; inspect vllm_server.log')
    try:
        with urllib.request.urlopen(a.url+'/v1/models',timeout=5) as f: result=json.load(f)
        if a.model in [m['id'] for m in result['data']]: print('vLLM ready',flush=True);break
    except (urllib.error.URLError,TimeoutError): pass
    time.sleep(5)
else: raise SystemExit('vLLM startup timed out; inspect vllm_server.log')
