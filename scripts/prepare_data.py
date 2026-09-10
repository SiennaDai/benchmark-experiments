#!/usr/bin/env python3
"""Prepare deterministic synthetic data or plan a bounded SlimPajama subset."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from data.frozen_tokens import sha256_file
def write_synthetic(output,seed,counts,overfit):
 output.mkdir(parents=True,exist_ok=True);splits={}
 for index,(name,count) in enumerate(counts.items()):
  rng=np.random.default_rng(seed+index)
  if overfit: pattern=np.asarray([(i*i+3*i+7)%256 for i in range(64)],dtype="<u2");arr=np.resize(pattern,count).astype("<u2")
  else:arr=rng.integers(0,256,size=count,dtype=np.uint16).astype("<u2")
  path=output/f"{name}.bin";arr.tofile(path);splits[name]={"file":path.name,"tokens":count,"sha256":sha256_file(path)}
 base={"schema_version":1,"protocol_id":"overfit-pattern-v1" if overfit else "synthetic-v1","source":{"kind":"synthetic","seed":seed},"file_pool":[],"document_fingerprint":"synthetic","split_algorithm":"independent seeded streams","tokenizer":{"name":"none","fingerprint":None},"eot":None,"dtype":"uint16","endianness":"little","max_token_id":255,"splits":splits};base["fingerprint"]=hashlib.sha256(json.dumps(base,sort_keys=True,separators=(",",":")).encode()).hexdigest();(output/"manifest.json").write_text(json.dumps(base,indent=2,sort_keys=True)+"\n");print(output/"manifest.json")

def prepare_slimpajama(output, limit, plan_only, caps, source_revision=None):
 from huggingface_hub import HfApi,hf_hub_download
 api=HfApi(); info=api.dataset_info("DKYoon/SlimPajama-6B",**({"revision":source_revision} if source_revision else {})); revision=info.sha
 entries=api.list_repo_tree("DKYoon/SlimPajama-6B",repo_type="dataset",revision=revision,recursive=True,expand=True)
 files=sorted([{"path":x.path,"size":x.size} for x in entries if getattr(x,"size",None) and x.path.endswith(".parquet")],key=lambda x:x["path"])
 selected=[];total=0
 for f in files:
  if total+f["size"]<=limit:selected.append(f);total+=f["size"]
 plan={"dataset":"DKYoon/SlimPajama-6B","revision":revision,"source_max_bytes":limit,"available_files":files,"selected_files":selected,"selected_bytes":total}
 output.mkdir(parents=True,exist_ok=True);(output/"source_plan.json").write_text(json.dumps(plan,indent=2)+"\n")
 if plan_only:print(json.dumps(plan,indent=2));return
 if not selected:raise SystemExit(f"No complete Parquet file fits source-max-bytes={limit}; see {output/'source_plan.json'}")
 import pyarrow.parquet as pq,tiktoken
 enc=tiktoken.get_encoding("gpt2"); rank_hash=hashlib.sha256()
 for token,rank in sorted(enc._mergeable_ranks.items()):rank_hash.update(token);rank_hash.update(rank.to_bytes(4,"little"))
 tokens={k:[] for k in caps};docs=[];seen=set();pool=[]
 for source in selected:
  local=Path(hf_hub_download("DKYoon/SlimPajama-6B",source["path"],repo_type="dataset",revision=revision));pool.append({**source,"sha256":sha256_file(local)})
  parquet=pq.ParquetFile(local)
  for batch in parquet.iter_batches(columns=["text"],batch_size=128):
   for raw in batch.column(0).to_pylist():
    text=(raw or "").replace("\r\n","\n");content_hash=hashlib.sha256(text.encode()).hexdigest()
    if content_hash in seen:continue
    seen.add(content_hash);bucket=int(hashlib.sha256(("split-v1:"+content_hash).encode()).hexdigest(),16)%10000;split="train" if bucket<9800 else ("validation" if bucket<9900 else "test")
    if len(tokens[split])>=caps[split]:continue
    ids=enc.encode_ordinary(text)+[enc.eot_token];tokens[split].extend(ids);docs.append({"content_sha256":content_hash,"split":split,"tokens":len(ids)})
  if all(len(tokens[k])>=caps[k] for k in caps):break
 if any(len(tokens[k])<caps[k] for k in caps):raise SystemExit(f"Selected files did not supply target split sizes: { {k:len(v) for k,v in tokens.items()} }")
 splits={}
 for name,values in tokens.items():
  path=output/f"{name}.bin";np.asarray(values,dtype="<u2").tofile(path);splits[name]={"file":path.name,"tokens":len(values),"sha256":sha256_file(path)}
 manifest={"schema_version":1,"protocol_id":"slimpajama-hash-split-v1","source":{"dataset":"DKYoon/SlimPajama-6B","revision":revision},"file_pool":pool,"document_fingerprint":hashlib.sha256(json.dumps(docs,sort_keys=True).encode()).hexdigest(),"documents":docs,"split_algorithm":"sha256(split-v1:content_hash) mod 10000; train<9800,val<9900,test","tokenizer":{"name":"tiktoken:gpt2","version":__import__('importlib.metadata').metadata.version('tiktoken'),"fingerprint":rank_hash.hexdigest()},"eot":enc.eot_token,"dtype":"uint16","endianness":"little","max_token_id":enc.eot_token,"splits":splits};manifest["fingerprint"]=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest();(output/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");print(output/"manifest.json")
def main():
 p=argparse.ArgumentParser();p.add_argument("--kind",choices=["synthetic","slimpajama"],required=True);p.add_argument("--output",required=True);p.add_argument("--plan-only",action="store_true");p.add_argument("--source-max-bytes",type=int,default=536870912);p.add_argument("--source-revision",help="optional Hugging Face dataset revision to resolve and pin");p.add_argument("--seed",type=int,default=1337);p.add_argument("--train-tokens",type=int,default=131073);p.add_argument("--validation-tokens",type=int,default=8193);p.add_argument("--test-tokens",type=int,default=8193);p.add_argument("--overfit",action="store_true");a=p.parse_args();output=Path(a.output)
 if a.kind=="synthetic":
  if a.plan_only:print(json.dumps({"kind":"synthetic","download_bytes":0,"output":str(output.resolve())},indent=2))
  else:write_synthetic(output,a.seed,{"train":a.train_tokens,"validation":a.validation_tokens,"test":a.test_tokens},a.overfit)
 else:
  counts={"train":a.train_tokens,"validation":a.validation_tokens,"test":a.test_tokens}
  if (a.train_tokens,a.validation_tokens,a.test_tokens)==(131073,8193,8193):counts={"train":8388609,"validation":65537,"test":65537}
  prepare_slimpajama(output,a.source_max_bytes,a.plan_only,counts,a.source_revision)
if __name__=="__main__":main()
