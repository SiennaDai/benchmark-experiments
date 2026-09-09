#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
import matplotlib.pyplot as plt
def nested(cfg,path):
 v=cfg
 for p in path.split("."):v=v[p]
 return v
def main():
 p=argparse.ArgumentParser();p.add_argument("--runs",nargs="+",required=True);p.add_argument("--vary",nargs="+",required=True);p.add_argument("--output",required=True);a=p.parse_args();runs=[Path(x).resolve() for x in a.runs];cfgs=[json.loads((r/"resolved_config.json").read_text()) for r in runs];summaries=[json.loads((r/"summary.json").read_text()) for r in runs]
 ignore=set(a.vary)|{"experiment.name"};flat=lambda d,p="":{(p+"."+k if p else k):v for k,x in d.items() for v in ([x] if not isinstance(x,dict) else [])}|{k:v for key,x in d.items() if isinstance(x,dict) for k,v in flat(x,p+"."+key if p else key).items()}
 base=flat(cfgs[0]);differences=[]
 for i,c in enumerate(cfgs[1:],1):
  f=flat(c)
  for key in sorted(set(base)|set(f)):
   if key in ignore or key.startswith("derived.recipe_path") or key=="fingerprint":continue
   if base.get(key)!=f.get(key):differences.append({"run":str(runs[i]),"field":key,"base":base.get(key),"other":f.get(key)})
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/"differences.json").write_text(json.dumps(differences,indent=2)+"\n")
 if differences:raise SystemExit("scientific conditions differ; see differences.json")
 rows=[{"run":str(r),"status":s["status"],"completed_updates":s["completed_updates"],"processed_target_tokens":s["processed_target_tokens"],**{v:nested(c,v) for v in a.vary}} for r,s,c in zip(runs,summaries,cfgs)]
 with (out/"comparison.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 (out/"comparison.md").write_text("|"+"|".join(rows[0])+"|\n|"+"|".join(["---"]*len(rows[0]))+"|\n"+"\n".join("|"+"|".join(str(row[k]) for k in row)+"|" for row in rows)+"\n")
 for xkey,name in (("processed_target_tokens","loss_vs_tokens.png"),("elapsed_seconds","loss_vs_time.png")):
  plt.figure()
  for run in runs:
   events=[json.loads(line) for line in (run/"metrics.jsonl").read_text().splitlines()]; elapsed=0.0;points={}
   for e in events:
    elapsed += e.get("elapsed_seconds",0.0)
    if e["event_type"]=="eval" and e.get("split")=="validation": points[e["completed_updates"]]={**e,"_cumulative_elapsed":elapsed}
   ordered=[points[k] for k in sorted(points)]; xs=[e["processed_target_tokens"] if xkey=="processed_target_tokens" else e["_cumulative_elapsed"] for e in ordered]; plt.plot(xs,[e["nll"] for e in ordered],marker="o",label=run.name)
  plt.xlabel(xkey);plt.ylabel("validation NLL (nats/token)");plt.legend();plt.tight_layout();plt.savefig(out/name);plt.close()
 print(out)
if __name__=="__main__":main()
