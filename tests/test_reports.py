import json,subprocess,sys
from pathlib import Path
def test_compare_refuses_mismatch(tmp_path):
    runs=[]
    for i in range(2):
        r=tmp_path/f"r{i}";r.mkdir();cfg=json.loads(Path("recipes/diagnostic_cpu.json").read_text());cfg["model"]["vocab_size"]+=i;(r/"resolved_config.json").write_text(json.dumps(cfg));(r/"summary.json").write_text(json.dumps({"status":"completed","completed_updates":1,"processed_target_tokens":512}));runs.append(r)
    p=subprocess.run([sys.executable,"scripts/compare_runs.py","--runs",*map(str,runs),"--vary","optimizer.name","--output",str(tmp_path/"out")],capture_output=True,text=True);assert p.returncode!=0;diff=json.loads((tmp_path/"out/differences.json").read_text());assert any(x["field"]=="model.vocab_size" for x in diff)
