import json,subprocess,sys
from pathlib import Path
from reporting import paired_trajectory_summary


def test_paired_trajectory_landmarks_sign_and_missing_events():
    base = {"experiment": {"seed": 0}, "optimizer": {"state_simulation": "none"}}
    treatment = {"experiment": {"seed": 0}, "optimizer": {"state_simulation": "low_precision"}}
    control_row = {"events": {"eval": [{"completed_updates": 10, "nll": 2.0}, {"completed_updates": 20, "nll": 1.5}]}}
    treatment_row = {"events": {"eval": [{"completed_updates": 10, "nll": 1.75}, {"completed_updates": 30, "nll": 1.0}]}}
    result = paired_trajectory_summary([control_row, treatment_row], [base, treatment], {
        "group_by": ["experiment.seed"], "treatment_field": "optimizer.state_simulation",
        "control_value": "none", "treatment_values": ["low_precision"], "landmark_updates": [10, 20, 30]})
    points = result["pairs"][0]["landmarks"]
    assert points[0]["paired_delta_validation_nll"] == -0.25
    assert points[1]["paired_delta_validation_nll"] is None
    assert points[2]["paired_delta_validation_nll"] is None
def test_compare_refuses_mismatch(tmp_path):
    runs=[]
    for i in range(2):
        r=tmp_path/f"r{i}";r.mkdir();cfg=json.loads(Path("recipes/diagnostic_cpu.json").read_text());cfg["model"]["vocab_size"]+=i;(r/"resolved_config.json").write_text(json.dumps(cfg));(r/"summary.json").write_text(json.dumps({"status":"completed","completed_updates":1,"processed_target_tokens":512}));runs.append(r)
    p=subprocess.run([sys.executable,"scripts/compare_runs.py","--runs",*map(str,runs),"--vary","optimizer.name","--output",str(tmp_path/"out")],capture_output=True,text=True);assert p.returncode!=0;diff=json.loads((tmp_path/"out/differences.json").read_text());assert any(x["field"]=="model.vocab_size" for x in diff)
