import json
from pathlib import Path
import pytest
from reporting import replication_summary, summarize_run


def write_run(path, events, status="completed"):
    path.mkdir(); cfg={"experiment":{"name":path.name,"protocol_id":"p","seed":1,"data_seed":2,"algorithm_seed":3},"optimizer":{"name":"torch_adamw"},"model":{"sequence_length":4},"train":{"target_tokens":16},"precision":{"compute":"fp32"},"derived":{"total_updates":4},"fingerprint":"recipe"}
    (path/"resolved_config.json").write_text(json.dumps(cfg)); (path/"summary.json").write_text(json.dumps({"status":status,"completed_updates":4,"processed_target_tokens":16,"total_updates":4,"elapsed_seconds":9.0})); (path/"data_manifest.json").write_text(json.dumps({"fingerprint":"data"})); (path/"source.json").write_text(json.dumps({"upstream_commit":"commit"})); (path/"precision.json").write_text(json.dumps({"compute":"fp32"})); (path/"metrics.jsonl").write_text("\n".join(json.dumps(e) for e in events)+"\n")


def test_summary_deduplicates_resumed_points_and_uses_update_time(tmp_path):
    base={"run_id":"r","schema_version":1}
    events=[{**base,"event_type":"eval","completed_updates":0,"processed_target_tokens":0,"segment_id":0,"event_index":0,"split":"validation","nll":5.0,"elapsed_seconds":1.0}, {**base,"event_type":"train","completed_updates":1,"processed_target_tokens":4,"segment_id":0,"event_index":1,"elapsed_seconds":2.0,"clipped":False}, {**base,"event_type":"eval","completed_updates":4,"processed_target_tokens":16,"segment_id":0,"event_index":2,"split":"validation","nll":4.0,"elapsed_seconds":1.0}, {**base,"event_type":"eval","completed_updates":4,"processed_target_tokens":16,"segment_id":1,"event_index":0,"split":"validation","nll":3.0,"elapsed_seconds":1.0}, {**base,"event_type":"resource","completed_updates":4,"processed_target_tokens":16,"segment_id":1,"event_index":1,"optimizer_state":{"unique_storage_bytes":12,"tensors":[{}]}}]
    run=tmp_path/"r"; write_run(run,events)
    summary=summarize_run(run)
    assert summary["initial_validation_nll"] == 5.0 and summary["final_validation_nll"] == 3.0 and summary["best_validation_update"] == 4
    assert summary["train_elapsed_seconds"] == 2.0 and summary["tokens_per_second"] == 8.0 and summary["optimizer_state_bytes"] == 12


def test_missing_metrics_are_null(tmp_path):
    run=tmp_path/"r"; write_run(run,[] ,status="paused_budget")
    summary=summarize_run(run)
    assert summary["initial_validation_nll"] is None and summary["median_update_seconds"] is None


def test_replication_summary_pairs_by_declared_fields_and_sample_std():
    rows = [
        {"status": "completed", "final_validation_nll": 4.0, "best_validation_nll": 3.5},
        {"status": "completed", "final_validation_nll": 4.2, "best_validation_nll": 3.7},
        {"status": "completed", "final_validation_nll": 5.0, "best_validation_nll": 4.5},
        {"status": "completed", "final_validation_nll": 5.5, "best_validation_nll": 4.8},
    ]
    configs = [{"experiment": {"seed": seed, "data_seed": seed + 100}, "optimizer": {"state_simulation": treatment}}
               for seed, treatment in [(0, "none"), (0, "bf16_roundtrip"), (1, "none"), (1, "bf16_roundtrip")]]
    result = replication_summary(rows, configs, {"group_by": ["experiment.seed", "experiment.data_seed"],
        "treatment_field": "optimizer.state_simulation", "control_value": "none", "treatment_values": ["bf16_roundtrip"]})
    assert result["aggregates"]["bf16_roundtrip"]["n"] == 2
    assert result["aggregates"]["bf16_roundtrip"]["final_validation_nll"]["mean"] == 4.85
    assert result["aggregates"]["bf16_roundtrip"]["final_validation_nll"]["sample_std"] == pytest.approx(0.9192388155)
    assert [p["delta_final_validation_nll"] for p in result["pairs"]] == [pytest.approx(0.2), pytest.approx(0.5)]


def test_replication_summary_missing_pair_and_singleton_std():
    rows = [{"status": "completed", "final_validation_nll": 4.0, "best_validation_nll": 4.0}]
    configs = [{"experiment": {"seed": 0, "data_seed": 1}, "optimizer": {"state_simulation": "none"}}]
    result = replication_summary(rows, configs, {"group_by": ["experiment.seed", "experiment.data_seed"],
        "treatment_field": "optimizer.state_simulation", "control_value": "none", "treatment_values": ["bf16_roundtrip"]})
    assert result["aggregates"]["none"]["n"] == 1
    assert result["aggregates"]["none"]["final_validation_nll"]["sample_std"] is None
    assert result["aggregates"]["bf16_roundtrip"]["paired_delta_final_validation_nll"]["n"] == 0
