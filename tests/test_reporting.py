import json
from pathlib import Path
from reporting import summarize_run


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
