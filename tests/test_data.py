import json
import numpy as np
import pytest
from data.frozen_tokens import DeterministicSampler, FrozenWindows, load_manifest, sha256_file

def make_manifest(tmp_path, values):
    splits={}
    for name in ("train","validation","test"):
        p=tmp_path/f"{name}.bin"; np.asarray(values,dtype="<u2").tofile(p); splits[name]={"file":p.name,"tokens":len(values),"sha256":sha256_file(p)}
    m={"schema_version":1,"fingerprint":"x","dtype":"uint16","endianness":"little","max_token_id":max(values),"splits":splits}; p=tmp_path/"manifest.json"; p.write_text(json.dumps(m)); return p

def test_windows_shift_no_target_overlap_tail(tmp_path):
    w=FrozenWindows(load_manifest(make_manifest(tmp_path,range(12))),"train",4); x0,y0=w.window(0); x1,y1=w.window(1)
    assert w.num_windows==2 and w.dropped_tail_tokens==3 and x0.tolist()==[0,1,2,3] and y0.tolist()==[1,2,3,4] and set(y0.tolist()).isdisjoint(set(y1.tolist()))

def test_hash_tamper_detected(tmp_path):
    p=make_manifest(tmp_path,range(12)); (tmp_path/"train.bin").write_bytes(b"bad")
    with pytest.raises(ValueError,match="hash mismatch"): load_manifest(p)

def test_sampler_resume_and_grouping():
    a=DeterministicSampler(17,3,True); first=a.take(8); state=a.state_dict(); rest=a.take(12); b=DeterministicSampler(17,3,True); b.load_state_dict(state)
    assert b.take(12)==rest and first==DeterministicSampler(17,3,True).take(8)
