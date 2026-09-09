import pytest, torch

@pytest.mark.skipif(not torch.cuda.is_available(), reason="T09 requires a supported CUDA GPU and bitsandbytes")
def test_bnb_gpu_probe():
    pytest.importorskip("bitsandbytes")
