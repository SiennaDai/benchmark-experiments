"""Optimizer construction with a lazy bitsandbytes boundary."""

import torch
from .adamw_reference import ReferenceAdamW


BNB_DEFAULTS = {"amsgrad": False, "percentile_clipping": 100, "min_8bit_size": 4096, "block_wise": True, "is_paged": False}


def make_optimizer(name, groups, cfg):
    common = dict(lr=cfg["lr"], betas=tuple(cfg["betas"]), eps=cfg["eps"], weight_decay=cfg["weight_decay"])
    if name == "torch_adamw":
        return torch.optim.AdamW(groups, **common, fused=cfg["fused"], foreach=cfg["foreach"])
    if name == "reference_adamw":
        return ReferenceAdamW(groups, **common, state_simulation=cfg["state_simulation"])
    if name in {"bnb_adamw32", "bnb_adamw8"}:
        if not torch.cuda.is_available():
            raise RuntimeError(f"{name} requires a supported CUDA device")
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError("bitsandbytes is not installed; install the optional GPU requirements") from exc
        cls = bnb.optim.AdamW32bit if name.endswith("32") else bnb.optim.AdamW8bit
        return cls(groups, **common, **BNB_DEFAULTS)
    raise ValueError(f"unknown optimizer {name}")


def optimizer_state_summary(optimizer) -> dict:
    tensors, storages, logical = [], set(), 0
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                size = value.numel() * value.element_size()
                logical += size
                storage = value.untyped_storage()
                identity = (storage.data_ptr(), storage.nbytes())
                storages.add(identity)
                tensors.append({"key": key, "dtype": str(value.dtype), "shape": list(value.shape), "bytes": size})
    return {"logical_tensor_bytes": logical, "unique_storage_bytes": sum(x[1] for x in storages), "tensors": tensors}
