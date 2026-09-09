# Validation record

Environment: WSL2 Linux, Python 3.12.3, 4-core/8-thread Intel i7-8565U class CPU, 3.7 GiB RAM, PyTorch 2.8.0+cpu, NumPy 2.3.2. CUDA unavailable. Raw test XML is `artifacts/validation/pytest-final.xml`; run artifacts are under ignored `runs/` and reports under ignored `reports/`.

| ID | Result | Evidence |
|---|---|---|
| T01 recipe | pass | duplicate/unknown fields, divisibility, FFN/head and precision rejection; deterministic resolve |
| T02 data | pass | shift/window/tail, sampler restoration, tamper detection |
| T03 model | pass | causal logits, CE equality, tied parameter deduplication |
| T04 AdamW | pass | grad-none/zero/decay/step semantics and torch multi-step tolerance |
| T05 accumulation | pass | batch vs four microbatches gradient norm/parameters |
| T06 schedule | pass | warmup, decay endpoint, W=0 and N=1 |
| T07 resume | pass | unit state/RNG path plus real paused run; continuous/resumed final model and sampler are bitwise equal; 100 unique train events in segments 0/1 |
| T08 lowp simulation | pass | round timing, FP32 persisted storage and exact disabled path; `reports/optimizer_diagnostics/diagnostics.json` |
| T09 bitsandbytes | skip | no CUDA device; bitsandbytes deliberately not installed |
| T10 reports | pass | incompatible scientific configuration rejected with differences artifact; successful report emits CSV/Markdown/two PNGs |
| T11 upstream parity | partial | deterministic local fixed-state forward covered; pristine independent-process update parity not completed |
| T12 overfit | pass | `runs/overfit_cpu_verified`: validation NLL 5.63190985 to 1.117587e-8 in 500 updates; LR=0 control has bitwise-equal model and identical NLL |
| T13 real corpus | pass | `scripts/prepare_data.py` plan-only and bounded materialization completed at revision `b5f90f419b7489cdba26fdbc8c022fcb5562f968`; selected 338,924,709 bytes; frozen manifest has train 8,388,846, validation 68,296, test 68,602 tokens; `runs/mini_cpu_smoke_real` completed 10/10 updates and 2,560 target tokens |
| Runtime device selection | pass on CPU | `auto` and `cpu` resolve correctly; invalid values and unavailable `cuda:0` are rejected; training and evaluation both report `device=cpu`; CUDA execution remains unverified |
| Preflight data capacity | pass | Manifest-backed train/validation window capacity is checked before model initialization; exact boundary and overflow, repeated-epoch isolation, and the 16384-token diagnostic regression are covered |
| Wall-clock pause checkpoint | pass by code path | `paused_budget` now saves `latest.pt` even at update 0 and always at a non-interval update boundary; checkpoint includes optimizer, scheduler-equivalent state, sampler, RNG, and processed-token counters |

Fresh suite command: `.venv/bin/python -m pytest --junitxml=artifacts/validation/pytest-to-device.xml` — 22 passed, 1 skipped in 5.18 s. Additional dry-run checks return exit 2 with an explicit unsupported reason for unavailable CUDA/BF16; FP32 diagnostic dry-run with `--to-device cpu` returns 0.

CPU run evidence:

- `runs/diagnostic_cpu_verified`: 100/100 updates, 51,200 target tokens, 12.384 s; validation NLL 5.58637850 to 5.55129068.
- `runs/diagnostic_state_sim_verified`: 100/100, 51,200 tokens, 9.932 s; validation NLL 5.58637850 to 5.55129037. Timing is not interpreted as a bit-width speed result.
- `runs/diagnostic_resume_verified`: paused at update 5 under 0.5 s boundary, resumed to 100; bitwise-equal final weights to continuous run.
- `runs/overfit_cpu_lr0_verified`: 10 updates; initial/final weights and validation NLL are exactly equal.
- `data/slimpajama_small/source_plan.json` and `manifest.json`: bounded source selection and token hashes for the downloaded subset.
- `reports/comparison_final_valid/`: explicit non-ranking comparison varying optimizer and protocol fields; CSV/Markdown plus both loss plots.
- `reports/profile124m/`: dry-run-only capacity probe; no CUDA training launched.
- `runs/to_device_cpu_verified`: explicit CPU device run, 10/10 updates; `precision.json` records both `to_device=cpu` and `device=cpu`.

GPU BF16, bnb32/bnb8 state coverage/recovery, real single-GPU training, CUDA memory/timing, 124M capacity, and 1B-token experiments are not run. The 124M and 1B recipes are only configuration artifacts.
