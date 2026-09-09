# 低精度优化器预训练实验平台操作手册

本手册中的命令均从项目根目录运行：

```bash
cd /home/sienna/projects/experiments/lowp-optimizer-bench
```

## 1. 创建环境

CPU 环境：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python numpy==2.3.2 tiktoken==0.11.0 matplotlib==3.10.5 pytest==8.4.1
```

准备 SlimPajama 还需要：

```bash
uv pip install --python .venv/bin/python huggingface-hub==0.34.4 pyarrow==21.0.0
```

CUDA 主机应安装与驱动匹配的 CUDA 版 PyTorch。运行 bitsandbytes 配方时再安装 `bitsandbytes==0.48.1`。

本地开发建议使用 Git 管理可复现实验版本：先在编辑器中修改代码或
recipe，运行测试和 dry-run，然后提交并推送；Kaggle 只从仓库 clone 并运行
固定的 commit/tag，不在 Notebook 中长期修改训练代码。

```bash
git status
git add src scripts recipes docs README.md
git commit -m "Describe the experiment change"
git push origin main
```

## 2. 本地运行

以下命令均在仓库根目录执行。先准备数据；离线联调可直接使用 synthetic
数据。recipe 中的相对 `data.manifest` 会相对于 `DATA_ROOT` 解析，默认是仓库根目录。

```bash
.venv/bin/python scripts/prepare_data.py --kind synthetic --output data/diagnostic
DATA_ROOT="$PWD" OUTPUT_ROOT="$PWD/runs" DEVICE=cpu \
  bash scripts/run_local.sh recipes/diagnostic_cpu.json diagnostic_01
```

启动前会检查配置、设备、manifest 内容以及固定窗口容量，不训练。validation/test 评估不会因为
`allow_repeated_epochs` 而重复窗口；容量不足会以非零退出码明确拒绝，并提示扩大数据 split
或降低 recipe 的 eval budget：

```bash
DATA_ROOT="$PWD" DEVICE=cpu DRY_RUN_ONLY=1 \
  bash scripts/run_local.sh recipes/diagnostic_cpu.json diagnostic_01
```

也可以直接调用 Python 入口。新运行的输出目录是
`OUTPUT_ROOT/<RUN_ID>`；如果未设置 `OUTPUT_ROOT`，launcher 使用仓库下的
`runs/`。`RUN_ID` 只能包含字母、数字、`.`, `_`, `-`，新运行目录必须为空。

```bash
.venv/bin/python src/main.py --recipe recipes/diagnostic_cpu.json \
  --data-root "$PWD" --run-dir runs/diagnostic_01 --to-device cpu
```

墙钟限制只在完整 update 边界生效，状态会是 `paused_budget`，可用最新 checkpoint
继续：

```bash
MAX_WALL_SECONDS=1800 \
  bash scripts/run_local.sh recipes/diagnostic_cpu.json diagnostic_01
RESUME_CHECKPOINT=runs/diagnostic_01/checkpoints/latest.pt \
  bash scripts/run_local.sh recipes/diagnostic_cpu.json diagnostic_01
```

## 3. Kaggle 运行

完整的、与具体实验无关的 Kaggle session 和 benchmark suite 固定流程见
[`KAGGLE_EXPERIMENT_GUIDE.zh-CN.md`](KAGGLE_EXPERIMENT_GUIDE.zh-CN.md)。本节保留
launcher 的项目级参数和路径说明。

在 Kaggle Notebook 设置中打开 GPU 和 Internet，使用仓库中的
`notebooks/kaggle_launcher.ipynb`。Notebook 会 clone
`https://github.com/SiennaDai/benchmark-experiments.git`，再 checkout
`GIT_REF`。正式实验应把 `GIT_REF` 改成已推送的不可变 commit SHA 或 tag，
不要使用会漂移的 `main`。

在 Kaggle Dataset 中挂载 token 数据。recipe 例如写着
`data/slimpajama_small/manifest.json` 时，应使文件实际位于：

```text
/kaggle/input/<dataset>/data/slimpajama_small/manifest.json
```

因此 launcher 通常使用 `DATA_ROOT=/kaggle/input`；如果 Dataset 根目录本身
就是 `data/`，则应把 `DATA_ROOT` 改成对应挂载目录。先执行 Kaggle preflight：

```bash
cd /kaggle/working/benchmark-experiments
DATA_ROOT=/kaggle/input DRY_RUN_ONLY=1 \
  bash scripts/run_kaggle.sh recipes/mini_bf16_adamw.json adamw_trial_01
```

训练一句话启动：

```bash
DATA_ROOT=/kaggle/input OUTPUT_ROOT=/kaggle/working/runs \
MAX_WALL_SECONDS=39600 \
  bash scripts/run_kaggle.sh recipes/mini_bf16_adamw.json adamw_trial_01
```

`run_kaggle.sh` 将 `KAGGLE_GPU_INDEX`（默认 `0`）映射为进程内的
`cuda:0`，严格检查 CUDA、BF16、manifest 和 bitsandbytes；第一版只支持
单进程单 GPU，不会自动启用 DDP、改成 FP16 或退回 CPU。Kaggle 镜像自带
CUDA 匹配的 PyTorch；只有 preflight 报缺少 add-on 时才设置
`KAGGLE_INSTALL_DEPS=1` 安装 `requirements/kaggle.txt`，该文件不会重新安装
Torch。

运行结果写在 `/kaggle/working/runs/<RUN_ID>`。Notebook runtime 结束前应
保存 Notebook version；需要跨 runtime 恢复的大 checkpoint 应发布为私有
Kaggle Dataset，再挂载并设置：

```bash
RESUME_CHECKPOINT=/kaggle/input/<resume-dataset>/run/checkpoints/latest.pt \
  bash scripts/run_kaggle.sh recipes/mini_bf16_adamw.json adamw_trial_01
```

续训必须使用相同 recipe、数据 manifest 和兼容环境；checkpoint 会拒绝科学
fingerprint 或数据 fingerprint 不一致的恢复。

## 4. Recipe 说明

Recipe 是完整 JSON 对象，不支持继承、Hydra `key=value` 覆盖或环境变量插值。
顶层必须且只能有以下 10 组；每组也必须且只能包含实现已声明的字段：

| 组 | 字段及用途 |
|---|---|
| `experiment` | `name` 实验名；`protocol_id` 数据/协议标识；`seed`、`data_seed`、`algorithm_seed` 三类随机种子 |
| `model` | `family`（当前为 `llama`）、层数/宽度/头数、`ffn_dim`/`multiple_of`、词表与序列长度、dropout/bias/embedding tying、初始化、RMSNorm 与 RoPE 参数 |
| `data` | `manifest`、`train_split`、`validation_split`、`allow_repeated_epochs` |
| `train` | `target_tokens`、`micro_batch_size`、`accumulation_steps`、`grad_clip_norm` |
| `optimizer` | `name`（`torch_adamw`、`reference_adamw`、`bnb_adamw32`、`bnb_adamw8`）、学习率/Betas/epsilon/weight decay、`fused`、`foreach`、`state_simulation` |
| `schedule` | `name`（`cosine` 或 `constant`）、`warmup_updates`、`final_lr_ratio` |
| `precision` | `compute`（`fp32`/`bf16`）、参数/梯度 dtype、`tf32`、attention backend、`deterministic`、`compile` |
| `eval` | 评估间隔、最大 token 数、batch size、compute 和 attention backend |
| `logging` | `every_updates`、`diagnostics`、`wandb`；当前平台主要写本地 JSONL，不会自动上传 W&B |
| `checkpoint` | 保存间隔以及 initial/final checkpoint 开关 |

`model.ffn_dim`、head dimension、`train.target_tokens` 与有效 batch 的整除关系
等一致性会在加载时检查。科学 fingerprint 覆盖整个 recipe；`--data-root`、
`--run-dir`、`--resume`、`--to-device`、`--max-wall-seconds` 是运行时参数，
不属于 recipe。数据挂载路径可变，但 manifest 内容 fingerprint 必须一致。

复制和命名 recipe 的推荐流程：

```bash
cp recipes/mini_bf16_adamw.json recipes/mini_bf16_adamw_trial02.json
# 编辑 experiment.name 以及明确要比较的科学字段
.venv/bin/python -m json.tool recipes/mini_bf16_adamw_trial02.json >/dev/null
.venv/bin/python src/main.py \
  --recipe recipes/mini_bf16_adamw_trial02.json \
  --data-root "$PWD" --dry-run --to-device cpu
```

最小可用示例见 `recipes/diagnostic_cpu.json`；单 GPU BF16 示例见
`recipes/mini_bf16_adamw.json`。不要手动删除字段或添加未声明字段；不要在
resume 时修改模型、数据、优化器、精度或总训练日程。

## 5. 选择运行设备

训练、dry-run 和独立评估支持 `--to-device`，同时接受等价拼写 `--to_device`：

| 值 | 含义 |
|---|---|
| `auto` | 默认值；有 CUDA 时选择 `cuda:0`，否则选择 CPU |
| `cpu` | 强制使用 CPU，即使机器有 GPU |
| `cuda` | 使用 `cuda:0`；CUDA 不可用时报错 |
| `cuda:N` | 使用指定编号的 CUDA 设备，例如 `cuda:1`；编号不存在时报错 |

设备是运行控制项，不改变 recipe 的科学配置。实际请求值和解析后的设备分别写入运行目录的 `precision.json`。BF16 和 bitsandbytes 后端要求 CUDA；不满足时 dry-run 返回退出码 2，并打印 `unsupported_reason`，不会回退到 FP16 或 CPU 模拟。

示例：

```bash
.venv/bin/python src/main.py --recipe recipes/diagnostic_cpu.json --dry-run --to-device cpu
.venv/bin/python src/main.py --recipe recipes/mini_bf16_adamw.json --dry-run --to-device cuda:0
.venv/bin/python src/main.py --recipe recipes/mini_bf16_bnb8.json --run-dir runs/bnb8_gpu0 --to-device cuda:0
```

## 6. 准备数据

离线人工数据：

```bash
.venv/bin/python scripts/prepare_data.py --kind synthetic --output data/diagnostic
.venv/bin/python scripts/prepare_data.py --kind synthetic --output data/overfit --overfit --train-tokens 65537 --validation-tokens 4097 --test-tokens 4097
```

先检查 SlimPajama 下载计划，再按 512 MiB 上限准备冻结子集：

```bash
.venv/bin/python scripts/prepare_data.py --kind slimpajama --output data/slimpajama_small --plan-only
.venv/bin/python scripts/prepare_data.py --kind slimpajama --output data/slimpajama_small --source-max-bytes 536870912
```

训练启动时会校验 manifest、token 文件长度和 SHA-256。不要手工修改已经用于实验的 token 文件或 manifest。

## 7. CPU 联调与训练

先检查配置、数据和设备：

```bash
.venv/bin/python src/main.py --recipe recipes/diagnostic_cpu.json --dry-run --to-device cpu
```

运行确定性诊断训练：

```bash
.venv/bin/python src/main.py --recipe recipes/diagnostic_cpu.json --run-dir runs/diagnostic_01 --to-device cpu
```

运行真实语料 CPU smoke：

```bash
.venv/bin/python src/main.py --recipe recipes/mini_cpu_smoke.json --run-dir runs/mini_cpu_smoke_01 --to-device cpu
```

设置墙钟上限只会在完整更新边界暂停：

```bash
.venv/bin/python src/main.py --recipe recipes/diagnostic_cpu.json --run-dir runs/bounded --max-wall-seconds 30 --to-device cpu
```

## 8. 续训与评估

严格续训使用原 recipe，不允许改变模型、数据、优化器、精度或总日程：

```bash
.venv/bin/python src/main.py --recipe recipes/diagnostic_cpu.json --resume runs/diagnostic_01/checkpoints/latest.pt --to-device cpu
```

独立验证或测试：

```bash
.venv/bin/python scripts/evaluate.py --checkpoint runs/diagnostic_01/checkpoints/final.pt --split validation --to-device cpu
.venv/bin/python scripts/evaluate.py --checkpoint runs/diagnostic_01/checkpoints/final.pt --split test --to-device cuda:0
```

第一版严格续训要求同类设备和相同环境。跨设备 checkpoint 仅用于显式独立评估，不承诺逐步轨迹一致。

## 9. 低精度实验

CPU 可运行 BF16 optimizer-state 持久化舍入模拟；模型计算和保存的状态张量仍是 FP32：

```bash
.venv/bin/python src/main.py --recipe recipes/diagnostic_state_sim.json --run-dir runs/state_sim_01 --to-device cpu
.venv/bin/python scripts/optimizer_diagnostics.py --recipe recipes/diagnostic_state_sim.json --output reports/optimizer_diagnostics
```

GPU 上按顺序验证 torch AdamW、bnb32、bnb8：

```bash
.venv/bin/python src/main.py --recipe recipes/mini_bf16_adamw.json --run-dir runs/mini_adamw_gpu --to-device cuda:0
.venv/bin/python src/main.py --recipe recipes/mini_bf16_bnb32.json --run-dir runs/mini_bnb32_gpu --to-device cuda:0
.venv/bin/python src/main.py --recipe recipes/mini_bf16_bnb8.json --run-dir runs/mini_bnb8_gpu --to-device cuda:0
```

先分别确认所有 run 都是 `completed`，再比较。不同 protocol ID 也必须显式列入变化字段：

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/compare_runs.py \
  --runs runs/mini_adamw_gpu runs/mini_bnb32_gpu runs/mini_bnb8_gpu \
  --vary optimizer.name experiment.protocol_id \
  --output reports/mini_gpu_comparison
```

## 10. 容量检查和长配方

124M 配方先做 dry-run，不自动启动长训练：

```bash
.venv/bin/python scripts/profile_run.py --recipe recipes/reference124m_pilot.json --max-wall-seconds 900 --output reports/profile124m --to-device cuda:0
.venv/bin/python src/main.py --recipe recipes/reference124m_1b.json --dry-run --to-device cuda:0
```

`profile_run.py` 当前只保存容量计划和 dry-run 结果；不会自动启动 124M 或 1B-token 作业。

## 11. 查看结果和排错

每个 run 的关键文件：

- `summary.json`：最终状态、完成更新数和 token 数；
- `metrics.jsonl`：逐更新训练、评估和资源事件；
- `resolved_config.json`：完整解析配置；
- `precision.json`：精度策略和实际设备；
- `checkpoints/latest.pt`、`final.pt`：恢复和最终评估 checkpoint。

常见错误：

- `CUDA is unavailable`：主机没有可用 CUDA，或 CUDA 版 PyTorch/驱动不匹配；
- `BF16 compute requires...`：设备不支持 CUDA BF16；不会自动切换到 FP16；
- `data manifest not found`：先执行对应数据准备命令；
- manifest/hash 错误：token 文件被修改或路径不匹配，应重新准备数据；
- 比较工具报告 scientific conditions differ：检查 `differences.json`，只把确实属于实验变量的字段加入 `--vary`。

## 12. 验收

```bash
.venv/bin/python -m pytest tests/
```

当前 CPU 验收范围和未执行的 GPU 项见 `docs/VALIDATION.md`。
