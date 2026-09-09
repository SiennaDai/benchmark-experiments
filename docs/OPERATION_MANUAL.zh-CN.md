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

## 2. 选择运行设备

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

## 3. 准备数据

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

## 4. CPU 联调与训练

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

## 5. 续训与评估

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

## 6. 低精度实验

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

## 7. 容量检查和长配方

124M 配方先做 dry-run，不自动启动长训练：

```bash
.venv/bin/python scripts/profile_run.py --recipe recipes/reference124m_pilot.json --max-wall-seconds 900 --output reports/profile124m --to-device cuda:0
.venv/bin/python src/main.py --recipe recipes/reference124m_1b.json --dry-run --to-device cuda:0
```

`profile_run.py` 当前只保存容量计划和 dry-run 结果；不会自动启动 124M 或 1B-token 作业。

## 8. 查看结果和排错

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

## 9. 验收

```bash
.venv/bin/python -m pytest tests/
```

当前 CPU 验收范围和未执行的 GPU 项见 `docs/VALIDATION.md`。
