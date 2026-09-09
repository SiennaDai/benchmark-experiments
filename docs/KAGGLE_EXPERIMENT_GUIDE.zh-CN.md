# Kaggle 实验通用操作指南

本文档记录在 Kaggle 上运行本项目 benchmark 的固定流程。它描述的是可复用的
实验协议，不绑定某一个模型、数据集、GPU、recipe 或 benchmark suite；带尖括号
的内容需要替换为本次实验的实际值。

## 流程概览

每次 Kaggle session 按以下顺序执行：

```text
确认环境 → 挂载冻结数据 → checkout 不可变代码版本 → 建立 DATA_ROOT
→ 验证数据身份 → 确认 suite → preflight → 运行 suite → 保存并检查报告
```

其中，session 级流程每次新建或恢复 Kaggle runtime 都要重新执行；suite 级流程
在每个 benchmark suite 上执行一次。

## 一、每个 Kaggle session 固定流程

### 1. 新建干净 Notebook 并确认环境

建议使用能表达实验协议的 Notebook 名称，例如：

```text
<project>-<protocol>-<version>
```

在 Kaggle 设置中选择本实验支持的 accelerator，并确认实际使用的设备。即使
选择了多 GPU，本项目当前运行入口也应明确指定一个 GPU；不要因为设备数量变化
而隐式启用 DDP。

```bash
!nvidia-smi
```

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
```

预期的 GPU 型号和 capability 以本次 recipe / protocol 的资源要求为准。若 CUDA
不可用、GPU 型号不符合要求，或 BF16 等能力不满足，应在训练前停止并记录原因。

### 2. 挂载冻结 dataset

通过 Kaggle 的 **Add Input** 加入本次实验批准的数据集：

```text
<frozen-dataset-name>
```

检查挂载内容：

```bash
!find /kaggle/input/<dataset-directory> -maxdepth 3 -type f -print
```

必须能找到 recipe 所引用的 manifest 以及对应 token 文件，例如：

```text
/kaggle/input/<dataset-directory>/data/<dataset-name>/manifest.json
```

正式实验使用冻结的 Input 数据，不要从 `/kaggle/working` 重新下载、修改或生成
实验数据。

### 3. checkout canonical repository

从 canonical repository clone 项目，并 checkout 本次实验批准的不可变 commit
或 tag：

```python
import os
import subprocess

REPO = "<canonical-repository-url>"
CHECKOUT = "/kaggle/working/<repository-directory>"
GIT_REF = "<commit-sha-or-tag>"

if not os.path.isdir(os.path.join(CHECKOUT, ".git")):
    subprocess.run(["git", "clone", REPO, CHECKOUT], check=True)

subprocess.run(["git", "-C", CHECKOUT, "fetch", "--tags", "origin"], check=True)
subprocess.run(["git", "-C", CHECKOUT, "checkout", "--detach", GIT_REF], check=True)
subprocess.run(["git", "-C", CHECKOUT, "status", "--short"], check=True)
subprocess.run(["git", "-C", CHECKOUT, "rev-parse", "HEAD"], check=True)
```

正式实验不要把会继续变化的 `main` 作为唯一版本标识。输出中的实际 commit
SHA 应记录在实验报告和运行 artifact 中。Notebook 可以使用 `main` 做开发联调，
但正式 benchmark 必须固定代码版本。

```bash
%cd /kaggle/working/<repository-directory>
```

### 4. 建立统一 DATA_ROOT

将 Kaggle Input 映射为项目预期的相对目录。以下示例假设 recipe 引用：

```text
data/<dataset-name>/manifest.json
```

```bash
!mkdir -p /kaggle/working/<data-mount>/data
!ln -sfn \
  /kaggle/input/<dataset-directory>/data/<dataset-name> \
  /kaggle/working/<data-mount>/data/<dataset-name>
!ls -lh /kaggle/working/<data-mount>/data/<dataset-name>
```

后续所有 suite 命令统一使用：

```text
DATA_ROOT=/kaggle/working/<data-mount>
```

这样只改变运行时挂载路径，不改变 recipe 中的 scientific data path，也不会改变
科学 fingerprint。

### 5. 验证冻结数据身份

读取 manifest，确认 protocol、数据 fingerprint、source revision、split 数量和
SHA-256 与实验记录一致：

```python
import json

manifest_path = "/kaggle/working/<data-mount>/data/<dataset-name>/manifest.json"
with open(manifest_path) as f:
    manifest = json.load(f)

print("protocol:", manifest["protocol_id"])
print("fingerprint:", manifest["fingerprint"])
print("source revision:", manifest["source"]["revision"])
for split_name, split in manifest["splits"].items():
    print(split_name, split["tokens"], split["sha256"])
```

若 manifest fingerprint、source revision 或 split hash 与批准版本不一致，应停止
实验。不能用手工修改 manifest 的方式绕过检查。

## 二、每个 benchmark suite 固定流程

### 1. 确认 suite 定义

检查 suite 文件和其中列出的完整 recipes：

```bash
!ls -lh benchmarks
!cat benchmarks/<suite-name>.json
```

重点确认：

- suite 中的 run 数量和执行顺序正确；
- 每个 run 指向预期的完整 recipe；
- `vary` 只列出本次实验允许变化的 scientific field；
- 其他 scientific fields、数据 protocol、seed 和评估设置应保持一致；
- runtime 中的 device 和 wall-time budget 符合本次 session 计划。

suite 是实验比较关系的声明，不是临时命令集合。不要在 Notebook 中临时改写
recipe 来代替 suite 定义。

### 2. 运行 suite preflight

正式训练前必须先运行：

```bash
python scripts/run_benchmark.py \
  --suite benchmarks/<suite-name>.json \
  --data-root /kaggle/working/<data-mount> \
  --output-root /kaggle/working/<runs-directory> \
  --preflight-only
```

preflight 应覆盖 suite comparability、recipe validity、设备与资源、数据容量、
manifest fingerprint 以及项目已有的 scientific guards。只有整体 PASS 才能开始
训练；不要只根据单个 recipe 的 dry-run 结果启动 suite。

### 3. 用同一条命令运行整个 suite

preflight 通过后，启动正式实验：

```bash
python scripts/run_benchmark.py \
  --suite benchmarks/<suite-name>.json \
  --data-root /kaggle/working/<data-mount> \
  --output-root /kaggle/working/<runs-directory>
```

suite runner 会按 suite 声明的顺序处理所有 run：新 run 启动，已完成 run 跳过，
可恢复的暂停或中断 run 从兼容的 latest checkpoint 继续，最后生成汇总和比较报告。
因此不要把同一个 suite 拆成多条手工 optimizer 命令运行。

运行中除非出现明确异常，否则等待 suite 完成。需要中止并检查的情况包括：

- exception 或 suite error；
- NaN/Inf；
- OOM；
- 设备或 precision 与预期不符；
- 不符合预期的暂停或 checkpoint 错误。

### 4. Kaggle wall-time 不足时的恢复

如果运行因墙钟预算变为 `paused_budget` 或发生可恢复中断：

1. 保存当前 Notebook version，并按本节重新建立 session；
2. 挂载同一个冻结 dataset；
3. checkout 同一个 commit/tag；
4. 恢复或重新挂载 run artifacts；
5. 再次执行完全相同的 suite 命令。

runner 会根据 run artifact 和 `checkpoints/latest.pt` 自动判断是否可以 resume。
只有 recipe fingerprint、data-manifest fingerprint、source commit 及 checkpoint
均兼容时才允许恢复。不要手工指定一个来自其他 recipe 或其他数据版本的 checkpoint。

若某个 run 已经 completed，同一条 suite 命令应跳过它，只处理尚未完成且可恢复的
run。

## 三、完成后的检查与交付

训练结束后，使用 runner 最后打印的 report 路径检查汇总：

```bash
!cat /kaggle/working/<report-directory>/benchmark_summary.md
```

至少保存以下信息：

- suite 名称和 suite resolved definition；
- repository commit SHA；
- Kaggle accelerator、PyTorch/CUDA 环境；
- frozen dataset 名称、manifest fingerprint 和 source revision；
- 每个 run 的最终状态（completed、paused 或 failed）；
- `benchmark_summary.md`、comparison 表格和必要的 figures；
- 跨 session 恢复所需的 checkpoint artifact。

完成后再分析 loss trajectory、最终或最佳评估指标、吞吐、显存和 optimizer-state
memory。若 run 未全部 completed，应明确标记为不完整结果，不要把 paused 或 failed
run 当作正式比较结论。

## 四、最小检查清单

### Session 级

- [ ] accelerator、CUDA、GPU 和 capability 已确认；
- [ ] 挂载的是批准的 frozen dataset；
- [ ] checkout 的是批准的不可变 commit/tag；
- [ ] `DATA_ROOT` 下 manifest 路径与 recipe 一致；
- [ ] manifest fingerprint 和 split hash 已核对。

### Suite 级

- [ ] suite 的 run、`vary` 和 recipes 已核对；
- [ ] `--preflight-only` 整体通过；
- [ ] 使用统一 suite 命令启动，而不是手工拆分 run；
- [ ] 暂停时保留 artifacts，并使用相同命令恢复；
- [ ] 完成后保存 report、commit、数据身份和运行状态。

## 五、每个 benchmark suite 完成后的归档打包

Kaggle 的 `/kaggle/working` 会随 session 结束而消失，因此正式实验不能只依赖
Notebook Save Version。suite 完成并生成报告后，应立即把原始运行 artifacts 和
比较报告打成一个压缩包，再保存 Notebook version，并将压缩包下载或复制到可靠
的外部位置。

### 1. 确认 suite 已完成并生成报告

先确认所有预期 run 都已完成，并确认报告目录存在。打包前不要把 `paused` 或
`failed` run 当作完整结果归档；如果确实需要保存中间状态，应在归档名称或交付
记录中明确标注其状态。

### 2. 创建 suite 归档

归档应包含两类内容：

- `runs/` 下本 suite 的原始运行 artifacts，包括 checkpoint、metrics、summary、
  resolved config 和运行元数据；
- `reports/<suite-name>/` 下的比较结果，包括 Markdown、CSV 和图。

使用固定命名规则：

```text
<suite-name>.tar.gz
```

例如，在 Kaggle Notebook 中执行：

```bash
!tar -czf /kaggle/working/<suite-name>.tar.gz \
  -C /kaggle/working/<repository-directory> \
  reports/<suite-name> \
  -C /kaggle/working \
  runs/<run-id-1> \
  runs/<run-id-2>
```

将命令中的 run 列表替换为 suite JSON 中声明的全部 run。不要把其他 suite 的
run 混入归档，也不要用宽泛的 glob 把无关实验一起打包。

### 3. 验证归档内容

先确认压缩包存在且大小合理：

```bash
!ls -lh /kaggle/working/<suite-name>.tar.gz
```

再列出压缩包目录，检查报告和每个 run 的关键文件：

```bash
!tar -tzf /kaggle/working/<suite-name>.tar.gz | head -100
```

至少应能看到类似以下结构：

```text
reports/<suite-name>/benchmark_summary.md
reports/<suite-name>/comparison.csv
reports/<suite-name>/comparison.md
reports/<suite-name>/<figure-file>
runs/<run-id-1>/summary.json
runs/<run-id-1>/metrics.jsonl
runs/<run-id-2>/summary.json
```

如果报告目录、某个 run 的 summary/metrics 或需要用于恢复的 checkpoint 缺失，先
修正归档命令并重新验证，不要把不完整压缩包作为最终交付物。

### 4. 固定收尾顺序

每个 suite 的标准收尾流程是：

```text
suite completed
→ report generated
→ tar.gz archive
→ verify archive contents
→ Save Version
→ download/copy tar.gz to local or durable storage
```

归档完成后，将压缩包名称、suite 名称、repository commit、数据 fingerprint、
归档时间和各 run 状态一并记录。这样即使 Kaggle session 已结束，也能根据压缩包
恢复实验上下文和必要的 checkpoint。

更详细的 suite schema、可比性规则和 artifact 语义见
[`BENCHMARK_SUITES.md`](BENCHMARK_SUITES.md)；Kaggle launcher 的项目级参数说明见
[`OPERATION_MANUAL.zh-CN.md`](OPERATION_MANUAL.zh-CN.md)。
