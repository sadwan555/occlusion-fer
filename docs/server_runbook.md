# HIVE 部署、E7 clean/mixed 正式实验与论文产物手册

## 1. 目的与研究边界

本手册用于在 HIVE 的 Linux + NVIDIA GPU 环境中验证并执行 FER2013 E7
clean/mixed 协议。当前唯一权威协议是
[`experiment_protocol.md`](experiment_protocol.md)；本手册中的 E0-E7 部分只保留
筛选过程的历史可追溯性。

当前状态：Stage 8 mixed 正式训练尚未开始，PrivateTest 尚未访问。三组 locked
E7 clean checkpoint 已保留；任何 screening、smoke、synthetic 或工程验证数值
都不是正式论文结果。

项目只预测 FER2013 定义的七个表情标签。不能把结果解释为识别真实内在情绪、
困惑、理解程度、参与度或学习效果，也不能据此宣称真实世界鲁棒性、跨数据集
泛化、创新性或最先进性能。

正式 E7 clean/mixed 协议在首次 PrivateTest 评估前锁定为：

- ImageNet 预训练、标准 stem 的 ResNet-18；
- 灰度图复制为三通道，bilinear resize 到 `224×224`，ImageNet normalization；
- 官方 `Training` 训练、`PublicTest` 验证、`PrivateTest` 最终测试；
- AdamW，learning rate `1e-4`，weight decay `0.001`；
- mild affine augmentation、label smoothing `0.1`；batch size `128`，epochs `50`，DataLoader workers `4`，CUDA AMP；
- seeds `42`、`123`、`2026`；
- 按 clean PublicTest macro-F1 选择 `best.pt`，相同分数保留更早 epoch；
- clean 与 mixed 的三个 seed 使用相同模型、处理、超参数和 checkpoint 规则；
- mixed 中每个 Training 样本按 sample ID、seed 和 one-based epoch 确定性选择：
  50% clean，50% 从九个批准条件中均匀选择；
- `occlusion-v2-224` PublicTest evaluator 固定使用 seed `20260804` 和同一个经验证
  manifest，masked 指标绝不参与 checkpoint 选择。

不要在查看 PrivateTest 结果后修改这些设置。任何后续 clean 与 mixed 对比也必须
使用同一套正式训练预算和选择规则。

## 2. 服务器目录与每次登录准备

所有内容使用本人 `${HOME}`，项目、数据、环境和输出互相隔离：

```bash
export FER_WORK_ROOT="${HOME}/anson-fer"
export FER_PROJECT_ROOT="${FER_WORK_ROOT}/project/occlusion-fer"
export FER_DATA_ROOT="${FER_WORK_ROOT}/data/raw"
export FER_DATA_CSV="${FER_DATA_ROOT}/fer2013.csv"
export FER_ENV_ROOT="${FER_WORK_ROOT}/envs/occlusion-fer"
export FER_OUTPUT_ROOT="${FER_WORK_ROOT}/outputs"
```

不要把这些真实路径写回 YAML。每次重新登录都重新设置变量并激活环境：

```bash
source "${FER_ENV_ROOT}/bin/activate"
cd "${FER_PROJECT_ROOT}"
```

## 3. 只读检查服务器与 Git

```bash
whoami
hostname
pwd
uname -a
nvidia-smi
df -h
nproc
python3 --version
git status --short --branch
git log --oneline --decorate -n 3
```

历史筛选和部署段落使用 `main` 作为初始 checkout；Stage 8 不使用该目录。
Stage 8 的唯一权威工作树和 `AGENTS.md` 是 integration worktree
`/home/ucla/anson-fer/project/occlusion-fer-stage-b`，其分支必须是
`stage-b/e7-occlusion-integration`。发现陌生 GPU 进程时先确认归属，不结束其他
用户的进程：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
ps -o user,pid,ppid,etime,cmd -p PID
```

## 4. 首次获取项目和创建环境

```bash
mkdir -p "${FER_WORK_ROOT}/project" "${FER_DATA_ROOT}" \
  "${FER_WORK_ROOT}/envs" "${FER_OUTPUT_ROOT}"
git clone git@github.com:sadwan555/occlusion-fer.git "${FER_PROJECT_ROOT}"
python3 -m venv "${FER_ENV_ROOT}"
source "${FER_ENV_ROOT}/bin/activate"
python -m pip install --upgrade pip
```

先根据 [PyTorch Start Locally](https://docs.pytorch.org/get-started/locally/)
安装与 HIVE 驱动匹配的 Linux CUDA wheel，再安装项目。项目依赖范围锁定为
`torch>=2.5,<2.6` 和 `torchvision>=0.20,<0.21`，与已验证的
`2.5.1+cu121 / 0.20.1+cu121` 组合相容，不会要求升级到 0.28。

```bash
cd "${FER_PROJECT_ROOT}"
python -m pip install -e ".[test]"
python -m pip check
python -c "import torch, torchvision; print('torch=', torch.__version__); print('torchvision=', torchvision.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0))"
```

`pip check` 必须无错误，`cuda=True`，GPU 名称应为当前获准使用的设备。

## 5. Kaggle API 与服务器直接下载 FER2013

在 Kaggle 网页的账号设置中创建 API token。下载到 Mac 的只是很小的
`kaggle.json` 凭据文件，不是数据集。先在 HIVE 创建受限目录：

```bash
# HIVE 执行
mkdir -p "${HOME}/.kaggle"
chmod 700 "${HOME}/.kaggle"
```

然后从 **Mac → Server** 传输凭据：

```bash
# Mac Terminal 执行；设置为实际的 SSH 账号和主机
export HIVE_SSH_TARGET="your_username@your_hive_hostname"
scp "${HOME}/Downloads/kaggle.json" \
  "${HIVE_SSH_TARGET}:~/.kaggle/kaggle.json"
```

服务器端先创建目标目录，再设置最小权限。不要打印、截图、提交或发送
`kaggle.json` 内容：

```bash
# HIVE 执行
chmod 600 "${HOME}/.kaggle/kaggle.json"
python -m pip install kaggle
kaggle datasets files -d deadskull7/fer2013
kaggle datasets download -d deadskull7/fer2013 \
  -p "${FER_DATA_ROOT}" --unzip
ls -lh "${FER_DATA_CSV}"
```

数据集始终直接进入 HIVE 的仓库外目录，不在 Mac 下载大文件，也不进入 Git。
CSV 必须包含 `emotion,pixels,Usage`，并保留三个官方 split 名称。

## 6. Preflight

Stage B 使用与 locked E7 baseline 相同的 `Usage`-first 路由。CSV reader 会将整行
词法读取为字符串字段，但对判定为 PrivateTest 的行只语义检查 `Usage`；其
emotion/pixels 不解析为 label/image，不验证或物化为 record/tensor，也不进入
Training/PublicTest hash、mean、manifest、训练、验证、checkpoint 选择、mask
生成、PublicTest 评估或指标。不能表述为 PrivateTest rows/bytes 从未被读取。

先用一个全新、仓库外目录生成正式 v2 mean/manifest，并生成服务器本地 mixed
配置。以下命令不手工拆分 CSV，也不修改仓库模板：

```bash
export FER_STAGE_B_ARTIFACTS="${FER_OUTPUT_ROOT}/stage-b-artifacts"
export FER_MIXED_CONFIG="${FER_STAGE_B_ARTIFACTS}/e7-occlusion-mixed.yaml"
python -m occlusion_fer.stage_b_artifacts \
  --data-path "${FER_DATA_CSV}" \
  --output-dir "${FER_STAGE_B_ARTIFACTS}"

python - <<'PY'
import os
from pathlib import Path
import yaml

template = Path("configs/experiments/fer2013_resnet18_e7_occlusion_mixed.yaml")
payload = yaml.safe_load(template.read_text(encoding="utf-8"))
artifact_root = Path(os.environ["FER_STAGE_B_ARTIFACTS"])
payload["dataset"]["path"] = os.environ["FER_DATA_CSV"]
payload["occlusion"]["artifacts"]["training_mean"] = str(
    artifact_root / "training_mean_v2.json"
)
payload["occlusion"]["artifacts"]["manifest"] = str(
    artifact_root / "publictest_manifest_v2.csv"
)
Path(os.environ["FER_MIXED_CONFIG"]).write_text(
    yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
)
PY
```

`stage_b_artifacts` 要求官方 `28709` Training 和 `3589` PublicTest；已有输出目录
会失败而不是覆盖。

preflight 检查环境、CSV schema、Training/PublicTest、batch、随机初始化模型
forward 和输出目录；它在 CSV 读取阶段跳过 PrivateTest 行，不解析其标签、像素、
数量或类别统计，也不会训练、下载预训练权重或保存 checkpoint。

```bash
cd "${FER_PROJECT_ROOT}"
set -o pipefail
python -m occlusion_fer.preflight \
  --config configs/experiments/fer2013_resnet18_e7_clean.yaml \
  --data-path "${FER_DATA_CSV}" \
  --output-dir "${FER_OUTPUT_ROOT}/preflight" \
  --device cuda \
  --batch-size 32 \
  2>&1 | tee "${FER_OUTPUT_ROOT}/preflight.log"
```

只有末尾出现 `PREFLIGHT PASSED` 才继续。应确认 Training、PublicTest 数量合理，
输出中没有 `test_samples` 或 `test_class_counts`，E7 clean batch 为 `[N,3,224,224]`，
logits 为 `[N,7]`。当前不要请求 PrivateTest；它只允许在三组 locked clean 与
Stage 8 三组 mixed checkpoint、v2 manifest 和最终报告计划全部锁定后由单独
批准的最终批次请求。

随后对 mixed 正式配置运行同一 preflight：

```bash
python -m occlusion_fer.preflight \
  --config "${FER_MIXED_CONFIG}" \
  --data-path "${FER_DATA_CSV}" \
  --output-dir "${FER_OUTPUT_ROOT}/preflight-mixed" \
  --device cuda --batch-size 128
```

## 7. 性能 smoke test

HIVE 已验证 `num_workers=0` 会让 CSV 图像预处理串行阻塞 GPU；`num_workers=4`
允许四个独立 worker 预取 batch，且与该账号可用的 6 个 CPU core 相符。先用
固定子集重现性能链路：

```bash
export FER_SMOKE_OUTPUT="${FER_OUTPUT_ROOT}/clean-smoke"
set -o pipefail
python -m occlusion_fer.train \
  --config configs/experiments/fer2013_resnet18_e7_high_resolution_longer.yaml \
  --data-path "${FER_DATA_CSV}" \
  --output-dir "${FER_SMOKE_OUTPUT}" \
  --seed 42 \
  --device cuda \
  --epochs 1 \
  --batch-size 128 \
  --num-workers 4 \
  --amp \
  --max-train-samples 8192 \
  --max-validation-samples 1024 \
  2>&1 | tee "${FER_OUTPUT_ROOT}/clean-smoke.log"
```

这必须显示 `SMOKE TEST — NOT A FORMAL EXPERIMENT`。确认 loss 有限、吞吐合理、
产生 best/last checkpoint 和 validation artifacts。smoke 数值不能写入正式结果。

## 8. 历史 E0-E7 seed-2026 筛选记录

本节及 8.1-8.4 记录配方如何收敛到 E7，仅用于 development provenance。不要
重跑这些命令来补正式证据，也不要把其中任何 seed-2026 指标并入 Stage 8 三种子
结果。E7 已经由当前协议锁定，下面的旧阈值、候选排序和筛选目录只表示历史
决策过程，不能重新触发配方选择或修改当前训练设置。

### 8.0 E0/E1 筛选

第一轮只比较 E0 原始配方与 E1 warmup/cosine。两者都按 PublicTest
macro-F1 保存 `best.pt`，early stopping 均关闭；不要运行 `final_evaluate`，也不要
把筛选阶段的 seed 2026 结果混入后续三 seed 正式汇总。

准备独立目录变量。preflight、run 和日志使用不同路径；训练 run 目录在命令执行前
必须不存在：

```bash
export FER_DATA_CSV="/home/ucla/anson-fer/data/raw/fer2013.csv"
export FER_SCREENING_ROOT="/home/ucla/anson-fer/results/screening"
export FER_SCREENING_LOGS="${FER_SCREENING_ROOT}/logs"
export FER_E0_PREFLIGHT="${FER_SCREENING_ROOT}/preflight/e0_baseline-seed2026"
export FER_E1_PREFLIGHT="${FER_SCREENING_ROOT}/preflight/e1_warmup_cosine-seed2026"
export FER_E0_RUN="${FER_SCREENING_ROOT}/e0_baseline/seed2026"
export FER_E1_RUN="${FER_SCREENING_ROOT}/e1_warmup_cosine/seed2026"
mkdir -p "${FER_SCREENING_LOGS}"
set -o pipefail
```

真实筛选必须从包含 PrivateTest 隔离修正的单一、干净 commit 启动，并固定两份配置
文件的内容哈希。以下 gate 有任何一步失败都不要继续：

```bash
if [ -n "$(git status --porcelain)" ]; then
  echo "Refusing screening run: Git worktree is dirty" >&2
  exit 1
fi
git rev-parse HEAD | tee "${FER_SCREENING_LOGS}/experiment-commit.txt"
sha256sum \
  configs/experiments/fer2013_resnet18_e0_baseline.yaml \
  configs/experiments/fer2013_resnet18_e1_warmup_cosine.yaml \
  | tee "${FER_SCREENING_LOGS}/experiment-config-sha256.txt"
```

E0 preflight：

```bash
python -m occlusion_fer.preflight \
  --config configs/experiments/fer2013_resnet18_e0_baseline.yaml \
  --data-path "${FER_DATA_CSV}" \
  --output-dir "${FER_E0_PREFLIGHT}" \
  --device cuda \
  --batch-size 128 \
  2>&1 | tee "${FER_SCREENING_LOGS}/e0-seed2026-preflight.log"
```

E0 training：

```bash
python -m occlusion_fer.train \
  --config configs/experiments/fer2013_resnet18_e0_baseline.yaml \
  --data-path "${FER_DATA_CSV}" \
  --output-dir "${FER_E0_RUN}" \
  --seed 2026 \
  --device cuda \
  --batch-size 128 \
  --num-workers 4 \
  --amp \
  2>&1 | tee "${FER_SCREENING_LOGS}/e0-seed2026-train.log"
```

E1 preflight：

```bash
python -m occlusion_fer.preflight \
  --config configs/experiments/fer2013_resnet18_e1_warmup_cosine.yaml \
  --data-path "${FER_DATA_CSV}" \
  --output-dir "${FER_E1_PREFLIGHT}" \
  --device cuda \
  --batch-size 128 \
  2>&1 | tee "${FER_SCREENING_LOGS}/e1-seed2026-preflight.log"
```

E1 training：

```bash
python -m occlusion_fer.train \
  --config configs/experiments/fer2013_resnet18_e1_warmup_cosine.yaml \
  --data-path "${FER_DATA_CSV}" \
  --output-dir "${FER_E1_RUN}" \
  --seed 2026 \
  --device cuda \
  --batch-size 128 \
  --num-workers 4 \
  --amp \
  2>&1 | tee "${FER_SCREENING_LOGS}/e1-seed2026-train.log"
```

查看日志和 GPU 进程：

```bash
tail -n 60 "${FER_SCREENING_LOGS}/e0-seed2026-train.log"
tail -n 60 "${FER_SCREENING_LOGS}/e1-seed2026-train.log"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

验证 resolved config；输出必须分别显示 `none` 与 `warmup_cosine`，且两者
`early_stopping.enabled` 都是 `false`：

```bash
python -c 'import pathlib,yaml; paths=[pathlib.Path(p) for p in ("/home/ucla/anson-fer/results/screening/e0_baseline/seed2026/resolved_config.yaml","/home/ucla/anson-fer/results/screening/e1_warmup_cosine/seed2026/resolved_config.yaml")]; [print(p, yaml.safe_load(p.read_text())["training"]) for p in paths]'
```

比较 PublicTest best 指标。筛选容差固定为
`E1 macro_f1 >= E0 macro_f1 - 0.0030`，使用原始 JSON 浮点值计算：

```bash
python -c 'import json,pathlib; root=pathlib.Path("/home/ucla/anson-fer/results/screening"); e0=json.loads((root/"e0_baseline/seed2026/validation/best_metrics.json").read_text()); e1=json.loads((root/"e1_warmup_cosine/seed2026/validation/best_metrics.json").read_text()); print("E0", e0["accuracy"], e0["macro_f1"]); print("E1", e1["accuracy"], e1["macro_f1"]); print("macro_f1_delta", e1["macro_f1"]-e0["macro_f1"]); print("passes_tolerance", e1["macro_f1"] >= e0["macro_f1"]-0.0030)'
```

## 8.1 E2 mild augmentation screening

E2 只在 Training split 应用锁定的轻量增强，PublicTest 保持确定性的 clean
预处理。增强参数写入
`configs/experiments/fer2013_resnet18_e2_mild_augmentation.yaml`：水平翻转和
仿射变换概率均为 `0.5`，旋转 `±7°`，平移上限为图像尺寸的 `5%`，缩放范围
`[0.97, 1.03]`，bilinear 插值，填充值 `0.0`。E2 的 scheduler 为 `none`，其余
训练预算、模型、优化器、checkpoint 指标和 seed 与 E0 相同。

E2 直接比较必须在包含 E2 的新 commit 上先重新运行一个全新的 E0-control，
再运行 E2。旧 commit 的 E0 结果只能作为一致性参考，不能作为正式配对对照：

```bash
export FER_E2_SCREENING_ROOT="${FER_WORK_ROOT}/results/screening-<E2_COMMIT_SHORT>"
export FER_E2_LOGS="${FER_E2_SCREENING_ROOT}/logs"
export FER_E2_PREFLIGHT="${FER_E2_SCREENING_ROOT}/preflight"
export FER_E2_CONTROL_RUN="${FER_E2_SCREENING_ROOT}/e0_control/seed2026"
export FER_E2_RUN="${FER_E2_SCREENING_ROOT}/e2_mild_augmentation/seed2026"
mkdir -p "${FER_E2_LOGS}" "${FER_E2_PREFLIGHT}"

git status --short
git rev-parse HEAD
sha256sum configs/experiments/fer2013_resnet18_e0_baseline.yaml \
  configs/experiments/fer2013_resnet18_e2_mild_augmentation.yaml
```

开始前两个 run 目录必须不存在；如果存在，停止并换用新的结果根目录。严格
顺序执行 E0-control 的 preflight 和训练，确认其产物完整后再执行 E2 的 preflight
和训练。命令只请求 Training 与 PublicTest，且不要调用 `final_evaluate`：

```bash
python -m occlusion_fer.preflight \
  --config configs/experiments/fer2013_resnet18_e0_baseline.yaml \
  --data-path "${FER_DATA_CSV}" \
  --output-dir "${FER_E2_PREFLIGHT}/e0-control-seed2026" \
  --device cuda --batch-size 128 \
  2>&1 | tee "${FER_E2_LOGS}/e0-control-seed2026-preflight.log"

python -m occlusion_fer.train \
  --config configs/experiments/fer2013_resnet18_e0_baseline.yaml \
  --data-path "${FER_DATA_CSV}" \
  --output-dir "${FER_E2_CONTROL_RUN}" \
  --seed 2026 --device cuda --batch-size 128 --num-workers 4 --amp \
  2>&1 | tee "${FER_E2_LOGS}/e0-control-seed2026-train.log"

python -m occlusion_fer.preflight \
  --config configs/experiments/fer2013_resnet18_e2_mild_augmentation.yaml \
  --data-path "${FER_DATA_CSV}" \
  --output-dir "${FER_E2_PREFLIGHT}/e2-seed2026" \
  --device cuda --batch-size 128 \
  2>&1 | tee "${FER_E2_LOGS}/e2-seed2026-preflight.log"

python -m occlusion_fer.train \
  --config configs/experiments/fer2013_resnet18_e2_mild_augmentation.yaml \
  --data-path "${FER_DATA_CSV}" \
  --output-dir "${FER_E2_RUN}" \
  --seed 2026 --device cuda --batch-size 128 --num-workers 4 --amp \
  2>&1 | tee "${FER_E2_LOGS}/e2-seed2026-train.log"
```

Both runs must have `status: completed`, 30 epochs, `best.pt`, `last.pt`, and
PublicTest validation artifacts, with no `final_test` directory or PrivateTest
counts/metrics. Compare raw `validation/best_metrics.json` values only after both
runs finish. This screening stage does not run E3/E4 or any PrivateTest evaluation.

## 8.2 E3 regularization screening

E3 只改变训练正则化：训练使用
`CrossEntropyLoss(label_smoothing=0.1)`，验证和最终评估继续使用普通
`CrossEntropyLoss(label_smoothing=0.0)`；AdamW `weight_decay=1e-3`。E3 的
augmentation 和 scheduler 均为 `none`，其余模型、输入、训练预算、seed 和
PublicTest macro-F1 checkpoint 规则与 E0 相同。

E3 的直接比较必须在同一 E3 commit 上先运行全新的 E0-control，再运行 E3。
旧 E0 结果只用于 consistency reference，不作为正式配对对照。建议结果根目录：

```text
/home/ucla/anson-fer/results/screening-<E3_COMMIT_SHORT>/
  e0_control/seed2026/
  e3_regularization/seed2026/
  logs/
  preflight/
```

筛选前验证 E0/E3 配置哈希和 Git commit；两个 run 必须严格顺序执行。训练和
preflight 只请求 Training 与 PublicTest，不调用 `final_evaluate`：

```bash
export FER_E3_SCREENING_ROOT="${FER_WORK_ROOT}/results/screening-<E3_COMMIT_SHORT>"
export FER_E3_LOGS="${FER_E3_SCREENING_ROOT}/logs"
export FER_E3_PREFLIGHT="${FER_E3_SCREENING_ROOT}/preflight"
export FER_E3_CONTROL_RUN="${FER_E3_SCREENING_ROOT}/e0_control/seed2026"
export FER_E3_RUN="${FER_E3_SCREENING_ROOT}/e3_regularization/seed2026"
mkdir -p "${FER_E3_LOGS}" "${FER_E3_PREFLIGHT}"

git status --short
git rev-parse HEAD
sha256sum configs/experiments/fer2013_resnet18_e0_baseline.yaml \
  configs/experiments/fer2013_resnet18_e3_regularization.yaml
```

先完成 E0-control 的 preflight 和 seed-2026 training，检查产物完整且没有
PrivateTest 统计；随后再运行 E3：

```bash
python -m occlusion_fer.preflight \
  --config configs/experiments/fer2013_resnet18_e0_baseline.yaml \
  --data-path "${FER_DATA_CSV}" \
  --output-dir "${FER_E3_PREFLIGHT}/e0-control-seed2026" \
  --device cuda --batch-size 128 \
  2>&1 | tee "${FER_E3_LOGS}/e0-control-seed2026-preflight.log"

python -m occlusion_fer.train \
  --config configs/experiments/fer2013_resnet18_e0_baseline.yaml \
  --data-path "${FER_DATA_CSV}" \
  --output-dir "${FER_E3_CONTROL_RUN}" \
  --seed 2026 --device cuda --batch-size 128 --num-workers 4 --amp \
  2>&1 | tee "${FER_E3_LOGS}/e0-control-seed2026-train.log"

python -m occlusion_fer.preflight \
  --config configs/experiments/fer2013_resnet18_e3_regularization.yaml \
  --data-path "${FER_DATA_CSV}" \
  --output-dir "${FER_E3_PREFLIGHT}/e3-seed2026" \
  --device cuda --batch-size 128 \
  2>&1 | tee "${FER_E3_LOGS}/e3-seed2026-preflight.log"

python -m occlusion_fer.train \
  --config configs/experiments/fer2013_resnet18_e3_regularization.yaml \
  --data-path "${FER_DATA_CSV}" \
  --output-dir "${FER_E3_RUN}" \
  --seed 2026 --device cuda --batch-size 128 --num-workers 4 --amp \
  2>&1 | tee "${FER_E3_LOGS}/e3-seed2026-train.log"
```

筛选条件固定为：
`E3_best_macro_f1 - E0_control_best_macro_f1 >= -0.0030`。不要调整 smoothing、
weight decay、checkpoint 指标或运行额外 seed；本阶段不访问 PrivateTest。

## 8.3 E4 combined screening

E4 只组合已通过筛选的 E2 与 E3，不引入新的算法：训练使用 E2 的 locked
`mild_affine` augmentation，并使用 E3 的
`CrossEntropyLoss(label_smoothing=0.1)` 与 AdamW `weight_decay=1e-3`。
validation 和未来锁定后的 final evaluation 继续使用普通交叉熵
(`label_smoothing=0.0`)；validation 不使用 augmentation。E1 的
warmup/cosine 不属于 E4，scheduler 必须保持 `none`，base learning rate 与
minimum learning rate 均为 `1e-4`。

E4 配置文件为
`configs/experiments/fer2013_resnet18_e4_combined.yaml`，基准为 E3
commit。由于 E4 改变训练数据管线，正式比较必须在同一个 E4 commit 上先
重新运行 E0-control，再严格顺序运行 E4；旧 E0/E2/E3 结果只作历史参考。
本阶段只使用 seed 2026 筛选，不运行 E2/E3 重复实验，不调用
`final_evaluate`，不访问 PrivateTest。

建议结果根目录：

```text
/home/ucla/anson-fer/results/screening-<E4_COMMIT_SHORT>/
  e0_control/seed2026/
  e4_combined/seed2026/
  logs/
  preflight/
```

先在服务器确认 clean checkout、四份历史配置和 E4 配置 hash；不得修改
E0/E1/E2/E3 YAML：

```bash
export FER_E4_SCREENING_ROOT="${FER_WORK_ROOT}/results/screening-<E4_COMMIT_SHORT>"
export FER_E4_LOGS="${FER_E4_SCREENING_ROOT}/logs"
export FER_E4_PREFLIGHT="${FER_E4_SCREENING_ROOT}/preflight"
export FER_E4_CONTROL_RUN="${FER_E4_SCREENING_ROOT}/e0_control/seed2026"
export FER_E4_RUN="${FER_E4_SCREENING_ROOT}/e4_combined/seed2026"
mkdir -p "${FER_E4_LOGS}" "${FER_E4_PREFLIGHT}"

git status --short
git rev-parse HEAD
sha256sum \
  configs/experiments/fer2013_resnet18_e0_baseline.yaml \
  configs/experiments/fer2013_resnet18_e1_warmup_cosine.yaml \
  configs/experiments/fer2013_resnet18_e2_mild_augmentation.yaml \
  configs/experiments/fer2013_resnet18_e3_regularization.yaml \
  configs/experiments/fer2013_resnet18_e4_combined.yaml
```

确认 E0-control 和 E4 run 目录不存在或为空后，依次执行以下命令。每个
preflight 必须只报告 Training 与 PublicTest；若出现 `PrivateTest`、test
counts 或 final-test artifacts，立即停止：

```bash
python -m occlusion_fer.preflight \
  --config configs/experiments/fer2013_resnet18_e0_baseline.yaml \
  --data-path "${FER_DATA_CSV}" \
  --output-dir "${FER_E4_PREFLIGHT}/e0-control-seed2026" \
  --device cuda --batch-size 128 \
  2>&1 | tee "${FER_E4_LOGS}/e0-control-seed2026-preflight.log"

python -m occlusion_fer.train \
  --config configs/experiments/fer2013_resnet18_e0_baseline.yaml \
  --data-path "${FER_DATA_CSV}" \
  --output-dir "${FER_E4_CONTROL_RUN}" \
  --seed 2026 --device cuda --batch-size 128 --num-workers 4 --amp \
  2>&1 | tee "${FER_E4_LOGS}/e0-control-seed2026-train.log"

python -m occlusion_fer.preflight \
  --config configs/experiments/fer2013_resnet18_e4_combined.yaml \
  --data-path "${FER_DATA_CSV}" \
  --output-dir "${FER_E4_PREFLIGHT}/e4-seed2026" \
  --device cuda --batch-size 128 \
  2>&1 | tee "${FER_E4_LOGS}/e4-seed2026-preflight.log"

python -m occlusion_fer.train \
  --config configs/experiments/fer2013_resnet18_e4_combined.yaml \
  --data-path "${FER_DATA_CSV}" \
  --output-dir "${FER_E4_RUN}" \
  --seed 2026 --device cuda --batch-size 128 --num-workers 4 --amp \
  2>&1 | tee "${FER_E4_LOGS}/e4-seed2026-train.log"
```

After both runs complete, compare the unrounded values in
`validation/best_metrics.json` using the pre-registered rule:
`E4_best_macro_f1 - E0_control_best_macro_f1 >= -0.0030`. Keep `best.pt`,
`last.pt`, history, validation metrics, logs and provenance for both runs. This
screening stage does not evaluate PrivateTest or determine any later E2/E3
parameters.

## 8.4 Second-round E5-E7 screening

The first-round E4 result is the locked reference for this limited second
round, not a successful final result. Its seed-2026 PublicTest values are
`accuracy=0.6870994706046253` and `macro_f1=0.6803268061914639`. The project
goal remains mean PrivateTest top-1 accuracy of at least `0.70` across formal
seeds 42, 123 and 2026; PrivateTest is not accessed during this screening.

Exactly three candidates are registered, and all retain the complete E4 recipe:

| Candidate | Config | Image size | Epochs | Only change from E4 |
|---|---|---:|---:|---|
| E5 | `fer2013_resnet18_e5_longer_training.yaml` | 112 | 50 | epochs |
| E6 | `fer2013_resnet18_e6_high_resolution.yaml` | 224 | 30 | image size |
| E7 | `fer2013_resnet18_e7_high_resolution_longer.yaml` | 224 | 50 | image size and epochs |

Do not change batch size 128, augmentation, label smoothing, weight decay,
learning rate, seed, scheduler, early stopping or checkpoint selection after
any intermediate result. E1 warmup/cosine remains excluded. All three candidates
must use one clean second-round commit and run sequentially unless a candidate
has a recorded execution failure.

This commit adds configurations and tests only, so E4 does not need to be
trained again by default. Before relying on the existing E4 reference, verify
the E4 YAML hash, commit ancestry, clean Git state and server tests. If any
production file under `src/occlusion_fer` differs from the E4 commit, stop and
do not run the second round under this protocol.

Use independent output directories:

```text
/home/ucla/anson-fer/results/second-round-<COMMIT_SHORT>/
  e5_longer_training/seed2026/
  e6_high_resolution/seed2026/
  e7_high_resolution_longer/seed2026/
  logs/
  preflight/
  second_round_decision.json
  second_round_decision.md
```

Prepare and verify provenance before any preflight:

```bash
export FER_SECOND_ROUND_ROOT="${FER_WORK_ROOT}/results/second-round-<COMMIT_SHORT>"
export FER_SECOND_ROUND_LOGS="${FER_SECOND_ROUND_ROOT}/logs"
export FER_SECOND_ROUND_PREFLIGHT="${FER_SECOND_ROUND_ROOT}/preflight"
mkdir -p "${FER_SECOND_ROUND_LOGS}" "${FER_SECOND_ROUND_PREFLIGHT}"

git status --short
git rev-parse HEAD
git merge-base --is-ancestor ee42c511b427e9a7a3fef8879ab83bad283b5cb3 HEAD
git diff ee42c511b427e9a7a3fef8879ab83bad283b5cb3..HEAD -- src/occlusion_fer
sha256sum configs/experiments/fer2013_resnet18_e{0,1,2,3,4,5,6,7}_*.yaml

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest \
  tests/test_config.py tests/test_models.py tests/test_torch_data.py \
  tests/test_losses.py tests/test_train.py tests/test_evaluation.py \
  tests/test_preflight.py \
  tests/test_final_evaluate.py tests/test_second_round_screening.py \
  -q -p no:cacheprovider
```

Run E5, E6 and E7 in that order. Each preflight and training entry point loads
only Training and PublicTest. Do not invoke `final_evaluate`:

```bash
set -o pipefail
for FER_CANDIDATE in \
  "e5_longer_training" \
  "e6_high_resolution" \
  "e7_high_resolution_longer"; do
  FER_CONFIG="configs/experiments/fer2013_resnet18_${FER_CANDIDATE}.yaml"
  FER_RUN="${FER_SECOND_ROUND_ROOT}/${FER_CANDIDATE}/seed2026"

  python -m occlusion_fer.preflight \
    --config "${FER_CONFIG}" \
    --data-path "${FER_DATA_CSV}" \
    --output-dir "${FER_SECOND_ROUND_PREFLIGHT}/${FER_CANDIDATE}-seed2026" \
    --device cuda --batch-size 128 \
    2>&1 | tee "${FER_SECOND_ROUND_LOGS}/${FER_CANDIDATE}-preflight.log" || break

  python -m occlusion_fer.train \
    --config "${FER_CONFIG}" \
    --data-path "${FER_DATA_CSV}" \
    --output-dir "${FER_RUN}" \
    --seed 2026 --device cuda --batch-size 128 --num-workers 4 --amp \
    2>&1 | tee "${FER_SECOND_ROUND_LOGS}/${FER_CANDIDATE}-train.log" || break
done
```

For 224x224 candidates, batch size remains 128. If CUDA reports OOM, preserve
the failure artifacts and stop that candidate; do not lower the batch size or
add gradient accumulation.

Use raw JSON floats for the pre-registered decision. A candidate passes the
basic protection only when both conditions hold:

```text
candidate_macro_f1 >= 0.6773268061914639
candidate_accuracy > 0.6870994706046253
```

It can enter formal three-seed training only when it also satisfies
`candidate_accuracy >= 0.70`. If multiple candidates satisfy all conditions,
rank by accuracy, then macro-F1, then lower compute in the order E5, E6, E7.
The following read-only script applies that rule without rounding:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["FER_SECOND_ROUND_ROOT"])
reference_accuracy = 0.6870994706046253
macro_f1_floor = 0.6773268061914639
formal_accuracy_gate = 0.70
specs = (
    ("E5", "e5_longer_training", 0),
    ("E6", "e6_high_resolution", 1),
    ("E7", "e7_high_resolution_longer", 2),
)
rows = []
for name, directory, compute_order in specs:
    path = root / directory / "seed2026/validation/best_metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    accuracy = float(metrics["accuracy"])
    macro_f1 = float(metrics["macro_f1"])
    protected = macro_f1 >= macro_f1_floor and accuracy > reference_accuracy
    rows.append({
        "name": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "passes_protection": protected,
        "eligible_for_formal_seeds": protected and accuracy >= formal_accuracy_gate,
        "compute_order": compute_order,
    })
eligible = [row for row in rows if row["eligible_for_formal_seeds"]]
eligible.sort(
    key=lambda row: (-row["accuracy"], -row["macro_f1"], row["compute_order"])
)
print(json.dumps({"candidates": rows, "selected": eligible[0] if eligible else None}, indent=2))
PY
```

If no candidate reaches PublicTest accuracy `0.70`, record the second round as
unsuccessful and stop. Do not run formal seeds, access PrivateTest, modify these
three candidates, or add a new candidate based on the results.

## 9. Stage 8 三个 mixed 正式 run（当前尚未开始）

只有以下 gate 全部通过才可以从服务器启动 Stage 8：集成实现已经形成干净
commit；HIVE/Linux 本地测试和 synthetic validation 通过；Usage-routed Training
与 PublicTest source、Training mean v2 和 PublicTest manifest v2 的身份已经
验证；运行计划和输出根目录已经锁定。若任一 gate 失败，保留证据并停止，不要
访问 PrivateTest。

Stage 8 的每一组命令都必须从同一个 integration worktree 开始。下面的
`STAGE8_LOCKED_HEAD` 是故意保留到修复完成、提交并完成 HIVE revalidation 后才
填写的占位符；仍为占位符时必须停止。

Required final form after revalidation: `STAGE8_LOCKED_HEAD=<FINAL_REVALIDATED_COMMIT>`.

```bash
export FER_STAGE8_WORKTREE="${FER_WORK_ROOT}/project/occlusion-fer-stage-b"
export STAGE8_LOCKED_HEAD="FINAL_REVALIDATED_COMMIT"
cd "${FER_STAGE8_WORKTREE}"
pwd
git branch --show-current
git rev-parse HEAD
git status --porcelain=v2 --untracked-files=all
test "$(pwd)" = "${FER_STAGE8_WORKTREE}" || exit 1
test "$(git branch --show-current)" = "stage-b/e7-occlusion-integration" || exit 1
test "${STAGE8_LOCKED_HEAD}" != "FINAL_REVALIDATED_COMMIT" || {
  echo "STOP: replace STAGE8_LOCKED_HEAD with the final revalidated commit" >&2
  exit 1
}
test "$(git rev-parse HEAD)" = "${STAGE8_LOCKED_HEAD}" || {
  echo "STOP: Stage 8 HEAD does not match STAGE8_LOCKED_HEAD" >&2
  exit 1
}
test -z "$(git status --porcelain=v2 --untracked-files=all)" || {
  echo "STOP: Stage 8 worktree is dirty" >&2
  exit 1
}
```

Failure of any path, branch, HEAD, or clean-worktree check blocks Stage 8 before
data access or training.

开始前记录 Git、环境和 GPU；不要使用 `max-*-samples`：

```bash
git status --short --branch
git rev-parse HEAD
python -m pip freeze > "${FER_OUTPUT_ROOT}/environment.txt"
nvidia-smi > "${FER_OUTPUT_ROOT}/nvidia-smi-before.txt"
```

首先核验三组 locked E7 clean checkpoint 存在并记录 SHA-256。不要重新训练、
筛选或覆盖它们：

```bash
export FER_LOCKED_CLEAN_ROOT="${FER_WORK_ROOT}/results/formal-e7-4cb1e0f"
for FER_SEED in 42 123 2026; do
  test -f "${FER_LOCKED_CLEAN_ROOT}/seed${FER_SEED}/best.pt" || exit 1
  sha256sum "${FER_LOCKED_CLEAN_ROOT}/seed${FER_SEED}/best.pt"
done
```

Stage 8 只新增三个 mixed formal training runs。三组 clean E7 checkpoint
(`seed42`, `seed123`, `seed2026`) are reused as-is for the comparison and later
evaluation. Do not describe this as six new training runs, and do not write any
output below the locked clean-result directory.

Stage 8 只新增 mixed。`${FER_MIXED_CONFIG}` 是第 6 节生成的仓库外解析副本；
每个 seed 使用新的独立目录：

```bash
set -o pipefail
for FER_SEED in 42 123 2026; do
  FER_RUN_OUTPUT="${FER_OUTPUT_ROOT}/formal/mixed/seed${FER_SEED}"
  python -m occlusion_fer.train \
    --config "${FER_MIXED_CONFIG}" \
    --data-path "${FER_DATA_CSV}" \
    --output-dir "${FER_RUN_OUTPUT}" \
    --seed "${FER_SEED}" --device cuda --amp \
    2>&1 | tee "${FER_OUTPUT_ROOT}/formal/mixed-seed${FER_SEED}.log" || break
done
```

任何 seed 失败都停止后续循环并保留 `failure.json`、日志和已有输出，不跳过后只
报告成功 seed。每个成功训练目录必须有：

```text
resolved_config.yaml
run_metadata.json
history.json
history.csv
best.pt
last.pt
validation/best_metrics.json
validation/best_per_class_metrics.csv
validation/best_confusion_matrix.csv
validation/best_predictions.csv
validation/last_metrics.json
validation/last_per_class_metrics.csv
validation/last_confusion_matrix.csv
validation/last_predictions.csv
```

`best.pt` 由 clean PublicTest macro-F1 严格提升决定；masked PublicTest 指标不
参与选择，PrivateTest 未被训练入口加载。检查三个 mixed `run_metadata.json` 的
`status` 均为 `completed`、Git commit 相同、`git_dirty` 为 `false`、seed 和
`training.mode` 正确；再与三组 locked clean metadata 核对配方字段和证据身份。

## 10. PublicTest 十条件评估与 PrivateTest 门禁

六个 best checkpoint 必须使用同一个 v2 PublicTest manifest，逐一评估 clean
加九个遮挡条件。每个 checkpoint/condition 保存 metrics、逐类指标、混淆矩阵、
逐样本 predictions、paired clean-to-occluded drop 和完整 provenance。该分析不
能回写 checkpoint 或训练配置。

每个 checkpoint 都必须通过同一个可复现 CLI 入口评价；不要手写 Python 调用或
改用旧的 clean-only evaluator：

```bash
PYTHONPATH=src python -m occlusion_fer.occlusion_evaluate \
  --config "${FER_EVAL_CONFIG}" \
  --checkpoint "${FER_CHECKPOINT}" \
  --data-path "${FER_DATA_CSV}" \
  --training-mean "${FER_TRAINING_MEAN_V2}" \
  --manifest "${FER_PUBLICTEST_MANIFEST_V2}" \
  --output-dir "${FER_EVAL_OUTPUT}" \
  --device cuda --batch-size 128 --num-workers 4 --amp
```

所有从 source tree 直接执行的 evaluator 命令，包括 `--help` 和 synthetic check，
都必须使用 `PYTHONPATH=src python -m occlusion_fer.occlusion_evaluate ...`。该命令
只按 `Usage` 路由 `Training` 与 `PublicTest`；CSV reader 会词法读取整行，但对排除的
`PrivateTest` 行只语义检查 `Usage`。其 emotion/pixels 不解析为 label/image，不验证
或物化为 record/tensor，不进入 Training/PublicTest hash、mean、manifest、训练、
验证、checkpoint 选择、mask 生成、PublicTest 评估或指标；没有 PrivateTest 输出路由。

当前 `occlusion_fer.occlusion_evaluate` 提供 PublicTest-only CLI，没有 PrivateTest
路由；不要把旧的 `final_evaluate` clean-only 命令当作 v2 最终评估。PrivateTest
只有在六个 checkpoint SHA-256、v2 artifacts、评估代码和报告计划全部锁定后，才
由另一个明确批准的最终批次执行一次。当前不要执行任何 PrivateTest 命令。

最终报告必须保留三个 seed、两种策略和十个条件；PrivateTest 结果不能反馈到
checkpoint、seed、mask 或方法选择。

```text
conditions/<condition>/<condition>_metrics.json
conditions/<condition>/<condition>_per_class_metrics.csv
conditions/<condition>/<condition>_confusion_matrix.csv
conditions/<condition>/<condition>_predictions.csv
conditions/<condition>_paired_drop.csv
conditions/<condition>_provenance.json
evaluation_provenance.json
```

`evaluation_provenance.json` 在十个 condition 产物全部成功写入后最后生成，是正式
十条件评价唯一的成功完成标志。目录中若存在 `failure.json`，或只有部分 condition
文件而没有 `evaluation_provenance.json`，必须判定为未完成；保留诊断证据并改用全新
输出目录，不能原地重跑或覆盖。

## 11. 指标定义与论文产物映射

类别顺序始终为 `angry, disgust, fear, happy, sad, surprise, neutral`。混淆矩阵
的行是真实标签，列是预测标签。macro-F1 对全部七类的 F1 做等权平均；零分母
按 0 处理，不因某类样本少而移除该类。

论文写作时使用原始产物建立可追溯关系：

| 论文内容 | 原始证据 | 推荐呈现 |
|---|---|---|
| Methods：环境与复现 | `run_metadata.json` | 版本、GPU、seed、Git commit 表 |
| Methods：训练设置 | `resolved_config.yaml` | 模型、预处理、优化器参数表 |
| 训练过程 | `history.csv` | train loss、validation loss、accuracy、macro-F1 曲线 |
| checkpoint 选择 | `validation/best_metrics.json` | 最佳 validation macro-F1 与对应 epoch |
| PublicTest 十条件结果 | `conditions/<condition>/<condition>_metrics.json` | 三 seed 的 accuracy/macro-F1 及均值、标准差 |
| clean-to-occluded drop | `conditions/<condition>_paired_drop.csv` | 同一 sample ID 的配对变化 |
| 类别差异 | `conditions/<condition>/<condition>_per_class_metrics.csv` | 每类 precision/recall/F1 表或柱状图 |
| 错误结构 | `conditions/<condition>/<condition>_confusion_matrix.csv` | 统一色阶的混淆矩阵 |
| 错误分析 | `conditions/<condition>/<condition>_predictions.csv` | 按 sample ID 追踪正确性、置信度和七类概率 |

图表应由 CSV/JSON 自动生成，保留脚本和输入 commit；不要把图中数值手工录入。
所有 seed 都必须报告，不能只选择最高分。当前阶段尚未提供跨种子汇总和绘图
脚本，后续应在遮挡实验设计锁定后统一实现，以确保 clean/occluded 使用同一
统计口径和图形模板。

## 12. 失败处理与常见问题

- `CUDA was requested but is not available`：确认会话仍有 GPU、`nvidia-smi` 和
  CUDA PyTorch 均正常。
- GPU 利用率低且 epoch 很慢：确认 `--num-workers 4`、没有其他用户大量占用
  CPU/GPU、数据不是网络故障盘；不要盲目增加超过可用 CPU 的 worker。
- 显存不足：先停止；若必须调整 batch size，应在任何 PrivateTest 之前锁定新值，
  并对全部 seed 和未来对比策略统一使用。
- `failure.json`：保留它和日志，修复后使用新输出目录；不得删除负面实验来制造
  全部成功的印象。
- Kaggle 401/403：检查账号许可、文件位置和 `600` 权限，不打印 token。
- placeholder 路径：始终用 `--data-path`，不修改提交的 YAML 为个人路径。
- 预训练权重下载失败：修复网络或缓存；不能静默切换到随机初始化并作为正式
  ResNet-18 结果。
- final results already exist：不覆盖。核对该 run 是否已完成最终评估；若是，
  使用原结果；若实验协议确需重做，必须先记录原因并使用全新 run 目录。

## 13. Git 与安全检查

数据、图片、checkpoint、outputs、logs、runs、虚拟环境、cache、`.env`、
`kaggle.json`、token 和 SSH 私钥都不能进入 Git。正式运行前后检查：

```bash
git status --short --branch
git ls-files | grep -E '(fer2013\.csv|\.pt$|\.pth$|\.ckpt$|kaggle\.json|\.env$)' && echo "STOP: forbidden tracked file"
```

不要使用 `git add .` 保存实验产物。训练输出只保留在 `${FER_OUTPUT_ROOT}` 并按
学校的数据与备份政策管理。
