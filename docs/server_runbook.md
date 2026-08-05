# HIVE 部署、正式 clean baseline 与论文产物手册

## 1. 目的与研究边界

本手册用于在 HIVE 的 Linux + NVIDIA GPU 环境中完成 FER2013 clean
ResNet-18 的环境验证、三种子正式训练和一次锁定后的 PrivateTest 评估。

项目只预测 FER2013 定义的七个表情标签。不能把结果解释为识别真实内在情绪、
困惑、理解程度、参与度或学习效果，也不能据此宣称真实世界鲁棒性、跨数据集
泛化、创新性或最先进性能。

正式 clean 协议在首次 PrivateTest 评估前锁定为：

- ImageNet 预训练、标准 stem 的 ResNet-18；
- 灰度图复制为三通道，bilinear resize 到 `112×112`，ImageNet normalization；
- 官方 `Training` 训练、`PublicTest` 验证、`PrivateTest` 最终测试；
- AdamW，learning rate `1e-4`，weight decay `1e-4`；
- batch size `128`，epochs `30`，DataLoader workers `4`，CUDA AMP；
- seeds `42`、`123`、`2026`；
- 按 clean PublicTest macro-F1 选择 `best.pt`，相同分数保留更早 epoch；
- 三个 seed 使用相同模型、处理、超参数和 checkpoint 规则。

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

正式运行必须满足：使用自己的账号、GPU 可见、磁盘足够、仓库在 `main`、工作区
干净且本地与 `origin/main` 同步。发现陌生 GPU 进程时先确认归属，不结束其他
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

preflight 检查环境、CSV schema、Training/PublicTest、batch、随机初始化模型
forward 和输出目录；它在 CSV 读取阶段跳过 PrivateTest 行，不解析其标签、像素、
数量或类别统计，也不会训练、下载预训练权重或保存 checkpoint。

```bash
cd "${FER_PROJECT_ROOT}"
set -o pipefail
python -m occlusion_fer.preflight \
  --config configs/fer2013_resnet18_clean.yaml \
  --data-path "${FER_DATA_CSV}" \
  --output-dir "${FER_OUTPUT_ROOT}/preflight" \
  --device cuda \
  --batch-size 32 \
  2>&1 | tee "${FER_OUTPUT_ROOT}/preflight.log"
```

只有末尾出现 `PREFLIGHT PASSED` 才继续。应确认 Training、PublicTest 数量合理，
输出中没有 `test_samples` 或 `test_class_counts`，batch 为 `[N,3,112,112]`，
logits 为 `[N,7]`。PrivateTest 只允许由锁定后的 `final_evaluate` 入口显式请求。

## 7. 性能 smoke test

HIVE 已验证 `num_workers=0` 会让 CSV 图像预处理串行阻塞 GPU；`num_workers=4`
允许四个独立 worker 预取 batch，且与该账号可用的 6 个 CPU core 相符。先用
固定子集重现性能链路：

```bash
export FER_SMOKE_OUTPUT="${FER_OUTPUT_ROOT}/clean-smoke"
set -o pipefail
python -m occlusion_fer.train \
  --config configs/fer2013_resnet18_clean.yaml \
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

## 8. E0/E1 seed 2026 筛选

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

## 9. 三种子正式 clean 训练

开始前再次确认 Git 干净，并记录环境。不要使用 `max-*-samples`：

```bash
git status --short --branch
git rev-parse HEAD
python -m pip freeze > "${FER_OUTPUT_ROOT}/environment.txt"
nvidia-smi > "${FER_OUTPUT_ROOT}/nvidia-smi-before.txt"
```

依次运行三个 seed，每个 run 使用新的独立目录。若目录已有正式文件，不复用该
目录；保留失败和负面实验，另建带时间或原因后缀的新目录。

```bash
set -o pipefail
for FER_SEED in 42 123 2026; do
  FER_RUN_OUTPUT="${FER_OUTPUT_ROOT}/clean-seed${FER_SEED}"
  python -m occlusion_fer.train \
    --config configs/fer2013_resnet18_clean.yaml \
    --data-path "${FER_DATA_CSV}" \
    --output-dir "${FER_RUN_OUTPUT}" \
    --seed "${FER_SEED}" \
    --device cuda \
    --epochs 30 \
    --batch-size 128 \
    --num-workers 4 \
    --amp \
    2>&1 | tee "${FER_OUTPUT_ROOT}/clean-seed${FER_SEED}.log" || break
done
```

任何 seed 失败都停止循环并保留 `failure.json`、日志和已有输出，不跳过后只报告
成功 seed。每个成功目录必须有：

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

`best.pt` 由 clean PublicTest macro-F1 严格提升决定；PrivateTest 未被训练入口
加载。检查三个 `run_metadata.json` 的 `status` 均为 `completed`、Git commit
相同、`git_dirty` 为 `false`、seed 分别正确。

## 10. 锁定后的 PrivateTest 最终评估

本节命令已准备好，但当前 clean 开发阶段不要立即执行。只有在以下条件全部满足
后才执行：clean 和 mixed 的六个正式 run 均完成；模型、超参数、九个遮挡条件、
final masks 与 checkpoint 规则全部锁定；不再根据测试结果选择 epoch、seed、
mask 或方法；计划报告全部三个 seed 和全部十个条件。下面的命令只是最终评估
批次中的 clean condition，后续遮挡阶段必须补齐其余九个 condition 的同批评估。

```bash
for FER_SEED in 42 123 2026; do
  FER_RUN_OUTPUT="${FER_OUTPUT_ROOT}/clean-seed${FER_SEED}"
  python -m occlusion_fer.final_evaluate \
    --config configs/fer2013_resnet18_clean.yaml \
    --checkpoint "${FER_RUN_OUTPUT}/best.pt" \
    --data-path "${FER_DATA_CSV}" \
    --output-dir "${FER_RUN_OUTPUT}" \
    --device cuda \
    --batch-size 128 \
    --num-workers 4 \
    --amp \
    --confirm-private-test || break
done
```

该入口只加载 checkpoint、只构建 PrivateTest dataset、不创建 optimizer、不反向
传播，并拒绝覆盖已有 `final_test/clean_metrics.json`。每个 run 新增：

```text
final_test/clean_metrics.json
final_test/clean_per_class_metrics.csv
final_test/clean_confusion_matrix.csv
final_test/clean_predictions.csv
```

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
| clean 总体结果 | `final_test/clean_metrics.json` | 三 seed 的 accuracy/macro-F1 及均值、标准差 |
| 类别差异 | `*_per_class_metrics.csv` | 每类 precision/recall/F1 表或柱状图 |
| 错误结构 | `*_confusion_matrix.csv` | 统一色阶的混淆矩阵 |
| 错误分析 | `*_predictions.csv` | 按 sample ID 追踪正确性、置信度和七类概率 |

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
