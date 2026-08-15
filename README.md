# Occlusion FER

这是一个用于 FER2013 合成遮挡实验的 PyTorch 研究仓库。项目使用同一套
ImageNet-pretrained ResNet-18，比较 clean-only 与 mixed clean/occluded training，
并在 clean、upper-face、lower-face 和 random-rectangle 条件下评估七类
FER2013 数据集标签。

本项目只预测数据集定义的 facial-expression labels。结果不能解释为对真实内在
情绪、困惑、理解、参与度、学习结果或其他认知状态的识别，也不支持真实场景
鲁棒性、跨数据集泛化、算法新颖性或 state-of-the-art 声明。

## 实验设计

正式第一版实验已经完成，锁定范围如下：

- 数据集：FER2013；`Training` 用于训练，`PublicTest` 用于验证与 checkpoint
  选择，`PrivateTest` 只用于协议冻结后的最终评估；
- 模型：标准 stem、七类输出的 ImageNet-pretrained ResNet-18；
- 输入：48x48 灰度图复制为三通道，bilinear resize 到 224x224，再做 ImageNet
  normalization；
- 正式种子：42、123、2026；
- 训练策略：clean-only 与 mixed clean/occluded；
- 遮挡：`upper_face`、`lower_face`、`random_rectangle`，target ratio 为 0.20、
  0.30、0.40；
- 协议：`occlusion-v2-224`，fill value 来自 Training split pixel mean，固定 mask
  seed 为 20260804；
- 评估：每个 best checkpoint 使用相同的 1 个 clean 与 9 个 masked conditions；
- 指标：accuracy、macro-F1、confusion matrix、per-class metrics、per-sample
  predictions 和 clean-to-occluded drop；
- checkpoint rule：每个 epoch 的 clean PublicTest macro-F1 严格提升时更新
  `best.pt`，同时保留 `last.pt`。

主要实验对应关系：

1. Experiment 1：三 seed clean-only E7 baseline。
2. Experiment 2：clean-trained checkpoints 在十个固定条件上的评估。
3. Experiment 3：三 seed mixed training，以及 clean-only 与 mixed 的同条件比较。
4. Final evaluation：六个冻结 best checkpoints 在 PrivateTest 的十个条件上一次性
   评估，共 60 个 model-condition evaluations。

Grad-CAM 是可选的 PublicTest 定性分析，只使用 seed-42 clean/mixed checkpoints，
不参与训练、checkpoint 选择或定量结论。

## 正式代码 lineage

正式结果来自 Git 中保留的不同阶段分支。当前仓库没有一个已审核的 integration
commit 把所有阶段合并到同一棵文件树，因此复现时必须 checkout 精确 commit，不能
把当前分支中的 112/v1 smoke 配置当作正式配置。

| 阶段 | Commit | 用途 |
|---|---|---|
| E7 clean formal | `4cb1e0ffe4b55efc090a45cfed560b28f50b9509` | 224x224、50 epoch clean-only 三 seed checkpoints |
| Stage 8 | `c1c9187aa2ddf7dd84906c7f139ad9a750ef202d` | `occlusion-v2-224` artifacts、mixed training、PublicTest 十条件评估 |
| Private final | `7e154aca1e95ef78ea7e3bc8767bcb21ca769335` | 冻结 checkpoint registry、PrivateTest manifest/plan/preflight/final evaluator |
| Legacy baseline | `da889bdd818cab6403767f2f6c7d5391d8317324` | 112x112、30 epoch 历史实验，仅用于追溯 |

`da889bd` 的 checkpoint、metrics、figures 和 v1 masks 不得混入当前论文结果。

## 仓库结构

```text
configs/                 当前 checkout 可见的配置；112/v1 文件属于 smoke/历史阶段
src/occlusion_fer/       数据、模型、评估、PrivateTest 和论文图表代码
tests/                   synthetic/unit/smoke 测试
docs/                    研究边界、运行说明、artifact inventory 与 provenance
outputs/                 本地输出，Git ignored
03_PAPER/                生成的论文图表，Git ignored
local_archive/           历史报告，Git ignored
```

当前本地正式图表位置：

- `outputs/gradcam_224_v2/`：正式 PublicTest Grad-CAM 输出；
- `03_PAPER/private_test_figures/`：PrivateTest 最终图表与绘图源表；
- `03_PAPER/occlusion-fer-paper-figures/`：方法图等生成图表。

正式训练结果、checkpoints、manifests、predictions 和 release archives 不进入 Git。
文件名、SHA-256、已知缺口以及 GitHub Release 建议见
[`docs/artifact_inventory.md`](docs/artifact_inventory.md)。

## 环境

Python 依赖以 `pyproject.toml` 为准。CUDA 服务器应先按 PyTorch 官方说明安装与
驱动匹配的 PyTorch，再安装本项目：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,paper]"
python -m pip check
```

FER2013 CSV、checkpoint、manifest 和输出路径都通过 YAML 或 CLI 显式提供；不要把
本机路径写回代码或配置。

## 复现正式 Stage 8

建议把复现用的独立 worktree 放在 workspace 的归档区，避免再次污染根目录：

```bash
git worktree add ../90_ARCHIVE/worktrees/occlusion-fer-e7-clean 4cb1e0ffe4b55efc090a45cfed560b28f50b9509
git worktree add ../90_ARCHIVE/worktrees/occlusion-fer-stage8 c1c9187aa2ddf7dd84906c7f139ad9a750ef202d
git worktree add ../90_ARCHIVE/worktrees/occlusion-fer-private-final 7e154aca1e95ef78ea7e3bc8767bcb21ca769335
```

在 E7 clean worktree 中，先 preflight，再对三个正式种子分别运行同一配置：

```bash
python -m occlusion_fer.preflight \
  --config configs/experiments/fer2013_resnet18_e7_clean.yaml \
  --data-path /path/to/fer2013.csv \
  --output-dir /path/to/preflight-output \
  --device cuda

python -m occlusion_fer.train \
  --config configs/experiments/fer2013_resnet18_e7_clean.yaml \
  --data-path /path/to/fer2013.csv \
  --output-dir /path/to/clean-seed42 \
  --seed 42 --device cuda --epochs 50 \
  --batch-size 128 --num-workers 4 --amp
```

在 Stage 8 worktree 中生成 Training/PublicTest v2 artifacts，再使用
`fer2013_resnet18_e7_occlusion_mixed.yaml` 运行三个 mixed seeds，并用
`occlusion_fer.occlusion_evaluate` 对六个 best checkpoints 运行相同十条件评估：

```bash
python -m occlusion_fer.stage_b_artifacts \
  --data-path /path/to/fer2013.csv \
  --output-dir /path/to/stage8-protocol-artifacts

python -m occlusion_fer.occlusion_evaluate --help
```

不要重新生成或调整已冻结的正式结果。上述入口用于复核代码与复现流程；实际复现
必须保留 resolved config、Git identity、artifact/checkpoint SHA 和失败记录。

## PrivateTest 与论文图表

PrivateTest evaluator 只有在 plan、FER2013 source、Training mean、manifest 和六个
checkpoint 全部通过身份检查，并显式提供 `--confirm-private-test` 时才会运行。已有
最终结果不得因整理或绘图而重算。

从现有冻结结果重新生成 PrivateTest 图表时，输入与输出路径必须显式给出：

```bash
python -m occlusion_fer.private_paper_figures \
  --private-root /path/to/extracted/final-private-test-v2 \
  --output-dir /path/to/new-private-test-figures
```

该脚本只读取已完成结果并核验 provenance，不训练模型、不执行 inference，也不修改
source results。

## 测试

```bash
PYTHONPATH=src pytest -q
python3 -m compileall -q src tests
git diff --check
```

测试使用 synthetic fixtures；不会把 small CNN、smoke run 或人工数据输出当作研究
结果。服务器运行与阶段边界见 [`docs/server_runbook.md`](docs/server_runbook.md)。
