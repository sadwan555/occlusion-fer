# Occlusion FER Server Quick Start

For detailed instructions, see docs/server_runbook.md

## 项目简介

本项目在 FER2013 数据集标签上训练 ImageNet 预训练 ResNet-18，并为后续遮挡
鲁棒性实验提供 clean baseline。数据、权重、输出和密钥必须保存在 Git 仓库外。

## 1. 服务器环境准备

```bash
export FER_ROOT="${HOME}/anson-fer"
export FER_PROJECT="${FER_ROOT}/project/occlusion-fer"
export FER_DATA="${FER_ROOT}/data/raw"
export FER_ENV="${FER_ROOT}/envs/occlusion-fer"
export FER_OUTPUT="${FER_ROOT}/outputs"

nvidia-smi
mkdir -p "${FER_ROOT}/project" "${FER_DATA}" "${FER_ROOT}/envs" "${FER_OUTPUT}"
git clone git@github.com:sadwan555/occlusion-fer.git "${FER_PROJECT}"
cd "${FER_PROJECT}"
git status --short --branch
```

## 2. 创建并激活 Python 环境

```bash
python3 -m venv "${FER_ENV}"
source "${FER_ENV}/bin/activate"
python -m pip install --upgrade pip
```

每次重新登录后，重新设置 `FER_*` 变量并激活该环境。

## 3. 安装依赖

先按 [PyTorch Start Locally](https://docs.pytorch.org/get-started/locally/)
安装与服务器驱动匹配的 Linux CUDA 版 PyTorch，再安装项目：

```bash
cd "${FER_PROJECT}"
python -m pip install -e ".[test]"
python -m pip check
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

项目依赖范围与 HIVE 已验证组合 `torch 2.5.x + torchvision 0.20.x` 一致。

## 4. 下载 FER2013 数据集

把 Kaggle API 凭据放在服务器的 `${HOME}/.kaggle/kaggle.json`，设置权限；CSV
直接下载到服务器，不经过 Mac 保存大数据集：

```bash
mkdir -p "${HOME}/.kaggle"
chmod 700 "${HOME}/.kaggle"
chmod 600 "${HOME}/.kaggle/kaggle.json"
python -m pip install kaggle
kaggle datasets files -d deadskull7/fer2013
kaggle datasets download -d deadskull7/fer2013 -p "${FER_DATA}" --unzip
ls -lh "${FER_DATA}/fer2013.csv"
```

## 5. Preflight 检查

```bash
python -m occlusion_fer.preflight \
  --config configs/fer2013_resnet18_clean.yaml \
  --data-path "${FER_DATA}/fer2013.csv" \
  --output-dir "${FER_OUTPUT}/preflight" \
  --device cuda \
  --batch-size 32
```

只有末尾出现 `PREFLIGHT PASSED` 才继续。

## 6. 训练入口

先做性能 smoke test：

```bash
python -m occlusion_fer.train \
  --config configs/fer2013_resnet18_clean.yaml \
  --data-path "${FER_DATA}/fer2013.csv" \
  --output-dir "${FER_OUTPUT}/clean-smoke" \
  --seed 42 --device cuda --epochs 1 \
  --batch-size 128 --num-workers 4 --amp \
  --max-train-samples 8192 --max-validation-samples 1024
```

确认 smoke 成功后，删除样本上限并分别使用 seeds `42`、`123`、`2026`，每个
run 使用独立输出目录和相同的 30 epoch 参数。完整三种子命令与最终 PrivateTest
步骤见 [`docs/server_runbook.md`](docs/server_runbook.md)。

## 7. 常见问题

- CUDA 不可用：检查当前会话、`nvidia-smi` 和 CUDA 版 PyTorch。
- GPU 利用率低：确认使用 `--num-workers 4`；HIVE 实测 workers 为 0 会严重
  阻塞数据供应。
- Kaggle 401/403：检查账号访问权限及 `kaggle.json` 的位置和 `600` 权限，
  不要输出文件内容。
- 数据路径是 placeholder：通过 `--data-path` 覆盖，不要把个人绝对路径写回
  YAML。
- 显存不足：减小 batch size，并为所有对比实验统一记录新值；不要只改变某个
  seed。
- 已存在最终结果：`final_evaluate` 会拒绝覆盖，保留原始证据并使用新的、明确
  命名的正式 run 目录。
