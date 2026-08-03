# Occlusion FER Server Quick Start

For detailed instructions, see docs/server_runbook.md

## 项目简介

本项目使用 FER2013 的七个数据集标签训练 ResNet-18，并研究合成面部遮挡对
分类性能的影响。数据、模型权重和实验输出必须保存在 Git 仓库外。

本页是服务器快速开始指南；环境解释、排错方法和安全边界请阅读
[`docs/server_runbook.md`](docs/server_runbook.md)。

## 1. 服务器环境准备

先确认基础环境和 NVIDIA GPU：

```bash
whoami
uname -a
nvidia-smi
df -h
git --version
python3 --version
```

在个人目录创建互相隔离的项目、数据、环境和输出目录：

```bash
export FER_ROOT="${HOME}/anson-fer"
export FER_PROJECT="${FER_ROOT}/project/occlusion-fer"
export FER_DATA="${FER_ROOT}/data/raw"
export FER_ENV="${FER_ROOT}/envs/occlusion-fer"
export FER_OUTPUT="${FER_ROOT}/outputs"

mkdir -p "${FER_ROOT}/project" "${FER_DATA}" "${FER_ROOT}/envs" "${FER_OUTPUT}"
git clone git@github.com:sadwan555/occlusion-fer.git "${FER_PROJECT}"
cd "${FER_PROJECT}"
git status
```

## 2. 创建并激活 Python 环境

项目要求 Python 3.10 或更高版本：

```bash
python3 -m venv "${FER_ENV}"
source "${FER_ENV}/bin/activate"
python --version
python -m pip install --upgrade pip
```

每次重新登录服务器后，都要重新设置上面的 `FER_*` 变量并激活环境。

## 3. 安装依赖

不要猜测或固定 CUDA wheel 地址。根据当天服务器的 `nvidia-smi` 输出，在
[PyTorch Start Locally](https://docs.pytorch.org/get-started/locally/) 选择 Linux、
Pip、Python 和适合该服务器的 CUDA 版本，安装匹配的 `torch` 与
`torchvision`。

然后安装项目和测试依赖：

```bash
cd "${FER_PROJECT}"
python -m pip install -e ".[test]"
python -m pip check
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available())"
```

只有 `pip check` 无错误且 `torch.cuda.is_available()` 为 `True` 时，才继续 GPU
检查。

## 4. 下载 FER2013 数据集

在服务器上配置 Kaggle 凭据，确保 `kaggle.json` 不在仓库中：

```bash
mkdir -p "${HOME}/.kaggle"
chmod 700 "${HOME}/.kaggle"
chmod 600 "${HOME}/.kaggle/kaggle.json"
python -m pip install kaggle
```

先确认数据条目包含项目需要的 `fer2013.csv`，再直接下载到仓库外：

```bash
kaggle datasets files -d deadskull7/fer2013
kaggle datasets download -d deadskull7/fer2013 -p "${FER_DATA}" --unzip
ls -lh "${FER_DATA}/fer2013.csv"
```

CSV 必须包含 `emotion,pixels,Usage`，并保留 `Training`、`PublicTest` 和
`PrivateTest` 的官方含义。不要把 CSV 或解压后的图片加入 Git。

## 5. Preflight 检查

preflight 会检查环境、CSV、split、一个 train/validation batch、一次随机初始化
ResNet-18 forward 和输出目录；它不会训练或保存 checkpoint。

```bash
cd "${FER_PROJECT}"
python -m occlusion_fer.preflight \
  --config configs/fer2013_resnet18_clean.yaml \
  --data-path "${FER_DATA}/fer2013.csv" \
  --output-dir "${FER_OUTPUT}/preflight" \
  --device cuda \
  --batch-size 8 \
  --max-train-samples 32 \
  --max-validation-samples 16
```

只有末尾出现 `PREFLIGHT PASSED` 才继续。

## 6. 训练入口

先运行一次小样本、单 epoch GPU smoke test，不要直接开始正式实验：

```bash
python -m occlusion_fer.train \
  --config configs/fer2013_resnet18_clean.yaml \
  --data-path "${FER_DATA}/fer2013.csv" \
  --output-dir "${FER_OUTPUT}/clean-smoke" \
  --epochs 1 \
  --max-train-samples 512 \
  --max-validation-samples 128
```

确认日志显示 `selected_device=cuda`、loss 为有限数，并生成
`${FER_OUTPUT}/clean-smoke/best.pt`。这仍然不是正式实验。

## 7. 常见问题

- **`CUDA was requested but is not available`**：确认当前会话已分配 GPU，并重新
  检查 `nvidia-smi`、PyTorch 安装和 `torch.cuda.is_available()`。
- **Kaggle 返回 401/403**：检查 `kaggle.json`、文件权限和 Kaggle 账号访问权限；
  不要在终端或日志中打印凭据内容。
- **提示数据路径仍是 placeholder**：必须通过 `--data-path` 指向仓库外的真实
  `fer2013.csv`，不要修改提交到 Git 的 YAML 为个人绝对路径。
- **CSV 解析失败**：确认表头、2304 个像素、0–6 标签和三个官方 split；不要
  静默跳过错误样本。
- **GPU 显存不足**：先减小 batch size；不要在失败状态下开始正式训练。
- **预训练权重下载失败**：检查服务器网络或缓存，不要静默改成随机初始化后
  冒充正式 ResNet-18 结果。

任何一步失败都应停止并保存完整错误信息。详细排错和正式实验边界见
[`docs/server_runbook.md`](docs/server_runbook.md)。
