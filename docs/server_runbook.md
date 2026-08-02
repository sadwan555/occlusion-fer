# Ubuntu 服务器预检与 GPU 冒烟训练操作说明

## 目的和边界

本说明用于在老师的 Ubuntu 电脑上确认项目、FER2013 CSV、Python 环境、
PyTorch GPU、输出目录和最小训练链路可以工作。这里的 GPU 运行只是冒烟
检查，不是正式实验。完成干净数据基线之前，不开始遮挡、mixed training 或
最终结果统计。

所有操作只使用自己的账号和自己的 `${HOME}` 目录。不使用管理员权限，不改
系统 Python，不改全局 CUDA 配置，不进入或修改其他用户的目录，也不把数据、
模型权重或 checkpoint 放进 Git。任何一步报错都先停止并保存完整输出，不要
在不清楚影响范围时继续尝试。

## 推荐目录

统一使用下面的个人目录：

```bash
export FER_WORK_ROOT="${HOME}/anson-fer"
export FER_PROJECT_PARENT="${FER_WORK_ROOT}/project"
export FER_PROJECT_ROOT="${FER_PROJECT_PARENT}/occlusion-fer"
export FER_DATA_ROOT="${FER_WORK_ROOT}/data/raw"
export FER_DATA_CSV="${FER_DATA_ROOT}/fer2013.csv"
export FER_ENV_ROOT="${FER_WORK_ROOT}/envs/occlusion-fer"
export FER_OUTPUT_ROOT="${FER_WORK_ROOT}/outputs"
```

项目、数据、环境和输出互相分开：

- 项目：`${HOME}/anson-fer/project/occlusion-fer`
- 原始数据：`${HOME}/anson-fer/data/raw/fer2013.csv`
- 虚拟环境：`${HOME}/anson-fer/envs/occlusion-fer`
- 输出：`${HOME}/anson-fer/outputs`

不要把这些服务器绝对路径写回仓库里的 YAML；运行时使用命令行覆盖。

## 1. 只读检查服务器

先依次运行：

```bash
whoami
pwd
uname -a
nvidia-smi
df -h
git --version
python3 --version
```

检查含义：

- `whoami`：确认正在使用自己的账号。
- `pwd`：确认当前位置，不要误入其他用户目录。
- `uname -a`：记录 Ubuntu 和内核信息。
- `nvidia-smi`：确认 NVIDIA 驱动、GPU 名称、显存和当前占用；命令不存在或
  报错时立即停止，请老师或管理员确认服务器状态。
- `df -h`：确认个人目录所在磁盘有足够空间。
- `git --version`、`python3 --version`：确认基础工具可用；本项目要求
  Python 3.10 或更高版本。

## 2. 创建个人目录

```bash
mkdir -p "${FER_PROJECT_PARENT}"
mkdir -p "${FER_DATA_ROOT}"
mkdir -p "${FER_WORK_ROOT}/envs"
mkdir -p "${FER_OUTPUT_ROOT}"
```

## 3. 获取项目

只有在服务器已经配置好该仓库的 SSH 访问时才运行：

```bash
cd "${FER_PROJECT_PARENT}"
git clone git@github.com:sadwan555/occlusion-fer.git
cd "${FER_PROJECT_ROOT}"
git status
git log --oneline --decorate -n 3
```

如果 SSH 访问没有配置好，不要把账号口令、令牌或私钥粘贴到终端、文档或
仓库。先和老师确认允许的仓库访问方式，再继续。

应确认当前分支为 `main`、工作区干净，并且本地提交和远端同步。

## 4. 创建隔离 Python 环境

```bash
python3 -m venv "${FER_ENV_ROOT}"
source "${FER_ENV_ROOT}/bin/activate"
python --version
python -m pip --version
python -m pip install --upgrade pip
```

后续每次重新登录服务器，都先重新设置本说明开头的目录变量，再激活这个
虚拟环境。

## 5. 安装匹配服务器的 PyTorch 和项目

先再次查看 `nvidia-smi` 的驱动和 GPU 信息。不要猜测 CUDA wheel 地址或版本。
在实际安装当天打开 PyTorch 官方
[Start Locally](https://docs.pytorch.org/get-started/locally/) 页面，选择 Linux、
Pip、Python 和该服务器适用的 CUDA 计算平台，然后只在当前虚拟环境中运行
页面生成的命令。`torch` 与 `torchvision` 必须作为兼容的一对安装。

安装 PyTorch 后先检查：

```bash
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available())"
```

然后安装项目和现有测试依赖。当前项目的可选依赖名是 `test`：

```bash
cd "${FER_PROJECT_ROOT}"
python -m pip install -e ".[test]"
python -m pip check
```

如果版本解析失败、`pip check` 报冲突，或 `torch.cuda.is_available()` 不是
`True`，立即停止并保存输出。

## 6. 放置 FER2013 CSV

由老师或数据负责人按许可方式把原始 FER2013 CSV 放到：

```text
${HOME}/anson-fer/data/raw/fer2013.csv
```

数据必须位于仓库外。不要通过本项目编写下载器，不要把 CSV 或解压后的图片
复制到项目目录。放好后只检查文件存在和大小：

```bash
ls -lh "${FER_DATA_CSV}"
```

## 7. 运行服务器 preflight

先运行有限样本预检。它会统计完整 CSV，但只用有限样本建立 DataLoader 和做
一次前向；它不会训练、下载预训练权重或保存 checkpoint。

```bash
cd "${FER_PROJECT_ROOT}"
set -o pipefail
python -m occlusion_fer.preflight \
  --config configs/fer2013_resnet18_clean.yaml \
  --data-path "${FER_DATA_CSV}" \
  --output-dir "${FER_OUTPUT_ROOT}/preflight" \
  --device cuda \
  --batch-size 8 \
  --max-train-samples 32 \
  --max-validation-samples 16 \
  2>&1 | tee "${FER_OUTPUT_ROOT}/preflight.log"
```

只有末尾出现 `PREFLIGHT PASSED` 才能继续。还应看到：

- `selected_device=cuda`；
- 正确的 CUDA 可用状态和 GPU 名称；
- 完整 CSV 的总数、Training、PublicTest、PrivateTest 数量；
- 每个 split 的七类计数，缺类只会作为独立 warning 显示；
- train/validation 批次形状为 `[N, 3, 112, 112]`；
- float32 图像、int64 标签、有限数值和样本 ID；
- `(N, 7)` 的有限 logits；
- `model_forward=PASS` 和 `output_directory=PASS`。

## 8. 运行一次 1 epoch GPU 冒烟训练

预检全部通过后，才运行一次小样本、单 epoch 训练。配置中的 `device: auto`
应在该服务器选择 CUDA：

```bash
export FER_SMOKE_OUTPUT="${FER_OUTPUT_ROOT}/clean-smoke"
cd "${FER_PROJECT_ROOT}"
set -o pipefail
python -m occlusion_fer.train \
  --config configs/fer2013_resnet18_clean.yaml \
  --data-path "${FER_DATA_CSV}" \
  --output-dir "${FER_SMOKE_OUTPUT}" \
  --epochs 1 \
  --max-train-samples 512 \
  --max-validation-samples 128 \
  2>&1 | tee "${FER_OUTPUT_ROOT}/clean-smoke.log"
```

这仍然是 `SMOKE TEST — NOT A FORMAL EXPERIMENT`。检查：

- `selected_device=cuda`，并记录 GPU 名称；
- 实际训练样本为 512 或训练 split 的实际较小值；
- 实际验证样本为 128 或验证 split 的实际较小值；
- train/validation loss 都是有限数；
- validation accuracy 在 0 到 1 之间；
- 日志没有显存不足、CSV 解析或设备错误；
- 生成了 `best.pt`。

只查看 checkpoint，不移动、不删除：

```bash
ls -lh "${FER_SMOKE_OUTPUT}/best.pt"
```

## 9. 必须停止的情况

出现下列任一情况就停止，不要继续训练：

- `nvidia-smi` 不可用，或预检没有选择 CUDA；
- Python、torch、torchvision 版本不兼容；
- `pip check` 报依赖冲突；
- FER2013 路径错误、CSV 为空、split 为空、像素或标签解析失败；
- 批次形状、dtype、有限性或 logits 检查失败；
- 输出目录不能创建、写入、读取或清理临时标记；
- GPU 显存不足，loss 为 NaN/Inf，accuracy 超出 0 到 1；
- 仓库不干净、提交不匹配，或出现未知文件；
- 任何密钥、真实数据、权重或输出进入 Git 工作区。

## 10. 本阶段禁止做的事

本次服务器检查不运行正式三种子实验，不运行 PrivateTest 最终评估，不生成
正式表格，不实现或运行遮挡、mixed training、macro-F1、confusion matrix，
不改模型、依赖或仓库配置，也不清理系统文件、其他用户文件或服务器软件。

## 11. 成功后的保存与下一步

保留这三类材料即可：

- `${FER_OUTPUT_ROOT}/preflight.log` 和
  `${FER_OUTPUT_ROOT}/clean-smoke.log`；
- 当前虚拟环境的版本记录；
- `${FER_SMOKE_OUTPUT}/best.pt` 冒烟 checkpoint。

保存环境版本：

```bash
python -m pip freeze > "${FER_OUTPUT_ROOT}/environment.txt"
```

完成后停止。下一步只规划正式的 clean baseline；在审核服务器预检和 GPU 冒烟
结果之前，不启动长时间正式训练。
