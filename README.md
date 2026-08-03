# Occlusion FER

一个可复现的 PyTorch 科研项目：在 FER2013 上训练 ImageNet 预训练的
ResNet-18，并逐步研究合成面部遮挡对七分类性能的影响。

本项目预测 FER2013 提供的数据集标签。实验结果不能证明模型识别了人的真实
内在情绪、困惑、理解程度、参与度或学习结果，也不用于宣称真实场景鲁棒性、
跨数据集泛化或最先进性能。

## 当前阶段

当前代码完成了 clean baseline 的可复现训练与评估基础：

- `Training` 用于训练；
- `PublicTest` 用于逐 epoch 验证，并按 clean validation macro-F1 选择
  `best.pt`；
- `PrivateTest` 不会被训练入口加载，只能通过显式确认的最终评估入口访问；
- 固定七类顺序：angry、disgust、fear、happy、sad、surprise、neutral；
- 灰度图复制为三通道，缩放到 `112×112`，使用 ImageNet normalization；
- 保存 best/last checkpoint、训练历史、逐类指标、混淆矩阵和逐样本预测；
- 记录解析后的配置、Git 状态、软件版本、设备和失败信息。

遮挡生成、mixed clean/occluded training、十种最终评估条件和跨种子汇总属于
后续批准阶段，当前尚未实现。

## 项目结构

```text
configs/                  可移植 YAML 配置（数据路径保持 placeholder）
src/occlusion_fer/        数据、模型、训练、评估和产物代码
tests/                    自动测试
docs/server_runbook.md    完整 HIVE 部署、排错和实验规范
SERVER_RUN.md             五分钟服务器快速开始
```

数据、checkpoint、输出、日志、虚拟环境和密钥都必须留在 Git 仓库外。

## 主要入口

环境与数据预检：

```bash
python -m occlusion_fer.preflight \
  --config configs/fer2013_resnet18_clean.yaml \
  --data-path /path/to/fer2013.csv \
  --output-dir /path/to/preflight-output \
  --device cuda
```

clean 训练：

```bash
python -m occlusion_fer.train \
  --config configs/fer2013_resnet18_clean.yaml \
  --data-path /path/to/fer2013.csv \
  --output-dir /path/to/run-output \
  --seed 42 \
  --device cuda \
  --epochs 30 \
  --batch-size 128 \
  --num-workers 4 \
  --amp
```

只有在模型、超参数、遮挡 masks、mixed protocol 和 checkpoint 规则全部锁定
后，才把下面的 clean 命令作为最终评估批次的一部分运行；当前 clean 开发阶段
不要提前查看 PrivateTest：

```bash
python -m occlusion_fer.final_evaluate \
  --config configs/fer2013_resnet18_clean.yaml \
  --checkpoint /path/to/run-output/best.pt \
  --data-path /path/to/fer2013.csv \
  --output-dir /path/to/run-output \
  --device cuda \
  --batch-size 128 \
  --num-workers 4 \
  --amp \
  --confirm-private-test
```

服务器上的完整顺序、三种子命令、输出解释和故障处理见
[`docs/server_runbook.md`](docs/server_runbook.md)。

## 论文可用产物

每个正式 run 的主要文件包括：

- `resolved_config.yaml`：Methods 中的模型、数据处理与训练设置；
- `run_metadata.json`：Git、软件、设备、种子和运行状态；
- `history.csv`：绘制 loss、accuracy、macro-F1 与训练吞吐曲线；
- `validation/best_*`：说明 checkpoint 选择依据；
- `final_test/clean_metrics.json`：锁定后的 clean test 总体结果；
- `*_per_class_metrics.csv`：逐类结果表或柱状图；
- `*_confusion_matrix.csv`：混淆矩阵图；
- `*_predictions.csv`：配对条件比较、错误分析和可追溯样本结果。

这些文件提供论文图表的原始证据；不要手工改写输出数值，也不要只报告表现
最好的 seed。
