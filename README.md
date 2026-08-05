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

## Clean baseline 论文图表

`occlusion_fer.paper_figures` 从三个正式 clean-only run 生成可追溯的
FER2013 PublicTest/validation 图表。它不读取 checkpoint、逐样本图像或 smoke
test，不修改输入 run，也不执行训练或 PrivateTest 评估。

安装独立的论文绘图依赖；需要运行测试时再额外安装 test 依赖：

```bash
python -m pip install -e '.[paper]'
python -m pip install -e '.[test]'
```

入口必须显式接收三个 run、包含官方 split 计数的 preflight 日志和一个位于正式
run 之外的输出目录：

```bash
python -m occlusion_fer.paper_figures \
  --run-dir /path/to/clean-seed42 \
  --run-dir /path/to/clean-seed123 \
  --run-dir /path/to/clean-seed2026 \
  --preflight-log /path/to/preflight.log \
  --output-dir /path/to/paper-figures
```

每个 run 的直接输入是：

- `run_metadata.json` 与 `resolved_config.yaml`：seed、run 状态、训练模式和实验
  commit；
- `history.csv`：原始 30-epoch train/validation 曲线，不进行平滑、插值或拟合；
- `validation/best_metrics.json`：clean validation 总体指标、类别顺序和内嵌
  confusion matrix；
- `validation/best_per_class_metrics.csv`：七类 precision、recall、F1 和 support；
- `validation/best_confusion_matrix.csv`：真实标签行、预测标签列的原始计数矩阵。

类别分布只解析显式传入 preflight 日志的 `train_samples`、
`validation_samples`、`train_class_counts` 和 `validation_class_counts`。脚本严格
验证 seeds 为 `42/123/2026`、`condition=clean`、`split=validation`、固定类别
顺序、PublicTest 3,589 个样本及所有 CSV/JSON 之间的一致性。缺失、错序、NaN、
Inf、总数或矩阵不一致都会在创建输出前失败。

跨 seed 的标准差使用 `pandas.Series.std(ddof=1)`，即 sample standard
deviation。混淆矩阵先分别对每个 seed 按真实类别行归一化，再逐单元格求三 seed
算术均值；不会把 10,767 次跨 seed 推断描述成相互独立的样本。类别分布柱高为
各 split 内百分比，柱顶保留原始 count。

四组图均生成 PDF、SVG 和 300-dpi PNG：

```text
clean_training_validation_curves.{pdf,svg,png}
clean_per_class_f1.{pdf,svg,png}
clean_validation_confusion_matrix.{pdf,svg,png}
fer2013_training_publictest_distribution.{pdf,svg,png}
```

同时生成：

```text
summary_metrics.csv
per_class_f1_summary.csv
mean_normalized_confusion_matrix.csv
class_distribution.csv
generation_manifest.json
```

`generation_manifest.json` 记录输入路径及 SHA-256、正式实验 commit、软件版本、
完整命令、统计规则和输出列表。重新生成后应核对 manifest 的
`split=validation`、`class_distribution_splits=[Training, PublicTest]`、
`condition=clean` 与 `smoke_test_read=false`。

这些图只支持 clean-only PublicTest/validation baseline 分析。它们不是
PrivateTest/final-test 结果，不包含合成遮挡或 mixed training，也不能用于回答
RQ1–RQ3、宣称 SOTA、真实场景鲁棒性，或推断人的真实内在情绪和认知状态。

## Overall experimental framework

`occlusion_fer.paper_framework` 生成 Section 3.1 使用的横版整体实验框架图。该图
采用从左到右的模块化布局，描述 FER2013 官方 split、图像预处理、两种训练策略、
共享 backbone 架构、10 个统一评价条件和评价输出，不读取实验结果，也不包含
结果数值。

```bash
python -m occlusion_fer.paper_framework \
  --output-dir /path/to/paper-framework
```

输出包括一份可编辑 SVG、一份矢量 PDF 和一份 300-dpi PNG：

```text
overall_experimental_framework_landscape.svg
overall_experimental_framework_landscape.pdf
overall_experimental_framework_landscape.png
```

图中保留 Training、PublicTest 和 PrivateTest 的官方角色：Training 用于模型拟合，
PublicTest 用于验证和 checkpoint selection，PrivateTest 仅用于最终评价。两种训练
策略使用相同的 ImageNet-pretrained ResNet-18 架构，但分别训练；随后使用相同的
1 个 clean 与 9 个 occluded conditions 评价。
