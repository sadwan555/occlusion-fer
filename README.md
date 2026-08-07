# Occlusion FER

一个可复现的 PyTorch 科研项目：在 FER2013 上训练 ImageNet 预训练的
ResNet-18，并逐步研究合成面部遮挡对七分类性能的影响。

本项目预测 FER2013 提供的数据集标签。实验结果不能证明模型识别了人的真实
内在情绪、困惑、理解程度、参与度或学习结果，也不用于宣称真实场景鲁棒性、
跨数据集泛化或最先进性能。

## 当前阶段

当前权威实验协议见
[`docs/experiment_protocol.md`](docs/experiment_protocol.md)。锁定边界如下：

- `Training` 用于训练；
- `PublicTest` 用于逐 epoch 验证，并按 clean validation macro-F1 选择
  `best.pt`；
- `PrivateTest` 尚未访问，只能在六个正式 checkpoint、v2 masks 和报告计划全部
  锁定后进行一次最终评估；
- 固定七类顺序：angry、disgust、fear、happy、sad、surprise、neutral；
- E7 灰度图复制为三通道，bilinear 缩放到 `224×224`，使用 ImageNet normalization；历史 v1/112 代码和产物只作为开发记录；
- 当前遮挡协议为 `occlusion-v2-224`，仅允许 `upper_face`、`lower_face`、`random_rectangle` 和 `0.20`、`0.30`、`0.40`；
- Stage B 与 locked E7 baseline 共用 `Usage` 路由：CSV reader 会将整行词法读取为
  字符串字段，但对判定为 PrivateTest 的行只语义检查 `Usage`；其 emotion/pixels
  不解析为 label/image，不验证或物化为 record/tensor，也不进入 Training/PublicTest
  hash、mean、manifest、训练、验证、checkpoint 选择、mask 生成、评估或指标；
- 保存 best/last checkpoint、训练历史、逐类指标、混淆矩阵和逐样本预测；
- 记录解析后的配置、Git 状态、软件版本、设备和失败信息。

E0-E7 seed-2026 运行是配方筛选历史，不是正式三种子证据。三组 locked E7 clean
正式 checkpoint 已保留；Stage 8 只新增三组 mixed 正式训练。当前没有可报告的
Stage B 正式结果。mixed
clean/occluded training 与十种 PublicTest 条件 evaluator 只在 v2 artifact 身份
完整时启用；PrivateTest 不属于当前 Stage 6/7 验证范围。

## 项目结构

```text
configs/                  可移植 YAML 配置（数据路径保持 placeholder）
src/occlusion_fer/        数据、模型、训练、评估和产物代码
tests/                    自动测试
docs/experiment_protocol.md 当前唯一正式实验协议
docs/server_runbook.md    完整 HIVE 部署、排错和实验规范
docs/paper_figure_runbook.md 论文图表导出与 provenance 核验
scripts/paper/            只读转换正式产物的论文导出 CLI
SERVER_RUN.md             五分钟服务器快速开始
```

数据、checkpoint、输出、日志、虚拟环境和密钥都必须留在 Git 仓库外。

## 主要入口

环境与数据预检：

```bash
python -m occlusion_fer.preflight \
  --config configs/experiments/fer2013_resnet18_e7_clean.yaml \
  --data-path /path/to/fer2013.csv \
  --output-dir /path/to/preflight-output \
  --device cuda
```

clean 训练：

```bash
python -m occlusion_fer.train \
  --config configs/experiments/fer2013_resnet18_e7_clean.yaml \
  --data-path /path/to/fer2013.csv \
  --output-dir /path/to/run-output \
  --seed 42 \
  --device cuda \
  --epochs 50 \
  --batch-size 128 \
  --num-workers 4 \
  --amp
```

上述命令只展示锁定的 clean 配方；必须先完成干净 commit、HIVE/Linux 验证和
v2 artifact gate，才能启动 Stage 8。先从同一 official CSV 生成 v2 artifacts：

```bash
python -m occlusion_fer.stage_b_artifacts \
  --data-path /path/to/fer2013.csv \
  --output-dir /path/to/new-stage-b-artifacts
```

mixed 配置为
`configs/experiments/fer2013_resnet18_e7_occlusion_mixed.yaml`，它要求服务器本地
YAML 中的 combined CSV、Training mean 和 PublicTest manifest 路径均已解析并
通过 preflight。当前不要运行 `final_evaluate`，也不要访问 PrivateTest。

服务器上的完整顺序、三种子命令、输出解释和故障处理见
[`docs/server_runbook.md`](docs/server_runbook.md)。

论文数据示例、正式 v2 遮挡示例和六 run 跨 seed 汇总使用
`scripts/paper/export_fer2013_examples.py`、
`scripts/paper/export_occlusion_examples.py` 和
`scripts/paper/export_formal_results.py`。这些入口只生成论文辅助产物，不运行训练或
推理；完整参数、PrivateTest 边界和 provenance 核验见
[`docs/paper_figure_runbook.md`](docs/paper_figure_runbook.md)。

## 论文可用产物

每个正式 run 的主要文件包括：

- `resolved_config.yaml`：Methods 中的模型、数据处理与训练设置；
- `run_metadata.json`：Git、软件、设备、种子和运行状态；
- `history.csv`：绘制 loss、accuracy、macro-F1 与训练吞吐曲线；
- `validation/best_*`：说明 checkpoint 选择依据；
- `conditions/<condition>/<condition>_*`：同一 PublicTest 样本集合上的十条件指标；
- `conditions/*_paired_drop.csv`：clean 与对应遮挡条件的逐样本配对变化；
- `conditions/<condition>/<condition>_per_class_metrics.csv`：逐类结果表或柱状图；
- `conditions/<condition>/<condition>_confusion_matrix.csv`：混淆矩阵图；
- `conditions/<condition>/<condition>_predictions.csv`：配对条件比较、错误分析和可追溯样本结果。

这些文件只有在三组 locked clean 与 Stage 8 三组 mixed 组成的六个正式 run
通过 provenance 校验后才是论文候选证据；当前状态为 pending。不要手工改写
输出数值，也不要只报告表现最好的 seed，synthetic、smoke 和 screening 产物
不能作为正式结果。
