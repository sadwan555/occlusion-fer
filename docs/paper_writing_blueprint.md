# FER2013 遮挡鲁棒性论文写作与证据蓝图

## 1. 使用原则

本文件把当前 `essay.docx` 的论文框架与项目将产生的 JSON/CSV 证据对应起来。
它是写作路线图，不是实验结果。任何 `[待填]` 都必须由锁定协议下的真实输出
填充，不能根据预期、单个 seed 或 smoke test 编造。

当前方法口径以 [`experiment_protocol.md`](experiment_protocol.md) 为准。E0-E7
seed-2026 screening、occlusion-v1/112、synthetic 和 smoke 产物均不能填入正式
Results。Stage 8 尚未开始，所有正式结果位置保持 `[待填：Stage 8]`。

全文使用“FER2013 dataset-defined facial-expression label classification”等准确
表述。不要写模型识别了真实内在情绪、困惑、理解程度、参与度或学习效果；
online classroom 只能作为假设性的 HCI motivation，不能写成实验应用。

## 2. 推荐论文主线

核心问题保持简单、可回答且与第一版实验一致：

- **RQ1：** 合成遮挡相对 clean 输入会造成多大分类性能下降？
- **RQ2：** upper-face、lower-face、random-rectangle 以及 0.20、0.30、0.40
  ratio 的影响有何差异？
- **RQ3：** mixed clean/occluded training 是否在保持 clean 性能的同时减小
  clean-to-occluded performance drop？

第一版只比较同一个 ImageNet-pretrained ResNet-18 的 clean-only 与 mixed
clean/occluded 两种训练策略。不要加入第二 backbone、Transformer、attention、
landmark/Grad-CAM occlusion 或跨数据集结论来扩大论文范围。

## 3. 章节填充顺序

### Abstract

最后写。按“问题—方法—实验设计—主要量化结果—严格结论”五句结构：

1. FER 分类在合成遮挡下可能退化；
2. 使用 FER2013、固定 ResNet-18、三类遮挡和三种 ratio；
3. 比较 clean-only 与 mixed training，报告三 seed；
4. 从最终汇总表填入 accuracy、macro-F1 与 performance drop；
5. 只总结本数据集和本遮挡协议内的证据。

### 1. Introduction

建议结构：

1. FER2013 七分类任务与输入遮挡问题；
2. 为什么只看 clean accuracy 不足以描述遮挡敏感性；
3. 明确可控合成遮挡实验的价值与局限；
4. 列出 RQ1–RQ3；
5. 用项目真实交付物描述贡献，不宣称算法创新或 SOTA。

当前 `essay.docx` 中把表情分类延伸为 confusion、understanding、engagement 或
learning outcome 检测的句子应重写为研究动机，并加入“本实验不测量这些认知
状态”的边界说明。

### 2. Literature Review

按主题组织，不按论文逐篇罗列：

- FER2013 与基于 CNN/ResNet 的表情标签分类；
- facial occlusion 对 FER 的影响；
- 数据增强或 mixed training 的鲁棒性思路；
- 现有工作的数据、遮挡、评价或证据边界不足。

每项事实都需要后来补充的真实文献引用。本项目文件不能作为相关工作引用，
也不要预先填造参考文献、DOI 或性能数值。

### 3. Methodology

#### 3.1 Experimental environment and reproducibility

从每个 run 的 `run_metadata.json` 提取 Python、NumPy、PyTorch、Torchvision、
GPU、Git commit、dirty state 和 seed。三 seed 的 commit 与正式协议必须一致。

#### 3.2 Dataset and preprocessing

写明官方 split：Training 用于优化，PublicTest 用于验证和 checkpoint 选择，
PrivateTest 只用于锁定后的单次最终评估。描述 48×48 灰度像素缩放到 `[0,1]`、
bilinear resize 到 224×224、复制为三通道并使用 ImageNet normalization。类别
顺序固定为：

```text
angry, disgust, fear, happy, sad, surprise, neutral
```

不要把 PrivateTest 写成 validation，也不要声称随机重划分数据。

#### 3.3 Model and optimization

从 `resolved_config.yaml` 填写标准-stem ResNet-18、ImageNet pretrained、AdamW、
learning rate `0.0001`、weight decay `0.001`、label smoothing `0.1`、无 scheduler、
batch size `128`、`50` epochs、AMP 和 `4` 个 DataLoader workers。
best checkpoint 依据 clean PublicTest macro-F1 的严格提升，tie 保留更早 epoch。

#### 3.4 Occlusion protocol

只允许 upper_face、lower_face、random_rectangle 与 0.20、0.30、0.40。写明填充
值来自 Training split pixel mean，记录 target/actual ratio，固定显式 seed，
源图不原地修改，所有 checkpoint 使用相同 `occlusion-v2-224` PublicTest manifest
和 evaluation seed `20260804`。历史 v1/112 只能作为开发历史。

#### 3.5 Training strategies

比较 clean-only 与 mixed clean/occluded；二者保持 backbone、优化器、epochs、
batch size、base preprocessing、seed set 和 checkpoint rule 一致。small CNN 只做
pipeline sanity check，不能作为研究结果。

#### 3.6 Metrics

总体 accuracy：

```text
accuracy = correct predictions / all samples
```

七类等权 macro-F1：

```text
macro-F1 = (F1_0 + F1_1 + ... + F1_6) / 7
```

对每个 seed、策略和遮挡条件定义：

```text
performance drop = clean metric - occluded-condition metric
```

混淆矩阵行是真实标签、列是预测标签。正式结果必须同时报告 accuracy 与
macro-F1，不能因 disgust 样本少而删除类别。

### 4. Results

推荐严格按以下顺序写，避免先看测试结果再改变方法：

1. clean-only 与 mixed 三 seed 的 clean PublicTest checkpoint 选择证据；
2. clean-only checkpoint 在九个 PublicTest 遮挡条件下的表现和 drop；
3. mixed checkpoint 的同条件表现；
4. 两种策略的逐条件、逐类和跨 seed 对比；
5. 六个 checkpoint 和协议完全锁定后的单次 PrivateTest 结果；
6. 失败、异常与负面结果。

PublicTest 用于 checkpoint 选择与遮挡分析；PrivateTest 只用于最终确认。二者必须
分表呈现且不可混用。正式表必须来自可追溯原始产物，不能使用 screening 或
synthetic 数值。

### 5. Discussion

围绕 RQ1–RQ3 解释已经观察到的模式，区分“数据直接显示”与“可能原因”。讨论
FER2013 类别不均衡、合成遮挡与真实遮挡差异、单 backbone、固定分辨率、单数据集
和 seed 数量等限制。不要从分类分数推断人的认知状态。

### 6. Conclusion

逐条回答 RQ1–RQ3，给出最主要的量化结果与适用边界。未来工作可以提真实物体
遮挡、其他数据集或模型，但不能把它们写成当前完成内容。

## 4. 图表计划与数据来源

| 编号 | 推荐图表 | 直接输入 | 当前状态 |
|---|---|---|---|
| Figure 1 | 实验 pipeline 框架图 | 方法配置与 split 规则 | 可先画结构，结果节点待填 |
| Figure 2 | 三类遮挡 × 三 ratio 示例网格 | 固定 v2 mask 产物 | `[待填：Stage 8]` |
| Figure 3 | train/validation loss 与 validation macro-F1 曲线 | 六个正式 run 的 `history.csv` | `[待填：Stage 8]` |
| Figure 4 | clean confusion matrix | `conditions/clean/clean_confusion_matrix.csv` | `[待填：Stage 8]` |
| Figure 5 | 各遮挡条件 accuracy/macro-F1 与 clean drop | 正式跨 seed 汇总 CSV | `[待填：Stage 8]` |
| Figure 6 | 各类别在遮挡下的 F1 变化 | `conditions/*/*_per_class_metrics.csv` | `[待填：Stage 8]` |
| Table 1 | 数据、模型与训练设置 | formal `resolved_config.yaml`、metadata | 配置已锁定，运行证据待填 |
| Table 2 | clean 三 seed 与 mean±std | formal clean condition metrics | `[待填：Stage 8]` |
| Table 3 | clean-only 十条件结果 | formal condition metrics | `[待填：Stage 8]` |
| Table 4 | mixed 与 clean-only 的 drop 对比 | formal 汇总 CSV | `[待填：Stage 8]` |

所有图表保持固定类别顺序、固定 condition 顺序、相同坐标范围和统一色标。均值与
标准差必须基于全部 `42/123/2026`，图注说明 `n=3 seeds`，同时保留单 seed 原始
点，避免均值掩盖波动。

## 5. 当前文件到论文证据的映射

- `resolved_config.yaml` → Methods 3.2/3.3、Table 1；
- `run_metadata.json` → Methods 3.1、复现附录；
- `history.csv` → Figure 3、训练稳定性讨论；
- `validation/best_metrics.json` → checkpoint 选择说明；
- `conditions/<condition>/<condition>_metrics.json` → PublicTest 十条件结果与 Tables 2–4；
- `conditions/*_paired_drop.csv` → 配对 clean-to-occluded 分析；
- `conditions/*/*_per_class_metrics.csv` → 逐类表和 Figure 6；
- `conditions/*/*_confusion_matrix.csv` → Figure 4；
- `conditions/*/*_predictions.csv` → 配对 condition drop、置信度和错误案例审计；
- `failure.json` 与运行日志 → 失败实验记录，不作为成功结果隐藏。

## 6. 写作前检查清单

- 代码 commit 固定且正式运行时 `git_dirty=false`；
- 三个 seed 全部完成并全部报告；
- PrivateTest 只在六个 best checkpoint、九个遮挡条件和 final masks 全部锁定后
  运行一次；
- clean 与 mixed 使用相同正式超参数和 final masks；
- 每个表格数字能追溯到一个 JSON/CSV 文件；
- mean、std、drop 和图中数值由脚本生成并可复算；
- 没有使用 smoke test、best seed 或被覆盖结果冒充正式结果；
- 没有超出 FER2013、合成遮挡和单 backbone 的证据边界；
- 未完成项目明确标记为 future work，而不是写成完成内容。
