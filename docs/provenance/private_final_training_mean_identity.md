# PrivateTest Final Training Mean Fix Report

## 1. Root Cause

旧 PrivateTest preflight 将 `TRAINING_MEAN_ARTIFACT_SHA256` 误解释为最终 `training_mean_v2.json` 文件的 whole-file SHA-256。正式服务器 artifact 因此会在 inference 前被错误拒绝。

## 2. Stage 8 SHA Semantics

Stage 8 的 frozen Training mean v2 identity 是：

`SHA256(canonical compact JSON of TrainingMeanV2Envelope + trailing LF)`

该 envelope 不包含 `artifact_sha256` 字段。正式 identity 保持为：

`02ff7194f653c0aa6c53a042beca4b44eac58c12cb2737a6e25e0ccbe79fc1d9`

loader 还要求最终 JSON 为 canonical compact UTF-8 JSON，且恰好包含一个 trailing LF；embedded `artifact_sha256` 必须等于重新计算的 envelope identity。

## 3. Why 02ff Differs From Final File Byte SHA

writer 在计算 `02ff...` 后，将该值作为 `artifact_sha256` 字段加入最终 JSON。因此最终文件包含一个 envelope identity 中不存在的自描述字段，其 byte SHA 是：

`becf171033a87fca4445b1081dcc656fbe6dba2c098c38ad1b06e4e20f279295`

两者不同是正式设计行为，不代表 artifact 损坏。manifest sidecar、plan、condition provenance 和 preflight 继续使用 `02ff...`，没有改为 whole-file SHA。

## 4. Files Changed

- `src/occlusion_fer/training_mean.py`：加入 Stage 8 v2 envelope、canonical bytes、SHA、loader 与 semantic validator。
- `src/occlusion_fer/private_preflight.py`：移除 whole-file SHA 检查，改为 canonical loader、embedded/envelope identity、Training dataset SHA、224 consumer、fill domain、algorithm 和 raw mean 验证。
- `tests/test_private_preflight.py`：增加正式结构与篡改/非 canonical artifact 回归测试。

没有改变 224 x 224、occlusion-v2-224、mask seed 20260804、Training raw mean、normalized fill、checkpoint SHA、PrivateTest canonical SHA、conditions 或 geometry。

## 5. Regression Tests

TDD 红灯确认旧实现拒绝正式 fixture。回归覆盖：

- 正式 canonical artifact 被接受。
- 最终 JSON byte SHA 为 `becf...` 且不同于 `02ff...` 时仍通过。
- embedded SHA 篡改时失败。
- raw mean 篡改时失败。
- Training dataset SHA 篡改时失败。
- 缺少 trailing LF 时失败。
- canonical envelope 加 LF 的 golden SHA 等于 `02ff...`。

clean worktree 目标命令：

```text
PYTHONPATH=src pytest -q tests/test_training_mean.py tests/test_formal_checkpoints.py tests/test_private_aggregate.py tests/test_private_final.py tests/test_private_manifest.py tests/test_private_plan.py tests/test_private_preflight.py
119 passed in 1.52s
```

## 6. Full Test Result

从新 exact commit 创建 detached clean worktree 后：

```text
PYTHONPATH=src pytest -q
800 passed, 2 skipped in 34.58s

python3 -m compileall -q src tests
PASS

git diff --check
PASS

private preflight/training mean imports
PASS

git status --porcelain --untracked-files=all
no output (clean)
```

## 7. Old Commit

`43afaa7a3fe6d588d8e3100d4025e2d6a1095bc9`

## 8. New Exact Commit

branch：`private-final-eval-v2`

commit：`7e154aca1e95ef78ea7e3bc8767bcb21ca769335`

message：`fix: validate Stage 8 training mean artifact identity`

## 9. GitHub Push Status

SUCCESS。使用普通非 force push 更新 `origin/private-final-eval-v2`。local、origin tracking ref 与 GitHub `ls-remote` 均为：

`7e154aca1e95ef78ea7e3bc8767bcb21ca769335`

## 10. PrivateTest Safety

Real PrivateTest inference = NO。

没有生成真实 PrivateTest manifest、plan、predictions 或 metrics，也没有运行真实 PrivateTest preflight 或 evaluator。
