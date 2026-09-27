# RESULTS FOR MANUSCRIPT — 实验重跑最终结果

> 生成日期：2026-09-06
> 依据：`rerun_submission_audit/` 全套重跑产物（formal bfloat16 + fail_fast，non_finite=0）

## 1. 哪些原稿数字被确认

| 指标 | 原稿 | 重跑 | 判定 |
|---|---|---|---|
| SITNE-Walk filtered MRR | 0.9806 ± 0.0008 | 0.980752 ± 0.000835（15 runs） | ✓ 确认 |
| Frequency prior MRR | 0.9734 | 0.973369（5 折逐位一致） | ✓ 确认 |
| metapath2vec MRR | 0.9718 | 0.971738 ± 0.000521（3 seeds） | ✓ 确认 |
| Degree / Binary tie MRR | 0.4085 | 0.408492 | ✓ 确认 |
| Macro relation MRR | 0.806 | 0.806113 | ✓ 确认 |
| Coverage / unsupported | 98.9% / 1,495 | 98.9% / 1,495 | ✓ 确认 |
| Calibration MAE（sampler） | ≈0.004 | 0.0043（rank α=0.25） | ✓ 确认 |

## 2. 哪些数字发生变化

| 指标 | 原稿 | 重跑 | 原因 |
|---|---|---|---|
| ComplEx MRR | 0.5428 | 0.538571 ± 0.008017（3 seeds） | 原 1 seed → 3 seeds |
| DistMult MRR | 0.5405 | 0.542791 ± 0.005474（3 seeds） | 原 1 seed → 3 seeds |
| no_rank ablation MRR | 0.4634 | 0.5320（高方差 0.34–0.71） | bfloat16 无 NaN-skip，且 no_rank 本身高方差 |
| 训练精度 | float16 | bfloat16 | **numerically repaired rerun**（见 §6） |

## 3. 哪些原结论仍成立

1. **ranking loss 是唯一主导组件**：no_rank Δ=−0.449（micro 0.981→0.532，macro 0.806→0.569）。
2. **degree correction 无性能贡献**：alpha0 Δ=−0.00001（≈full）。
3. **coarse 4-type 下 frequency prior 接近天花板**：prior 0.9734，SITNE-Walk 仅 +0.007。
4. **degree/binary 为 relation-invariant tie floor**（0.4085，与 degree-feature classifier 0.9734 彻底区分）。

## 4. 哪些结论必须降级

1. **"方法优越性"表述**：SITNE-Walk vs frequency prior 的 overall micro Δ 仅 +0.007，且 comparator
   tuning budget 不一致（typed skip-gram 50 trials/fold vs metapath2vec 固定）。应降级为
   "reference implementations under a common evaluator, not tuning-matched algorithm superiority"。
2. **no_rank 精确数值**：因高方差（0.34–0.71），应以区间而非单点报告。

## 5. 哪些结论必须删除

- 无需要删除的结论。历史负结果（C-4.2-2 / C-4.4-3 / C-4.5-2）保留，但数值以本次重跑为准。

## 6. 哪些新增结果可进入主文

1. **数值修复**：formal_v1 的 77/120 epoch NaN 根因是 float16 梯度溢出（epoch1 step1465 实测）；
   formal_v2 改 bfloat16 后 15 runs 全部 non_finite=0。此为主线可信度修复。
2. **enzymatic 的 delta-vs-prior = +0.337**：这是本次最重要的正向发现。SITNE-Walk 对 rare relation
   `enzymatic`（prior 仅 0.332）提升最大（formal 0.668），而对 frequent `physical`（prior 已 1.0）
   无提升（−0.006）。说明模型价值集中在 low-frequency relation，而非被 overall micro Δ=+0.007 掩盖。
3. **relation-wise 全景**：enzymatic 0.668 / general 0.711 / physical 0.994 / spatial 0.851（macro 0.806）。
4. **source confounding 量化**：enzymatic 100% 来自 IntAct（exclusive），physical/general/spatial
   为 BioGRID 主导（74–92%）→ SOURCE-CONFOUNDING MATERIAL（需在 limitation 中声明）。
5. **unfiltered sensitivity（filtered 仍为 primary）**：filtered ranking 保持 primary protocol；
   unfiltered 仅作 sensitivity，因为 metapath2vec/ComplEx/DistMult/typed skip-gram 无 raw scores，无法统一生成 unfiltered。
   unfiltered 对 enzymatic 影响显著（filtered 0.668 → unfiltered 0.423），对 physical 影响小（0.994 → 0.990），
   说明 filtered 屏蔽其他 known-positive types 对 multi-label minority relation 的难度影响明显。
6. **degree-feature classifier 与 frequency prior 的关系**：二者逐 query `filtered_rank`/`unfiltered_rank`
   100% 一致（5/5 folds），但 raw scores 不同。严谨表述为：
   > The degree-feature classifier induces the same query-level ranking as the train-frequency prior on this benchmark.
   不应表述为「退化为 frequency prior」（内部函数不同，仅排序行为不可区分）。

## 7. 哪些只适合放 Supplementary

- 完整 sensitivity strata（degree strata 0.974→0.987 随 degree 递增；frequency strata 0.696→0.983）。
- pair-multiplicity（single-type 0.981 vs multi-type 0.972）。
- comparator fairness matrix（tuning/seed 差异明细）。
- rank-form calibration 完整指标（MAE/R²/TV/Spearman）。
- provenance / 工程 hash 信息（`09_provenance/`）。

## 附：核心数字速查

```text
SITNE-Walk      micro 0.9808  macro 0.8061  (15 runs, bfloat16, non_finite=0)
frequency prior 0.9734   metapath2vec 0.9717   ComplEx 0.5386   DistMult 0.5428
degree/binary tie 0.4085
ablation: no_rank 0.532 (Δ-0.449) | 其余 ≈ full (Δ<0.0002)
enzymatic delta-vs-prior +0.337 | physical -0.006
```
