# SITNE-Walk-BX 输出目录合同

## 1. 根目录和不可覆盖规则

所有动态实验输出只能写入：

```text
08_results/sitne_bx/<run_id>/
```

`run_id` 必须唯一。代码和人工流程都不得覆盖已有 run directory、checkpoint、
prediction、metric 或 manifest。失败重试使用新 `run_id`，并在回报中关联旧运行。

禁止把动态结果写入以下位置：

- `02_data_canonical/**`
- `04_data_audit/**`
- `06_code/**`
- `10_reproducibility/dataset_bx_v1/**`

## 2. 阶段状态

运行目录遵循单向状态机：

```text
train_complete -> selection_sealed -> test_complete -> run_verified
```

后续状态只能追加新文件，不能修改已经冻结的配置、训练 provenance、checkpoint
内容或前一阶段 prediction。每个阶段都要验证前置文件和哈希，缺失时 fail-fast。

## 3. 必要输出类别

以 CLI 实际生成的文件名为准，但完整运行必须包含以下类别；`verify-run` 是最终
机器判定者。

### Train

- config snapshot：实际解析后的完整配置；
- provenance：Python、PyTorch、CUDA、GPU、代码和数据 SHA-256；
- checkpoints：每个候选 checkpoint 及其 epoch/step 对应关系；
- training history：loss 和训练过程记录；
- train completion marker：证明训练正常完成，而不是中途目录。

### Select

- Validation prediction：每个 Validation pair 恰有一条预测；
- Validation metrics：至少包含 AUPRC 和阈值选择所需数据；
- sealed selection manifest：选中 checkpoint、checkpoint SHA-256、MCC 阈值、
  配置/data/code hash 和确定性并列规则结果。

### Test

- Test prediction：52,048 条唯一 pair prediction，不缺失、不重复；
- Test metrics：使用 sealed threshold 计算，不重新选择阈值；
- Test provenance：selection manifest、checkpoint、配置、代码和数据 hash 链。

### Verify-run

- 完整性验证结果或 completion marker；
- 明确记录 prediction 行数、唯一 pair 数、finite score/metric 检查和所有哈希状态。

## 4. Prediction 表最低字段合同

Validation/Test prediction 至少应能唯一识别和审计：

- split；
- 原始或 canonical pair identifier；
- 两个 protein identifier；
- binary label；
- 连续 prediction score/probability；
- sealed threshold 下的 predicted label；
- run ID、checkpoint identity 或其可追溯引用。

无论实际列名如何，pair 的无向 canonicalization 必须唯一；同一 split 不能同时
出现 `(i,j)` 与 `(j,i)` 两条记录。score 必须 finite。

## 5. 归档和回传

实验结束后不得把动态 `08_results` 重新混入原始服务器准备 ZIP。应单独归档每个
已验证 run directory，并为结果归档生成独立 SHA-256。数据分析人员必须同时
收到实际配置、运行 provenance、selection manifest 和 CUDA acceptance 日志；
只提供聚合指标不足以开展可复核分析或论文写作。
