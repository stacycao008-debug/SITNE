# Dataset B/B* reproducibility index

本目录只保存 SITNE-Walk-BX 服务器部署、运行回报和证据治理文档。推荐阅读顺序：

1. `SERVER_RUNBOOK_CN.md`
2. `OUTPUT_CONTRACT.md`
3. `EXPERIMENT_RETURN_REPORT_TEMPLATE.md`
4. `PAPER_EVIDENCE_BOUNDARY.md`
5. `MANIFEST_POLICY.md`

当前冻结交付和服务器 ZIP 均包含 `BUNDLE_SHA256SUMS.txt`。该文件由打包器生成、
排除自身以避免递归哈希，不应手工创建或编辑。原 188 项
`manifests/EXPORT_SHA256SUMS.txt` 只用于证明升级前基线未变；服务器 ZIP 排除了
其中的历史结果和缓存，因此解压验收必须使用新的 bundle manifest。

> **快照漂移注记（2026-08-23 审计核实）**：本 bundle 冻结时点为 2026-08-05。此后工作区核心代码已演进（`06_code/sitne_walk/{cli,config,graph,trainer}.py` 于 8-11 修改），`07_experiments/` 亦已归档——因此对当前工作区执行 `sha256sum -c BUNDLE_SHA256SUMS.txt` 会出现 5 项 FAILED + 47 项缺失，**属预期**。bundle 作为 8-5 冻结交付仍然有效；但不得把当前工作区代码视为与 bundle 等价，任何新导出必须重建清单。
