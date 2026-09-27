# SITNE-Walk-BX

本包是与原 `sitne_walk` 完全并列的 Dataset B/B* 归纳式实现。它不会修改或
导入原 transductive typed-ranking core。

核心边界：

- Dataset B 的全部 Train pair 进入 binary BCE；
- binary-only 与 typed 模式的 Train pair/sequence 都固定读取 B* 七项 allow-list
  中的 `views/X/train.tsv`、`nodes/train_protein.tsv`；binary-only 不打开 typed
  edges，训练期任何模式均不打开 B 的全量序列表；
- Train-positive 同时进入 positive-only topology auxiliary，不合成未知负 PPI；
- B* typed auxiliary 只使用七项 Train allow-list，对无安全 typed fact 的正例
  使用 mask，但这些正例仍进入 binary/topology loss；
- `select` 只读取 Validation；`test` 只有在 selection manifest sealed 且所有
  hash 匹配后才可读取 Test；
- B* `targets/**` 与 held-out views 默认拒绝，`select` 始终无权读取；runner
  完整验证 sealed selection 后才签发绑定 TEST/VERIFY 的进程内 capability。
  binary pipeline 本身仍不读取这些 typed targets；冻结选择后的 coverage/
  open-set 分析可调用 `create_post_selection_guard`；
- 正式配置显式 `device: cuda`，CUDA 不可用立即失败。

CLI：

```bash
python 06_code/run_sitne_bx.py inspect --config 06_code/configs/sitne_bx.fine_fp32.yaml
python 06_code/run_sitne_bx.py train --config 06_code/configs/sitne_bx.fine_fp32.yaml
python 06_code/run_sitne_bx.py select --config 06_code/configs/sitne_bx.fine_fp32.yaml --run-dir RUN_DIR
python 06_code/run_sitne_bx.py test --config 06_code/configs/sitne_bx.fine_fp32.yaml --run-dir RUN_DIR
python 06_code/run_sitne_bx.py verify-run --config 06_code/configs/sitne_bx.fine_fp32.yaml --run-dir RUN_DIR
```

`precomputed_embedding` 后端要求分别提供 train/validation/test 三个 NPZ；每个
文件必须且只能包含一维 `accessions` 和二维 `embeddings`，并使用
`allow_pickle=False` 读取。三个路径必须互异，且每个 NPZ 的 accession 集合
必须与当前阶段所需蛋白集合完全相等，多一个或少一个都会 fail-fast。
