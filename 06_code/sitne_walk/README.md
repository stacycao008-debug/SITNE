# SITNE-Walk CUDA 实现

本目录实现算法草案中的 Shortcut-Invariant Typed Network Embedding with
Corrected Walks。代码基于 PyTorch，核心随机游走、embedding、decoder、GRL 和
训练损失均可在 CUDA 张量上运行；不依赖 PyG、DGL、CuPy 或自定义 `.cu` 扩展。

当前代码是**可测试的方法实现**，不是经验优越性、创新性或 unseen-protein
泛化的证据。

## 1. v1 冻结语义

现有清洗数据没有可靠的逐事实方向、source 或 detection-method 字段。为避免
推断不存在的语义，v1 固定如下：

1. `type_name` 是关系主键；冲突的 `type_id` 不参与编码。
2. protein pair 使用 `min(i,j), max(i,j)` canonicalization。
3. 完全相同的 canonical pair-type 事实按二值边合并。
4. 所有关系显式按 symmetric 处理；不从反向记录是否存在推断方向。
5. self-loop 保留在 typed ranking 监督中，默认不进入 random walk。
6. degree 是 train-only 唯一非自环邻居数。
7. type frequency 是实际进入 walk 的 train-only 唯一 canonical pair-type 数；
   默认排除 self-loop，显式允许 self-loop 游走时会同步计入。
8. `lambda_hierarchy` 必须为 0；现有 `coarse_type` 不是已审核的 ontology
   hierarchy。
9. nuisance adversary 当前只启用 train-only degree 的 log2 bin。请求 source 或
   detection-method head 会直接报错。

如果后续获得可靠的逐事实方向和 ontology，需要建立新版本数学合同，不能仅修改
一个配置开关后沿用 v1 结论。

## 2. 目标函数

### Topology channel

从 typed walk 中删除 relation token，在 protein 序列上构造窗口 context，使用
SGNS：

```text
L_topo = -mean(log sigmoid(s_pos) + sum log sigmoid(-s_noise))
```

noise token 按 train-only degree 的 `d^0.75` 分布采样。它只是表示学习中的噪声
token，不是“真实无相互作用蛋白质对”。

### Semantic channel

保留 relation token，使 semantic protein embedding 在窗口内预测 train relation
类别，采用全部 train relation 的交叉熵：

```text
L_typed = CE(z_sem @ relation_sem.T / sqrt(dim), relation_id)
```

### Typed decoder

v1 使用对称双通道 DistMult：

```text
s(i,j,k) = <z_topo_i * z_topo_j, r_topo_k> / sqrt(dim)
           + <z_sem_i * z_sem_j, r_sem_k> / sqrt(dim)
```

同一 pair 的全部已记录类型构成 positive mask；其他 train relation 只被称为
`unlabeled contrastive candidates`。ranking loss 不会把多个真类型互相作为负例。

总损失为：

```text
L = L_topo
    + lambda_typed * L_typed
    + lambda_decorr * L_decorr
    + lambda_adversary * L_adv
    + lambda_rank * L_rank
```

## 3. CUDA 数据路径

1. CPU 从 train TSV 建词表、canonical facts、degree 和 relation frequency。
2. CPU 以 float64 计算 corrected transition weight，并为每个 CSR row 构建
   Vose Alias Table。
3. `indptr/destination/relation/alias_q/alias_local` 一次性迁移到 CUDA。
4. GPU 按 batch 生成 `[B, walk_length+1]` protein 序列及 `[B, walk_length]`
   relation 序列。
5. context 在线生成和消费，不预生成全量 walk corpus。
6. Alias 概率、log-sigmoid、softplus 和 decorrelation 最终归约保持 FP32；矩阵
   运算可使用 CUDA AMP。

## 4. 文件结构

```text
sitne_walk/
├── config.py             # dataclass 配置和 fail-fast 校验
├── data.py               # train-only 词表、canonical pair、manifest/split 审计
├── graph.py              # packed CSR、corrected weights、segmented alias table
├── walks.py              # CUDA/MPS/CPU 批量 typed walks
├── contexts.py           # topology 和 relation-token contexts
├── gradient_reversal.py  # GRL autograd 原语
├── model.py              # 双通道 embedding、decoder、nuisance heads
├── losses.py             # SGNS、CE、decorrelation、adversary、ranking
├── evaluator.py          # all-known-positive filtered ranking
├── trainer.py            # 流式训练、AMP、早停和 checkpoint/RNG 恢复
├── provenance.py         # 环境、Git、输入 hash 与不可覆盖输出
└── cli.py                # inspect/train 命令
```

## 5. 环境检查

项目已有依赖足以运行代码：Python、NumPy、Pandas、PyYAML 和 PyTorch。目标
NVIDIA 主机必须安装与驱动匹配的 CUDA 版 PyTorch。

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

当配置为 `device: cuda` 时，CUDA 不可用会立即报错，不会退回 CPU。`auto` 才会
按 CUDA、MPS、CPU 顺序选择。

固定配置、软件栈、设备类型和 batch 参数时，代码会固定模型初始化及四条独立
随机流。checkpoint 同时保存 Python、NumPy、PyTorch CPU、CUDA/MPS 全局 RNG
及自建 Generator 状态，并拒绝加载配置不一致的 checkpoint。不同 GPU、不同
PyTorch/CUDA 版本或不同后端之间不承诺逐位一致；MPS 等后端即使指标可重复，
参数最低有效位也可能存在浮点差异，因此必须保留 provenance 并按数值容差验收。

## 6. 使用方法

先执行只读输入审计：

```bash
python 06_code/run_sitne_walk.py inspect \
  --config 06_code/configs/sitne_walk.example.yaml
```

`inspect` 会在训练前核对 split 行数/hash、canonical pair 交集、transductive
coverage，以及显式 `all_known_positive_paths` 是否包含每个可支持的验证/测试
target，避免训练结束后才发现 filtered evaluator 输入不完整。

训练：

```bash
python 06_code/run_sitne_walk.py train \
  --config 06_code/configs/sitne_walk.example.yaml
```

当前历史 split 含 train-unseen protein/relation，严格示例会在训练前拒绝不完整
coverage。若只验证代码路径，可运行明确标记为非科研结果的 SMOKE：

```bash
python 06_code/run_sitne_walk.py train \
  --config 06_code/configs/sitne_walk.smoke.yaml
```

也可以使用模块入口：

```bash
PYTHONPATH=06_code python -m sitne_walk inspect \
  --config 06_code/configs/sitne_walk.example.yaml
```

每次训练创建全新的 `08_results/sitne_walk/<run_id>/`，拒绝覆盖同名文件，并保存：

- resolved config；
- Python/PyTorch/CUDA/GPU 环境；
- Git commit；
- 输入路径、SHA-256、实际行数和 split audit；
- 实际执行的每个 Python 实现文件 SHA-256 与 Git worktree dirty 状态；
- protein/relation vocab；
- checkpoint、RNG 状态和训练历史；
- `completed.json` 或保留异常类型、消息和 traceback 的 `failure.json`；
- filtered test ranks、覆盖率和指标。

## 7. 必须拒绝的输入

代码会在以下情况下停止：

- manifest 行数或 hash 与 TSV 实体不一致；
- train/validation/test canonical pair 有交集；
- 缺少必要列、空字段、非法权重或空图；
- 显式请求 CUDA 但 CUDA 不可用；
- 请求 ordered/directional v1；
- 请求 source/detection-method adversary；
- `lambda_hierarchy != 0`；
- primary evaluation 存在 unseen protein 或 unseen relation；
- known-positive filter 没有包含当前 target relation。
- 整数配置写成小数、请求未启用的 hierarchy，或 CUDA BF16 不受硬件支持。

自由 node embedding 无法表示 train-unseen protein。protein-disjoint split 应返回
`unsupported`，不能把 validation/test 蛋白加入 embedding 表，也不能对空的支持
子集报告 MRR。

## 8. 测试

只收集 SITNE-Walk 新测试，避免运行 archive/external 的历史测试：

```bash
pytest -q 06_code/tests/sitne_walk
python -m compileall -q 06_code/sitne_walk 06_code/run_sitne_walk.py
```

CUDA 条件测试在没有 NVIDIA GPU 时会跳过。本机 CPU/MPS 测试通过不能替代目标
CUDA 主机上的显存、吞吐、AMP 和 CUDA parity 验收。
