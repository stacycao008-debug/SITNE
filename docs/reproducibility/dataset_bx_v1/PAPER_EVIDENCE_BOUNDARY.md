# SITNE-Walk-BX 论文证据边界

## 1. 当前准备包能证明什么

代码、配置、数据镜像、哈希、测试和服务器脚本准备完成，只能证明“实验输入与工程
流程已被组织为可执行、可审计的候选实现”。如果本机完成 CPU 合成前后向，也只
是工程 smoke evidence。

目标 NVIDIA 主机上的 `cuda_acceptance.sh` PASS 可进一步证明：指定环境中真实
CUDA FP32/AMP 可执行、无 CPU/MPS fallback、无 skipped BX tests、loss/gradient
有限、显存被使用且 checkpoint 可往返。它仍不能证明模型准确率、稳定性、泛化、
统计显著性或优于基线。

## 2. 形成科研结果的最低条件

单次运行至少要完整执行并通过：

```text
inspect -> train -> select -> test -> verify-run
```

同时保存实际配置、数据/代码/environment hashes、checkpoint、完整 prediction、
Validation selection manifest 和 Test metrics。论文级比较还需要预先冻结的重复
实验、随机种子、基线、消融、统计分析和失败处理方案；不能由本准备包的存在自动
推导这些结论。

## 3. Split 与负例语义

- Train/Validation/Test 的蛋白不重叠是 Dataset B 的归纳式评测条件，不得把
  Validation/Test 蛋白提前加入 train-only 自由 node embedding 表。
- Dataset B 的显式 label 可进入二分类 BCE；B* auxiliary type 只可用于安全的
  Train positive typed fact。
- 缺少 type annotation 不等于负类型；held-out type 不得用于训练。
- 未知 pair 不是生物学负例；SGNS noise 只是表示学习噪声 token，也不是非互作
  证据。
- Test 只在 Validation selection sealed 后解锁；Test 不参与模型或阈值选择。

## 4. B* 结论和分发限制

`fine` 是 B* 主配置。`coarse` 仅是 legacy grouping，除非另有经过审核的 ontology
证据，不得写成层级本体监督。B* 标识为 `LOCAL_INTERNAL_ONLY`：

- 不公开原始 B*、targets、由其可逆恢复的中间表或整个服务器 ZIP；
- 公开论文可报告经过许可审查的聚合方法和结果，但数据可用性声明必须准确说明
  限制、访问条件和不可公开原因；
- 任何公开 source data 都要单独检查是否能泄漏 B* 受限内容。

## 5. 明确禁止的论文表述

在没有对应证据时，不得声称：

- “CUDA 已验证”——仅凭本机 CPU/MPS 或 CUDA test 被 skip；
- “模型实现了 unseen-protein 泛化”——仅凭代码能加载序列；
- “优于现有方法”——没有同协议、同 split、经审核的正式基线结果；
- “B* 类型监督全面覆盖”——存在没有安全 typed fact 的 Train positive；
- “coarse 类型构成生物学 ontology hierarchy”——当前仅为 legacy grouping；
- “缺少 annotation 或未知 pair 是负例”；
- “Test 结果用于选择后仍是无偏最终测试”。

## 6. 写作移交要求

论文作者必须收到完整 run directory、CUDA acceptance 日志、实验回报、
实际配置和哈希。任何只存在于 README、计划、synthetic smoke 或未执行配置中的
功能，应使用“实现/预设/待验证”措辞，而不能写成观察到的科研结果。
