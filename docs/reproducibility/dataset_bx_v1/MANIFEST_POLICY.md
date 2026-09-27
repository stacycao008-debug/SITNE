# 服务器 ZIP 与 SHA-256 清单策略 v1

## 1. 目标

`bundle_manifest.py` 为服务器 ZIP 选择普通文件、生成稳定排序的逐文件 SHA-256，
并在解压后验证“无缺失、无额外受管文件、无哈希不一致”。manifest 自身写入 ZIP
但不递归记录自己；ZIP 外另生成同名 `.zip.sha256` 用于传输前校验。

## 2. 显式排除规则

v1 排除：

- `08_results/**`：历史及动态实验结果；
- 任意 `.venv/**` 或 `.venv.*`：服务器本地/隔离虚拟环境；
- `.git/**`、`.svn/**`：版本控制元数据；
- `.cache/**`、`__pycache__/**`、`.pytest_cache/**`、`.mypy_cache/**`、
  `.ruff_cache/**`、`.ipynb_checkpoints/**`、`.tox/**`、`.nox/**`：缓存；
- `.pyc`、`.pyo`：散落的 Python bytecode；
- `.coverage`；
- `.DS_Store` 和 `._*`：macOS metadata / AppleDouble；
- bundle manifest 自身；
- symlink 和其他非普通文件。

原包的 `manifests/EXPORT_SHA256SUMS.txt` 仍原样保留，作为升级前 188 项冻结
基线。它包含随后被服务器 ZIP 策略排除的历史 `08_results/**` 和缓存条目，因此
不能替代服务器包清单，也不应在解压后的精简 ZIP 上作为完整性判据。服务器传输
与解压验收一律以本目录的 `BUNDLE_SHA256SUMS.txt` 为准。

Python ZIP writer 不写 filesystem extended attributes。服务器代码、配置、Dataset
B/B*、审计表、文档和必要测试均应进入受管清单。

## 3. Dry-run

```bash
bash scripts/dataset_bx_v1/package_for_server.sh --dry-run
```

该命令只输出 included/excluded 文件数、字节数和排除原因统计；不创建 staging、
manifest、ZIP、checksum 或结果文件。

## 4. 正式生成（服务器准备完成后由用户执行）

```bash
bash scripts/dataset_bx_v1/package_for_server.sh
```

或指定项目目录之外的输出：

```bash
bash scripts/dataset_bx_v1/package_for_server.sh \
  --output /absolute/output/SITNE-Walk-BX-CUDA-server-v1.zip
```

打包器拒绝把 ZIP 写到项目根目录内部，也拒绝覆盖已有 ZIP 或 `.zip.sha256`。
archive entry 按 UTF-8 path 排序并使用固定 timestamp；逐文件 manifest 是跨传输的
内容依据。不同 Python/zlib 平台生成的 ZIP 压缩字节不承诺相同，因此每次以打包器
同时产生的外部 ZIP SHA-256 为准。

当前交付目录在最终冻结时也会生成一份整体 manifest。若需要为另一个尚未含该
文件的非 ZIP 目录生成 manifest，可执行：

```bash
python3 scripts/dataset_bx_v1/bundle_manifest.py generate --root "$PWD"
```

该命令同样拒绝覆盖已有 manifest。通常无需在源目录运行，因为正式打包会把新
manifest 直接写入 ZIP 而不改变源目录。

## 5. 解压后验证

```bash
bash scripts/dataset_bx_v1/verify_server_bundle.sh
```

验证器先核对必要目录/入口/五个正式配置，再逐条重算 SHA-256，并拒绝 missing、
unexpected 或 mismatch。`.venv`、缓存和 `08_results` 仍按同一策略忽略，因此安装
环境和产生动态结果后也可复核原包内容。

## 6. 与原 188 项 export baseline 的关系

`manifests/EXPORT_SHA256SUMS.txt` 是原 SITNE-Walk 导出包的不可变历史证据，仍随
包保留；其 188 条记录包含现在按服务器策略排除的 51 个 `08_results` 文件和 4 个
`.pytest_cache` 文件。因此它用于证明源目录中原基线未被本次追加修改，不能作为
新服务器 ZIP 的 completeness manifest，也不应在解压后的精简包上执行 `-c`。

新 ZIP 的唯一完整性清单是
`10_reproducibility/dataset_bx_v1/BUNDLE_SHA256SUMS.txt`。两份清单用途不同，
不得覆盖、合并或手工改写旧清单来消除缺失项。
