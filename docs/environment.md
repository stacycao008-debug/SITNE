# Environment setup

## Runtime

- Python 3.10+
- Linux with a CUDA-capable NVIDIA GPU (for the full pipeline)
- CPU / MPS fallback is supported for smoke tests only

## Dependencies

`requirements.txt` (development/runtime, includes pytest) and
`environment.yml` (conda) are provided at the repo root. A frozen
reproducibility snapshot is kept in `docs/reproducibility/requirements.txt`.

Core libraries:

| Package | Version |
|---|---|
| numpy | ≥ 1.24 |
| pandas | ≥ 2.0 |
| scipy | ≥ 1.11 |
| scikit-learn | ≥ 1.3 |
| torch | ≥ 2.0 (CUDA build) |
| pyyaml | ≥ 6.0 |
| matplotlib | ≥ 3.7 |
| seaborn | ≥ 0.12 |
| tqdm | ≥ 4.65 |
| pytest | ≥ 7.0 |

No PyG / DGL / CuPy or custom CUDA extensions are required.

## CUDA server setup

On the CUDA host, install the PyTorch wheel matching your driver from the
official index before installing the rest, so that a later dependency install
cannot silently change the framework version. See
`scripts/dataset_bx_v1/setup_cuda_env.sh` for the exact pinning used for the
Dataset BX server (torch 2.10.0, cu126/cu128/cu130).

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Reproducibility notes

- Fixed config, software stack, device type and batch size reproduce model
  initialisation and four independent random streams. Seeds are recorded in
  `docs/reproducibility/seeds.json`.
- Checkpoints save Python / NumPy / PyTorch CPU / CUDA / MPS global RNG state
  and reject config-mismatched loads.
- Bitwise identity across different GPUs, PyTorch/CUDA versions, or backends is
  **not** guaranteed; validate against declared numerical tolerances and keep
  the provenance records.
- When `device: cuda` is set and CUDA is unavailable, the code fails fast
  (no CPU fallback); `device: auto` selects CUDA → MPS → CPU.
