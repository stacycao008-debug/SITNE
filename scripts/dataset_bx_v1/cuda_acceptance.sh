#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
venv_python="$project_root/.venv/bin/python"
require_bf16=0

usage() {
  cat <<'EOF'
Usage: cuda_acceptance.sh [--require-bf16]

Runs environment, Dataset B/B* inspect, CUDA FP32, CUDA FP16 AMP, checkpoint
round-trip, and the complete SITNE-BX pytest suite. Any skipped test fails the
acceptance. BF16 is exercised when supported and is mandatory with
--require-bf16. No training, checkpoint selection, or test evaluation is run.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --require-bf16)
      require_bf16=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[ -x "$venv_python" ] || {
  echo "ERROR: missing bundle environment: $venv_python" >&2
  echo "Run scripts/dataset_bx_v1/setup_cuda_env.sh first." >&2
  exit 1
}
[ -f "$project_root/06_code/run_sitne_bx.py" ] || {
  echo "ERROR: missing SITNE-BX entry point" >&2
  exit 1
}
[ -d "$project_root/06_code/tests/sitne_bx" ] || {
  echo "ERROR: missing SITNE-BX test suite" >&2
  exit 1
}
[ -f "$project_root/06_code/tests/sitne_bx/test_cuda.py" ] || {
  echo "ERROR: missing mandatory SITNE-BX CUDA test module" >&2
  exit 1
}

cd "$project_root"
bash "$script_dir/probe_cuda_environment.sh"

PYTHONDONTWRITEBYTECODE=1 "$venv_python" 06_code/run_sitne_bx.py inspect \
  --config 06_code/configs/sitne_bx.fine_fp32.yaml \
  --project-root "$project_root"

SITNE_BX_REQUIRE_BF16="$require_bf16" PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$project_root/06_code${PYTHONPATH:+:$PYTHONPATH}" \
  "$venv_python" - <<'PY'
from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile

import torch

from sitne_bx.config import LossConfig, ModelConfig
from sitne_bx.losses import composite_loss
from sitne_bx.model import SITNEBXModel


def fail(message: str) -> None:
    raise SystemExit(message)


if torch.__version__.split("+", 1)[0] != "2.10.0":
    fail(f"expected torch 2.10.0, got {torch.__version__}")
if torch.version.cuda is None or not torch.cuda.is_available():
    fail("CUDA PyTorch is unavailable; CPU/MPS fallback is forbidden")
if torch.cuda.device_count() < 1:
    fail("no visible CUDA device")

device = torch.device("cuda:0")
torch.cuda.set_device(device)
torch.manual_seed(20260728)
torch.cuda.manual_seed_all(20260728)


def exercise_precision(name: str, dtype: torch.dtype, use_scaler: bool) -> None:
    model_config = ModelConfig(
        embedding_dim=32,
        token_embedding_dim=16,
        pair_hidden_dim=48,
        chunk_size=32,
        chunk_stride=24,
        chunk_batch_size=4,
        convolution_kernel_size=3,
        dropout=0.0,
    )
    model = SITNEBXModel(model_config, num_types=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sequences = [
        "ACDEFGHIKLMNPQRSTVWYUACDEFGHIKLMNPQRSTVWY",
        "MKTIIALSYIFCLVFADYKDDDDKACDEFGHIKLMNPQ",
        "QRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMN",
        "GASGASGASGASGASGASGASGASGASGASGASGAS",
        "MNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIK",
    ]
    left_index = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
    right_index = torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device)
    labels = torch.tensor([1.0, 0.0, 1.0, 1.0], device=device)
    typed_targets = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]],
        device=device,
    )
    typed_mask = torch.tensor([True, False, True, True], device=device)
    optimizer.zero_grad(set_to_none=True)

    amp_enabled = dtype != torch.float32
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=amp_enabled):
        output = model(sequences, left_index, right_index)
        reverse = model(sequences, right_index, left_index)
        losses = composite_loss(
            output,
            labels,
            typed_targets,
            typed_mask,
            LossConfig(binary_weight=1.0, topology_weight=0.0, typed_weight=0.2),
        )
        loss = losses.total
    if output.binary_logits.device.type != "cuda" or loss.device.type != "cuda":
        fail(f"{name}: detected non-CUDA output; fallback is forbidden")
    if output.typed_logits is None or output.typed_logits.device.type != "cuda":
        fail(f"{name}: typed head did not execute on CUDA")
    if not torch.allclose(
        output.binary_logits.float(), reverse.binary_logits.float(), atol=1e-5, rtol=1e-5
    ):
        fail(f"{name}: undirected symmetric output check failed")
    if not math.isfinite(float(loss.detach().float().cpu())):
        fail(f"{name}: non-finite loss")

    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [parameter.grad for parameter in model.parameters()]
    if not gradients or any(gradient is None for gradient in gradients):
        fail(f"{name}: missing gradient")
    if any(not torch.isfinite(gradient).all() for gradient in gradients if gradient is not None):
        fail(f"{name}: non-finite gradient")
    if not any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients if gradient is not None):
        fail(f"{name}: all gradients are zero")
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)
    if torch.cuda.max_memory_allocated(device) <= 0:
        fail(f"{name}: CUDA memory accounting stayed at zero")

    with tempfile.TemporaryDirectory(prefix="sitne_bx_cuda_acceptance_") as temp_dir:
        checkpoint = Path(temp_dir) / "roundtrip.pt"
        torch.save({"model": model.state_dict(), "precision": name}, checkpoint)
        restored = SITNEBXModel(model_config, num_types=3).to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        restored.load_state_dict(payload["model"])
        for original, loaded in zip(model.parameters(), restored.parameters(), strict=True):
            if original.device.type != "cuda" or loaded.device.type != "cuda":
                fail(f"{name}: checkpoint restored outside CUDA")
            if not torch.equal(original, loaded):
                fail(f"{name}: checkpoint round-trip mismatch")
    print(
        f"PRECISION_ACCEPTED={name} loss={float(loss.detach().float().cpu()):.8f} "
        f"peak_bytes={torch.cuda.max_memory_allocated(device)}"
    )


torch.cuda.reset_peak_memory_stats(device)
exercise_precision("fp32", torch.float32, use_scaler=False)
torch.cuda.reset_peak_memory_stats(device)
exercise_precision("fp16_amp", torch.float16, use_scaler=True)

bf16_supported = bool(torch.cuda.is_bf16_supported())
require_bf16 = os.environ.get("SITNE_BX_REQUIRE_BF16") == "1"
if bf16_supported:
    torch.cuda.reset_peak_memory_stats(device)
    exercise_precision("bf16_amp", torch.bfloat16, use_scaler=False)
elif require_bf16:
    fail("BF16 was required but torch.cuda.is_bf16_supported() is false")
else:
    print("BF16_CAPABILITY=false; BF16 was not selected as an acceptance requirement")

print("CUDA_SYNTHETIC_ACCEPTANCE=PASS")
print("CUDA_DEVICE", torch.cuda.get_device_name(device))
print("CUDA_RUNTIME", torch.version.cuda)
PY

acceptance_tmp="$(mktemp -d "${TMPDIR:-/tmp}/sitne_bx_cuda_acceptance.XXXXXX")"
trap 'rm -rf "$acceptance_tmp"' EXIT
pytest_log="$acceptance_tmp/pytest.log"

set +e
SITNE_BX_REQUIRE_CUDA=1 PYTHONDONTWRITEBYTECODE=1 \
  "$venv_python" -m pytest -q -rs -p no:cacheprovider \
  06_code/tests/sitne_bx 2>&1 | tee "$pytest_log"
pytest_status="${PIPESTATUS[0]}"
set -e
if [ "$pytest_status" -ne 0 ]; then
  echo "ERROR: SITNE-BX test suite failed with status $pytest_status" >&2
  exit "$pytest_status"
fi
if grep -Eiq '(^|[[:space:]])[0-9]+ skipped([,[:space:]]|$)|^SKIPPED' "$pytest_log"; then
  echo "ERROR: at least one SITNE-BX test was skipped; CUDA acceptance rejected" >&2
  exit 1
fi

echo "SITNE_BX_CUDA_ACCEPTANCE=PASS"
echo "NO_CPU_OR_MPS_FALLBACK=VERIFIED"
echo "NO_SKIPPED_TESTS=VERIFIED"
