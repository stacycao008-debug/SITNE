from __future__ import annotations

import numpy as np
import pytest

from sitne_bx.artifacts import seal_payload, verify_sealed_payload
from sitne_bx.metrics import auprc, auroc, binary_metrics, select_mcc_threshold


def test_metrics_and_deterministic_threshold():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])
    assert auroc(labels, probabilities) == 1.0
    assert auprc(labels, probabilities) == 1.0
    threshold, mcc = select_mcc_threshold(labels, probabilities)
    assert threshold == 0.6
    assert mcc == 1.0
    result = binary_metrics(labels, probabilities, threshold)
    assert result["confusion_matrix"] == {"tp": 2, "tn": 2, "fp": 0, "fn": 0}


def test_sealed_manifest_detects_tampering():
    payload = seal_payload({"state": "frozen", "epoch": 2})
    verify_sealed_payload(payload)
    payload["epoch"] = 3
    with pytest.raises(ValueError):
        verify_sealed_payload(payload)
