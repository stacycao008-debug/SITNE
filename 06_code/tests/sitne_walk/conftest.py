"""SITNE-Walk 单元测试公共 fixtures。"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


@pytest.fixture
def toy_train_tsv(tmp_path: Path) -> Path:
    """含反向重复、多标签、自环的四节点 toy train graph。"""

    path = tmp_path / "train.tsv"
    path.write_text(
        "protein_i\tprotein_j\ttype_name\n"
        "A\tB\tr1\n"
        "B\tA\tr1\n"
        "A\tB\tr2\n"
        "B\tC\tr1\n"
        "C\tD\tr2\n"
        "D\tA\tr2\n"
        "C\tC\tr1\n",
        encoding="utf-8",
    )
    return path

