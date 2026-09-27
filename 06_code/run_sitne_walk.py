"""无需安装包即可运行 SITNE-Walk CLI 的薄入口。"""

from __future__ import annotations

from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from sitne_walk.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

