from __future__ import annotations

from pathlib import Path
import sys


def test_ms_human_700_data_installed() -> None:
    """
    Verify that MS-Human-700 data files are present after installation.

    data_files configured in pyproject.toml are installed relative to
    sys.prefix, so we check there for the main XML model file.
    """

    target_rel = Path("MS-Human-700") / "MS-Human-700.xml"

    base = Path(sys.prefix)
    candidate = base / target_rel
    if not candidate.exists():
        msg = "MS-Human-700 data file not found under sys.prefix: "
        raise AssertionError(msg + f"looked for {candidate}")
