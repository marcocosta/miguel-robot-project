"""Run a safe read-only HiWonder environment probe."""

from __future__ import annotations

import json
from pathlib import Path

from miguel_core import MiguelHiWonderRealProbe


def main() -> None:
    probe = MiguelHiWonderRealProbe()
    result = probe.probe()
    print(json.dumps(result, indent=2, sort_keys=True))

    output_path = Path("data") / "miguel_hiwonder_probe.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[MIGUEL_HIWONDER_PROBE] saved {output_path}")


if __name__ == "__main__":
    main()
