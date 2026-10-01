"""Start the Baadhi dashboard:  python serve.py   (then open http://127.0.0.1:8765)"""
import os
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
tmp = here / ".work" / "tmp"          # keep temporary files inside the project folder
tmp.mkdir(parents=True, exist_ok=True)
for var in ("TEMP", "TMP", "TMPDIR"):
    os.environ[var] = str(tmp)
os.environ.setdefault("TORCH_HOME", str(here / ".work" / "torch"))

from baadhi.server import main  # noqa: E402

main()
