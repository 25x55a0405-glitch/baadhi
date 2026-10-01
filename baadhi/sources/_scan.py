"""Worker process of geofabrik.extract:  python -m baadhi.sources._scan <args.json>  (not meant to be run by hand)."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    args = json.loads(Path(sys.argv[1]).read_text(encoding="utf8"))
    from . import geofabrik
    area, ctx = geofabrik.scan(Path(args["pbf"]), tuple(args["keys"]), set(args["owned"]), args["area_bbox"], args["area_layers"],
                               args["ctx_bbox"], args["ctx_layers"])
    Path(args["out"]).write_text(json.dumps({"area": area, "ctx": ctx}), encoding="utf8")


if __name__ == "__main__":
    main()
