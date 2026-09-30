#!/usr/bin/env bash
# Downloads the Kuro Siwo GRD training shards (CC BY 4.0, Bountos et al. 2024) to D:, resumable.
set -u
source /d/claude-code/baadhi/env.sh
OUT=/d/claude-code/baadhi/data/train/kurosiwo/train_GRD
mkdir -p "$OUT"
BASE=https://huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-Webdataset/resolve/main/train_GRD
# Python on Windows ends lines with \r\n — strip the \r or it ends up inside the file names.
NAMES=$(curl -s -m 60 https://huggingface.co/api/datasets/orion-ai-lab/Kuro-Siwo-Webdataset/tree/main/train_GRD \
  | /d/claude-code/baadhi/.venv/Scripts/python.exe -c 'import sys,json; [print(e["path"].split("/")[-1]) for e in json.load(sys.stdin) if e["type"]=="file"]' \
  | tr -d '\r')
for f in $NAMES; do
  if [ -s "$OUT/$f" ]; then echo "have $f"; continue; fi
  echo "$(date +%H:%M:%S) start $f"
  for try in 1 2 3 4 5; do
    curl -s -f -L -C - --retry 5 --retry-delay 10 -o "$OUT/$f.part" "$BASE/$f" && break
    echo "  retry $try for $f (curl exit $?)"; sleep 20
  done
  [ -s "$OUT/$f.part" ] && mv "$OUT/$f.part" "$OUT/$f" && echo "$(date +%H:%M:%S) done  $f $(du -h "$OUT/$f" | cut -f1)"
done
echo "ALL DONE"
