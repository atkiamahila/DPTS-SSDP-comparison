#!/bin/bash
set -euo pipefail

QDIR=QDIR=/lus/grand/projects/VeloC/atkia/ssdp/SSDP/results/_queue/llama-3b_math500
PBS=/lus/grand/projects/VeloC/atkia/ssdp/SSDP/scripts/ssdp_debugscaling.pbs
mkdir -p "$QDIR/claim" "$QDIR/done"
if [[ ! -f "$QDIR/chunks.txt" ]]; then
  declare -A CH=( [3000]=50 [5000]=40 [8000]=25 [10000]=20 )
  : > "$QDIR/chunks.txt"
  for B in 3000 5000 8000 10000; do
    sz=${CH[$B]}; s=0
    while [ $s -lt 500 ]; do
      e=$(( s+sz )); [ $e -gt 500 ] && e=500
      printf "b%05d_%03d %d %d %d\n" "$B" "$s" "$B" "$s" "$e" >> "$QDIR/chunks.txt"
      s=$e
    done
  done
  echo "[QUEUE] built $(wc -l < "$QDIR/chunks.txt") chunks"
fi
while :; do
  total=$(wc -l < "$QDIR/chunks.txt" 2>/dev/null || echo 999999)
  done_count=$(ls "$QDIR/done" 2>/dev/null | wc -l)

  echo "[$(date -Is)] progress: $done_count / $total"

  [ "$done_count" -ge "$total" ] && {
    echo "ALL DONE"
    break
  }

  jid=$(qsub "$PBS")
  echo "submitted $jid"

  while qstat "$jid" >/dev/null 2>&1; do
    sleep 60
  done
done
