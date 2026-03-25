#!/usr/bin/env bash

LOG=db_sweep.log

for b in 1 2 4 8 16 32 64 128; do
  {
    echo "------------------------------------------------------------------------------------------"
    python sweep.py -db --batch ${b} --no-plot
    echo ""
  } 2>&1 | tee -a "$LOG"
done