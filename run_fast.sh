#!/bin/bash
MODEL="log/model/drqn_C_u200_tf4_v2_C_ep2550.pkl"

for round in 1 2 3 4; do
  for d in 50 100 200 500; do
    echo "=== r$round d=$d $(date +%H:%M:%S) ==="
    python3 -u train_drqn.py --users $d --constellation C --device cuda --tag tf4_fast --resume "$MODEL" --min-episodes 25 --slots 200 --lr 0.0001 --patience 20 2>&1 | tail -2
    MODEL=$(ls -t log/model/drqn_C_u${d}_tf4_fast_ep*.pkl 2>/dev/null | head -1)
    [ -z "$MODEL" ] && MODEL="log/model/drqn_C_u200_tf4_v2_C_ep2550.pkl"
  done
done
echo "DONE $(date +%H:%M:%S)" && ls -t log/model/drqn_C_u*_tf4_fast_*.pkl | head -5
