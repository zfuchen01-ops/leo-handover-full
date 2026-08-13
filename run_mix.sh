#!/bin/bash
MODEL="log/model/drqn_C_u200_tf4_v2_C_ep2550.pkl"
CONST="C"

for round in 1 2 3 4 5 6; do
  for d in 50 100 150 200; do
    echo "=== round $round dens=$d ==="
    python3 -u train_drqn.py \
      --users $d --constellation $CONST --device cuda --tag tf4_phase \
      --resume "$MODEL" --min-episodes 50 --slots 500 --lr 0.0001 \
      --patience 20 2>&1 | grep -E "训练完成|episode_reward|new best"
    MODEL=$(ls -t log/model/drqn_${CONST}_u${d}_tf4_phase_ep*.pkl 2>/dev/null | head -1)
    [ -z "$MODEL" ] && MODEL="log/model/drqn_C_u200_tf4_v2_C_ep2550.pkl"
    echo "  -> next model: $MODEL"
  done
done
echo "ALL DONE"
ls -t log/model/drqn_C_u*_tf4_phase_*.pkl | head -5
