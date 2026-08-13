#!/bin/bash
cd /root/leo_handover
M="$1"
for round in 1 2 3; do
  for d in 50 100 200; do
    echo "=== r$round d=$d $(date +%H:%M:%S) ==="
    python3 -u train_drqn.py --users $d --constellation C --device cuda --tag tf4_v2 --resume "$M" --min-episodes 25 --slots 200 --lr 0.0001 --patience 20 2>&1 | tail -2
    M=$(ls -t log/model/drqn_C_u${d}_tf4_v2_ep*.pkl 2>/dev/null | head -1)
    [ -z "$M" ] && M="$LATEST"
  done
done
echo "DONE $(date +%H:%M:%S)"
