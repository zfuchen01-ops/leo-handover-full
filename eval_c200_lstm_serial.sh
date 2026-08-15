#!/bin/bash
cd /root/leo_handover
export PYTHONPATH=/root/leo_handover
env LEO_USE_TRANSFORMER=0 LEO_LSTM_ATTN=0 python3 exp/eval_drqn.py --constellation C --users 200 --slots 2000 --seed 42 \
    --ckpt log/model/drqn_C_u200_lstm_s30_seed42_ep2000.pkl --tail 100 --tag c200_lstm_s42 \
    > /tmp/eval_c200_lstm_s42.log 2>&1 < /dev/null
env LEO_USE_TRANSFORMER=0 LEO_LSTM_ATTN=0 python3 exp/eval_drqn.py --constellation C --users 200 --slots 2000 --seed 43 \
    --ckpt log/model/drqn_C_u200_lstm_s30_seed42_ep2000.pkl --tail 100 --tag c200_lstm_s43 \
    > /tmp/eval_c200_lstm_s43.log 2>&1 < /dev/null
echo "LSTM_SERIAL_DONE"
