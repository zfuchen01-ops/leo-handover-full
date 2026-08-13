cd /root/leo_handover
for u in 50 100 200; do
  echo "===== TEST users=$u constellation=B ====="
  LEO_TEST_DEVICE=cuda LEO_MODEL_PATH=log/model/drqn_C_u200_tf4_per_ep200.pkl LEO_USE_TRANSFORMER=0 LEO_TEST_USERS=$u LEO_TEST_CONSTELLATION=B python3 -u test_model.py
done
echo "===== ALL DONE ====="