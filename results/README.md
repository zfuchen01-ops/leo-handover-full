# results/ — 30s 时隙 LEO 切换实验结果备份

从远程服务器 `/root/leo_handover/log/` 备份的训练结果。这部分是 git 历史里不存在的原始产物（原仓库 `.gitignore` 排除了 `*.pkl`/`*.csv`，因此此前从未入库）。

## 内容

- `model/`：62 个 final/best checkpoint（`.pkl`，合计约 674 MB）。
  - 不含中间过程 checkpoint `_epNNNN.pkl`（801 个，约 12.3 GB，体积过大未备份）。
- `RL/`：102 个 `*_per_ep.csv` 训练曲线（合计约 3 MB）。

## 命名约定

`drqn_<星座>_u<用户数>_<口径>_<架构/标签>.pkl`

| 标签 | 含义 |
|------|------|
| `lstm` | 纯 LSTM（`Q_net`） |
| `tf` / `tf4_v2` | 纯 Transformer（`TransformerQ_net`） |
| `attn` / `attn_gate` | LSTM + attention（`LSTMAttnQ_net`，`_gate` 为门控残差变体） |
| `iscur` | 含 `is_current_sat` 特征（`FEAT_PER_SAT=5`） |
| `eps600` / `eps600_floor0` | 探索系数 / 下限相关变体 |

## 备份时间

2026-08-16
