import torch, hashlib, sys
import DRQNAgent  # 提供 Q_net 类定义, 使 torch.load 能反序列化模块

def state_hash(path):
    m = torch.load(path, map_location='cpu', weights_only=False)
    # m 可能是模块或 state_dict, 统一取 state_dict
    sd = m.state_dict() if hasattr(m, 'state_dict') else m
    h = hashlib.md5()
    for k in sorted(sd.keys()):
        t = sd[k]
        # 用 tensor 的原始字节做哈希, 排除 pickle 元数据/对象地址等非确定性
        h.update(k.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest(), len(sd)

for tag in ['e', 'f', 'g']:
    p = f'log/model/drqn_C_u200_{tag}.pkl'
    h, n = state_hash(p)
    print(f'{tag}: state_dict_hash={h} tensors={n}')
