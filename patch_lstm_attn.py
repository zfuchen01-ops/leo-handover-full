#!/usr/bin/env python3
"""Patch: add LSTMAttnQ_net (LSTM temporal + cross-sat self-attention spatial).
接口与 Q_net 完全一致 (forward(input,h,c)->(y,new_h,new_c), init_hidden_state),
走 LSTM 状态路径, 由 env LEO_LSTM_ATTN=1 开启."""
import pathlib

p = pathlib.Path('/root/leo_handover/DRQNAgent.py')
src = p.read_text()

cls = '''

class LSTMAttnQ_net(nn.Module):
    """LSTM(时序) + 跨星 self-attention(空间) 混合:
       保留 DRQN 的 LSTM 时序记忆, 把独立 per-sat MLP 换成跨星 self-attention,
       让卫星之间互相比较 (公平对比: LSTM+DQN vs LSTM+attention)."""
    def __init__(self, state_space, action_space, hidden_size, num_agents=200, embed_dim=64,
                 top_k=TOP_K, feat_per_sat=FEAT_PER_SAT):
        super().__init__()
        nhead = int(os.environ.get('LEO_ATTN_HEADS', '4'))
        nlayers = int(os.environ.get('LEO_ATTN_LAYERS', '2'))
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.input_size = state_space
        self.output_size = action_space
        self.num_agents = num_agents
        self.top_k = top_k
        self.feat_per_sat = feat_per_sat

        self.agent_embed = nn.Embedding(num_agents, embed_dim)
        nn.init.uniform_(self.agent_embed.weight, -0.1, 0.1)

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name: nn.init.orthogonal_(param, gain=1.0)
            elif 'bias' in name: nn.init.constant_(param, 0.0)

        # 跨星 self-attention
        self.sat_proj = nn.Linear(feat_per_sat, hidden_size)
        enc = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead,
                                         dim_feedforward=hidden_size * 4, dropout=0.05,
                                         activation='relu', batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=nlayers)

        self.ctx_fcs = nn.ModuleList([nn.Linear(hidden_size, 16) for _ in range(num_agents)])
        self.sat_nets = nn.ModuleList([nn.Sequential(nn.Linear(feat_per_sat, 32), nn.ReLU(),
                                                     nn.Linear(32, 16), nn.ReLU())
                                       for _ in range(num_agents)])
        self.cross_heads = nn.ModuleList([nn.Linear(hidden_size, 16) for _ in range(num_agents)])
        self.v_heads = nn.ModuleList([nn.Linear(16, 1) for _ in range(num_agents)])
        self.a_heads = nn.ModuleList([nn.Linear(16 + 16, 1) for _ in range(num_agents)])
        self._head_idx = 0

    def forward(self, input, h, c, head_indices=None):
        B, seq_len = input.shape[0], input.shape[1]
        K = self.top_k
        F = self.feat_per_sat
        H = self.hidden_size

        if head_indices is not None:
            embed = self.agent_embed(head_indices)
        else:
            embed = self.agent_embed(torch.tensor([self._head_idx], device=input.device)).expand(B, -1)
        embed_seq = embed.unsqueeze(1).expand(-1, seq_len, -1)
        x, (new_h, new_c) = self.lstm(torch.cat([input, embed_seq], dim=-1), (h, c))  # [B, seq, H]

        ctx = torch.zeros(B, seq_len, 16, device=input.device)
        if head_indices is not None:
            for idx in range(self.num_agents):
                m = (head_indices == idx)
                if m.any():
                    ctx[m] = torch.relu(self.ctx_fcs[idx](x[m]))
        else:
            ctx = torch.relu(self.ctx_fcs[self._head_idx](x))
        V = torch.zeros(B, seq_len, 1, device=input.device)
        if head_indices is not None:
            for idx in range(self.num_agents):
                m = (head_indices == idx)
                if m.any():
                    V[m] = self.v_heads[idx](ctx[m])
        else:
            V = self.v_heads[self._head_idx](ctx)

        # 跨星 self-attention (padding 空位 mask 掉)
        per_sat = input[:, :, :K * F].reshape(B, seq_len, K, F)  # [B, seq, K, F]
        pad = (per_sat.abs().sum(dim=-1) < 1e-8)                 # [B, seq, K] True=padding
        sat_h = self.sat_proj(per_sat)
        tokens = sat_h.view(B * seq_len, K, H)
        out = self.transformer(tokens, src_key_padding_mask=pad.view(B * seq_len, K))
        out = out.view(B, seq_len, K, H)

        sat_feat = torch.zeros(B, seq_len, K, 16, device=input.device)
        if head_indices is not None:
            for idx in range(self.num_agents):
                m = (head_indices == idx)
                if m.any():
                    ns, sq = per_sat[m].shape[0], per_sat[m].shape[1]
                    raw_feat = self.sat_nets[idx](per_sat[m].reshape(ns * sq * K, F)).reshape(ns, sq, K, 16)
                    cross_feat = self.cross_heads[idx](out[m])
                    sat_feat[m] = raw_feat + cross_feat
        else:
            raw_feat = self.sat_nets[self._head_idx](per_sat.reshape(B * seq_len * K, F)).reshape(B, seq_len, K, 16)
            cross_feat = self.cross_heads[self._head_idx](out)
            sat_feat = raw_feat + cross_feat

        ctx_exp = ctx.unsqueeze(2).expand(-1, -1, K, -1)
        combined = torch.cat([sat_feat, ctx_exp], dim=-1)
        if head_indices is not None:
            y = torch.zeros(B, seq_len, K, device=combined.device)
            for idx in range(self.num_agents):
                m = (head_indices == idx)
                if m.any():
                    A = self.a_heads[idx](combined[m]).squeeze(-1)
                    A = A - A.mean(dim=-1, keepdim=True)
                    y[m] = A + V[m]
        else:
            A = self.a_heads[self._head_idx](combined).squeeze(-1)
            A = A - A.mean(dim=-1, keepdim=True)
            y = A + V
        y = torch.clamp(y, -20, 20)
        return y, new_h, new_c

    def init_hidden_state(self, batch_size, train=None):
        if train is True:
            return torch.zeros([1, batch_size, self.hidden_size]), torch.zeros([1, batch_size, self.hidden_size])
        else:
            return torch.zeros([1, 1, self.hidden_size]), torch.zeros([1, 1, self.hidden_size])

'''

# 1. 追加类 (定义在 UserAgent 之前)
anchor = 'class UserAgent:'
assert src.count(anchor) == 1, 'UserAgent anchor not unique'
src = src.replace(anchor, cls + anchor, 1)

# 2. UserAgent 分支 (transformer 行无 num_agents, 唯一)
old_u = """        if self.use_transformer:
            self.evaluate_net = TransformerQ_net(state_dim + 64, K, hidden_size, top_k=K).to(self.device)
        else:
            self.evaluate_net = Q_net(state_dim + 64, K, hidden_size, top_k=K).to(self.device)"""
new_u = """        use_lstm_attn = os.environ.get('LEO_LSTM_ATTN', '0') == '1'
        if use_lstm_attn:
            self.evaluate_net = LSTMAttnQ_net(state_dim + 64, K, hidden_size, top_k=K).to(self.device)
        elif self.use_transformer:
            self.evaluate_net = TransformerQ_net(state_dim + 64, K, hidden_size, top_k=K).to(self.device)
        else:
            self.evaluate_net = Q_net(state_dim + 64, K, hidden_size, top_k=K).to(self.device)"""
assert src.count(old_u) == 1, f'UserAgent block count={src.count(old_u)}'
src = src.replace(old_u, new_u, 1)

# 3. CenterAgent 分支 (transformer 行有 num_agents, 唯一)
old_c = """        if use_tf:
            self.evaluate_net = TransformerQ_net(state_dim + 64, K, hidden_size, num_agents=n_agents, top_k=K).to(self.device)
            print("  [CenterAgent] 使用 TransformerQ_net", flush=True)
        else:
            self.evaluate_net = Q_net(state_dim + 64, K, hidden_size, top_k=K).to(self.device)"""
new_c = """        use_lstm_attn = os.environ.get('LEO_LSTM_ATTN', '0') == '1'
        if use_lstm_attn:
            self.evaluate_net = LSTMAttnQ_net(state_dim + 64, K, hidden_size, num_agents=n_agents, top_k=K).to(self.device)
            print("  [CenterAgent] 使用 LSTMAttnQ_net", flush=True)
        elif use_tf:
            self.evaluate_net = TransformerQ_net(state_dim + 64, K, hidden_size, num_agents=n_agents, top_k=K).to(self.device)
            print("  [CenterAgent] 使用 TransformerQ_net", flush=True)
        else:
            self.evaluate_net = Q_net(state_dim + 64, K, hidden_size, top_k=K).to(self.device)"""
assert src.count(old_c) == 1, f'CenterAgent block count={src.count(old_c)}'
src = src.replace(old_c, new_c, 1)

p.write_text(src)
print('patched ok: LSTMAttnQ_net added, UserAgent+CenterAgent branched')
