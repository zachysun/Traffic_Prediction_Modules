"""
'ScaledDotProductAttention' module was modified from
https://github.com/xmu-xiaoma666/External-Attention-pytorch
"""
import numpy as np
import torch
from torch import nn
from torch.nn import init


class PositionEncoder(nn.Module):
    """
    - Params: d_model(c*n), max_seq_len(t)
    - Input: x(b, t, c*n)
    - Output: x_pe(b, t, c*n)
    """

    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        pe = torch.zeros(max_seq_len, d_model)
        # pos: (max_seq_len,)
        pos = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos.float() * div_term)
        pe[:, 1::2] = torch.cos(pos.float() * div_term)
        # pe: (1, t, c*n)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x_pe = x + self.pe[:, :x.size(1), :]
        return x_pe


class ScaledDotProductAttention(nn.Module):
    """
    - Params: d_model(c*n), d_k(dimension of keys), d_v(dimension of values), h(number of heads), dropout
    - Input: queries(b, t, c*n), keys(b, t, c*n), values(b, t, c*n), attention_mask, attention_weights
    - Output: x_self_att(b, t, c*n)
    """

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        # queries: (b, t, c*n)
        batch_size, nq = queries.shape[:2]
        # keys: (b, t, c*n)
        nk = keys.shape[1]
        # q, v: (b, t, c*n)(c*n, c*n*h) -> (b, t, c*n*h) -> (b, t, h, c*n) -> (b, h, t, c*n)
        # k: (b, t, c*n)(c*n, c*n*h) -> (b, t, c*n*h) -> (b, t, h, c*n) -> (b, h, c*n, t)
        q = self.fc_q(queries).view(batch_size, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(batch_size, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(batch_size, nk, self.h, self.d_v).permute(0, 2, 1, 3)
        # (b, h, t, c*n)(b, h, c*n, t) -> (b, h, t, t)
        # (b, h, t, t)/(c*n) -> (b, h, t, t)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        att = torch.softmax(att, -1)
        att = self.dropout(att)
        # (b, h, t, t)(b, h, t, c*n) -> (b, h, t, c*n) -> (b, t, h*c*n)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(batch_size, nq, self.h * self.d_v)
        # (b, t, h*c*n)(h*c*n, c*n) -> (b, t, c*n)
        x_self_att = self.fc_o(out)
        return x_self_att


if __name__ == '__main__':
    device = 'cuda:0'
    # inputs, shape = (batch size, channels, time sequence length, number of vertices)
    inputs = torch.randn(32, 2, 12, 207).to(device)
    distance_adj = torch.randn(207, 207).to(device)
    neighbour_adj = torch.randn(207, 207).to(device)
    inout_adj = torch.randn(8, 207, 207).to(device)
    print('Shape of inputs(feature matrix):', inputs.shape)
    print('Shape of Distance-based adjacency matrix:', distance_adj.shape)
    print('Shape of Neighbours-based adjacency matrix:', neighbour_adj.shape)
    print('Shape of InOut-based adjacency matrix:', inout_adj.shape)
    print('------')

    print('Temporal Transformer usage:')
    b, c, t, n = inputs.shape
    x = inputs.reshape(b, t, c * n)

    per = PositionEncoder(c * n, t)
    per.to(device)
    x_pe = per(x)
    print('Output shape of Position Encoder:', x_pe.shape)

    sa = ScaledDotProductAttention(d_model=c * n, d_k=c * n, d_v=c * n, h=8)
    sa.to(device)
    x_self_att = sa(x_pe, x_pe, x_pe)
    x_self_att_reshape = x_self_att.reshape(b, t, c, n)
    print('Output shape of temporal transformer:', x_self_att.shape)
    print('Reshape of above output:', x_self_att_reshape.shape)
