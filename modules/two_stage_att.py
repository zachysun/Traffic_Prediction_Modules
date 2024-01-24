"""
Modified from paper
'Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting' ICLR'23
https://openreview.net/forum?id=vSVLM2j9eie
"""
import torch
import torch.nn as nn

from einops import rearrange, repeat

from utils import SelfAttentionLayer


class TwoStageAttentionLayer(nn.Module):
    """
    - Params: input_length, factor, in_dim, n_heads, device, d_ff, dropout
    - Input: x(b, n, t, c)
    - Output: final_out(b, n, t, c)
    """

    def __init__(self, input_length, factor, in_dim, n_heads, device, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()

        d_ff = d_ff or 4 * in_dim
        self.time_attention = SelfAttentionLayer(in_dim, n_heads, dropout=dropout)
        self.dim_sender = SelfAttentionLayer(in_dim, n_heads, dropout=dropout)
        self.dim_receiver = SelfAttentionLayer(in_dim, n_heads, dropout=dropout)
        self.router = nn.Parameter(torch.randn(input_length, factor, in_dim)).to(device)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm4 = nn.LayerNorm(in_dim)

        self.MLP1 = nn.Sequential(nn.Linear(in_dim, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, in_dim))
        self.MLP2 = nn.Sequential(nn.Linear(in_dim, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, in_dim))

    def forward(self, x):
        batch = x.shape[0]
        time_in = rearrange(x, 'b num_nodes input_length in_dim -> (b num_nodes) input_length in_dim')
        time_enc = self.time_attention(time_in, time_in, time_in)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        dim_send = rearrange(dim_in, '(b num_nodes) input_length in_dim -> (b input_length) num_nodes in_dim', b=batch)
        batch_router = repeat(self.router, 'input_length factor in_dim -> (repeat input_length) factor in_dim',
                              repeat=batch)
        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b input_length) num_nodes in_dim -> b num_nodes input_length in_dim', b=batch)

        return final_out


if __name__ == '__main__':
    device = 'cuda:0'
    # inputs, shape = (batch size, channels, time sequence length, number of vertices)
    inputs = torch.randn(32, 256, 28, 7).to(device)
    distance_adj = torch.randn(7, 7).to(device)
    neighbour_adj = torch.randn(7, 7).to(device)
    inout_adj = torch.randn(8, 7, 7).to(device)
    print('Shape of inputs(feature matrix):', inputs.shape)
    print('Shape of Distance-based adjacency matrix:', distance_adj.shape)
    print('Shape of Neighbours-based adjacency matrix:', neighbour_adj.shape)
    print('Shape of InOut-based adjacency matrix:', inout_adj.shape)
    print('------')

    print('Two-Stage Attention Layer Usage:')
    # x_in, shape = (batch size, number of vertices, time sequence length, channels)
    x_in = inputs.permute(0, 3, 2, 1)
    _, _, input_length, in_dim = x_in.shape
    factor = 10
    n_heads = 4
    tsa = TwoStageAttentionLayer(input_length, factor, in_dim, n_heads, device, d_ff=None, dropout=0.1)
    tsa.to(device)
    outs = tsa(x_in)
    print('Output shape of Two-Stage Attention Layer:', outs.shape)
