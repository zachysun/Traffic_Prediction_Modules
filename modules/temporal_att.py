"""
Modified from paper
"ASTGCN: Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting"
https://ojs.aaai.org/index.php/AAAI/article/view/3881
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionLayer(nn.Module):
    """
    - Params: in_channels, num_of_vertices, in_seq_length, device
    - Input: x(b, n, c, t)
    - Output: T_normalized(b, t, t)
    """

    def __init__(self, in_channels, num_of_vertices, in_seq_length, device):
        super(TemporalAttentionLayer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(device))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(device))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(device))
        self.be = nn.Parameter(torch.FloatTensor(1, in_seq_length, in_seq_length).to(device))
        self.Ve = nn.Parameter(torch.FloatTensor(in_seq_length, in_seq_length).to(device))

    def forward(self, x):
        # (b, n, c, t) -> (b, t, c, n)
        # (b, t, c, n)(n) -> (b, t, c)
        # lhs: (b, t, c)(c, n) -> (b, t, n)
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # rhs: (c)(b, n, c, t) -> (b, n, t)
        rhs = torch.matmul(self.U3, x)
        # product: (b, t, n)(b, n, t) -> (b, t, t)
        product = torch.matmul(lhs, rhs)
        # (b, t, t)+(1->b, t, t) -> (b, t, t)
        # T: (t, t)(b, t, t) -> (b, t, t)
        T = torch.matmul(self.Ve, torch.sigmoid(product + self.be))
        T_normalized = F.softmax(T, dim=1)

        return T_normalized


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

    print('Temporal Attention Layer Usage:')
    t_att_layer = TemporalAttentionLayer(2, 207, 12, device)
    x = inputs.permute(0, 3, 1, 2)
    batch_size, num_of_vertices, channels, in_seq_length = x.shape
    temporal_att = t_att_layer(x)
    print('Output shape of Temporal Attention Layer:', temporal_att.shape)
    # (b, n, c, t) -> (b, n*c, t)
    # (b, n*c, t)(b, t, t) -> (b, n*c, t)
    # (b, n*c, t) -> (b, n, c, t)
    x_tatt = torch.matmul(x.reshape(batch_size, -1, in_seq_length), temporal_att).reshape(batch_size,
                                                                                          num_of_vertices,
                                                                                          channels,
                                                                                          in_seq_length)
    print('Shape of x with temporal attention layer:', x_tatt.shape)
