"""
Modified from ASTGCN: Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting
https://ojs.aaai.org/index.php/AAAI/article/view/3881
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from cheb_graph_conv import ChebGraphConv
from utils import calculate_scaled_laplacian_torch, get_Tk_torch


class SpatialAttentionLayer(nn.Module):
    """
    - Params: in_channels, num_of_vertices, in_seq_length, device
    - Input: x(b, n, c, t)
    - Output: S_normalized(b, n, n)
    """

    def __init__(self, in_channels, num_of_vertices, in_seq_length, device):
        super(SpatialAttentionLayer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(in_seq_length).to(device))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, in_seq_length).to(device))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(device))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(device))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(device))

    def forward(self, x):
        # (b, n, c, t)(t) -> (b, n, c)
        # lhs: (b, n, c)(c, t) -> (b, n, t)
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)
        # (c)(b, n, c, t) -> (b, n, t)
        # rhs: (b, n, t) -> (b, t, n)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)
        # product: (b, n, t)(b, t, n) -> (b, n, n)
        product = torch.matmul(lhs, rhs)
        # (b, n, n)+(1->b, n, n) -> (b, n, n)
        # S: (n, n)(b, n, n) -> (b, n, n)
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))
        S_normalized = F.softmax(S, dim=1)

        return S_normalized


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

    print('Spatial Attention Layer Usage:')
    K = 3
    x = inputs.permute(0, 3, 1, 2)
    s_att_layer = SpatialAttentionLayer(2, 207, 12, device)
    spatial_att = s_att_layer(x)
    print('Output shape of Spatial Attention Layer:', spatial_att.shape)
    L_tilde = calculate_scaled_laplacian_torch(distance_adj, device)
    Tks = get_Tk_torch(L_tilde, K, device)
    Tks_att = []
    for i in range(K):
        Tks_att.append(torch.matmul(Tks[i], spatial_att))
    print('Shape of Tks with Spatial Attention:', Tks_att[0].shape)
    # x_in, shape = (batch size, time sequence length, number of vertices, channels)
    x_in = inputs.permute(0, 2, 3, 1)
    cheb_conv = ChebGraphConv(K, 2, 32, device, False)
    x_chebgraph_satt = cheb_conv(x_in, Tks_att)
    print('Output shape of Chebyshev GCN Module(with Tks with Spatial Attention):', x_chebgraph_satt.shape)
