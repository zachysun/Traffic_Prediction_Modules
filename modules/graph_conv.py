import math
import torch
import torch.nn as nn
import torch.nn.init as init
from utils import calculate_scaled_laplacian_torch


class GraphConv(nn.Module):
    """
    - Params: in_channels, out_channels, device, bias
    - Input: x(b, t, n, c_in), L_tilde
    - Output: x(b, t, n, c_out)
    """

    def __init__(self, in_channels, out_channels, device, bias):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels)).to(device)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.out_channels)).to(device)
        else:
            self.register_parameter('bias', None)
        self.init_parameters()

    def init_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, L_tilde):
        x_graph = torch.einsum('hn,btnc->bthc', L_tilde, x)
        graph_conv = torch.einsum('bthc,cj->bthj', x_graph, self.weight)

        if self.bias is not None:
            output = torch.add(graph_conv, self.bias)
        else:
            output = graph_conv

        return output


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

    print('General Graph Convolution Layer Usage:')
    graph_conv = GraphConv(2, 32, device, True)
    L_tilde = calculate_scaled_laplacian_torch(distance_adj, device)
    # x, shape = (batch size, time sequence length, number of vertices, channels(in))
    x = inputs.permute(0, 2, 3, 1)
    print('Input shape of General Graph Convolution Layer:', x.shape)
    # x_graph, shape = (batch size, time sequence length, number of vertices, channels(out))
    x_graph = graph_conv(x, L_tilde)
    print('Output shape of General Graph Convolution Layer:', x_graph.shape)
