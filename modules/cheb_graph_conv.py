import math
import torch
import torch.nn as nn
import torch.nn.init as init
from utils import calculate_scaled_laplacian_torch, get_Tk_torch


class ChebGraphConv(nn.Module):
    """
    - Params: K, in_channels, out_channels, device, bias
    - Input: x(b, t, n, c_in), Tks
    - Output: x(b, t, n, c_out)
    """

    def __init__(self, K, in_channels, out_channels, device, bias):
        super(ChebGraphConv, self).__init__()
        self.K = K
        self.device = device
        self.out_channels = out_channels
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels)).to(device) for _ in range(K)]
        )
        if bias:
            self.bias = nn.ParameterList(
                [nn.Parameter(torch.FloatTensor(out_channels)).to(device) for _ in range(K)]
            )
        else:
            self.register_parameter('bias', None)
        self.init_parameters()

    def init_parameters(self):
        for i in range(self.K):
            init.kaiming_uniform_(self.Theta[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.Theta[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.bias[i], -bound, bound)

    def forward(self, x, Tks):
        batch_size, seq_length, num_of_vertices, in_channels = x.shape
        output = torch.zeros(batch_size, seq_length, num_of_vertices, self.out_channels).to(self.device)

        for k in range(self.K):
            T_k = Tks[k]
            theta_k = self.Theta[k]
            if T_k.dim() == 2:
                temp = torch.einsum('nn,btnc->btnc', (T_k, x))
            else:
                temp = torch.einsum('bnn,btnc->btnc', (T_k, x))
            output = output + torch.einsum('btnc,co->btno', (temp, theta_k))

            if self.bias is not None:
                output = torch.add(output, self.bias[k])
            else:
                output = output

        return torch.Tensor(output)


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

    print('Chebyshev Graph Convolution Layer Usage:')
    K = 3
    cheb_conv = ChebGraphConv(K, 2, 32, device, False)
    L_tilde = calculate_scaled_laplacian_torch(distance_adj, device)
    Tks = get_Tk_torch(L_tilde, K, device)
    # x, shape = (batch size, time sequence length, number of vertices, channels(in))
    x = inputs.permute(0, 2, 3, 1)
    print('Input shape of Chebyshev Graph Convolution Layer:', x.shape)
    # x_chebgraph, shape = (batch size, time sequence length, number of vertices, channels(out))
    x_chebgraph = cheb_conv(x, Tks)
    print('Output shape of Chebyshev Graph Convolution Layer:', x_chebgraph.shape)
