import torch
import torch.nn as nn
from gated_temporal_conv import GatedTemporalConv


class GatedTCN(nn.Module):
    """
    - Params: in_dims, out_dims, dilation, kernel_size, device
    - Input: x(b, c_in, n, t_in)
    - Output: x(b, c_out, n, t_out)
    """

    def __init__(self, in_dims, out_dims, dilation, kernel_size, device):
        super(GatedTCN, self).__init__()
        self.first_dilation = 1
        self.num_layers = len(in_dims)
        ks = kernel_size[1]
        self.paddings = [dilation ** i * (ks - 1) for i in range(self.num_layers)]
        self.modules = []
        for i in range(self.num_layers):
            self.modules.append(GatedTemporalConv(in_dims[i], out_dims[i], dilation ** i, kernel_size).to(device))

    def forward(self, x):
        for i in range(self.num_layers):
            x = nn.functional.pad(x, (self.paddings[i], 0, 0, 0))
            x = self.modules[i](x)
        out = x
        return out


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

    print('Gated GatedTCN Module Usage:')
    gated_tcn = GatedTCN([2, 16, 32], [16, 32, 2], 2, (1, 3), device)
    gated_tcn.to(device)
    # x, shape = (batch size, channels(in), number of vertices, time sequence length(in))
    x = inputs.permute(0, 1, 3, 2)
    # x_gtconv, shape = (batch size, channels(out), number of vertices, time sequence length(out))
    x_gtconv = gated_tcn(x)
    print('Input shape of GatedTCN Module:', x.shape)
    print('Input sequence length:', x.shape[3])
    print('Output shape of GatedTCN Module:', x_gtconv.shape)
    print('Output sequence length:', x_gtconv.shape[3])
