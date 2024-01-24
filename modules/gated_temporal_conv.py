import torch
import torch.nn as nn


class GatedTemporalConv(nn.Module):
    """
    - Params: in_dim, out_dim, dilation, kernel_size
    - Input: x(b, c_in, n, t_in)
    - Output: x(b, c_out, n, t_out)
    """

    def __init__(self, in_dim, out_dim, dilation, kernel_size):
        super(GatedTemporalConv, self).__init__()
        self.filter_convs = nn.Conv2d(in_channels=in_dim,
                                      out_channels=out_dim,
                                      kernel_size=kernel_size, dilation=dilation)

        self.gated_convs = nn.Conv2d(in_channels=in_dim,
                                     out_channels=out_dim,
                                     kernel_size=kernel_size, dilation=dilation)

    def forward(self, x):
        filter = self.filter_convs(x)
        filter = torch.tanh(filter)
        gate = self.gated_convs(x)
        gate = torch.sigmoid(gate)
        x = filter * gate
        return x


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

    print('Gated Temporal Convolution Layer Usage:')
    gated_tconv = GatedTemporalConv(2, 2, 2, (1, 3))
    gated_tconv.to(device)
    # x, shape = (batch size, channels(in), number of vertices, time sequence length(in))
    x = inputs.permute(0, 1, 3, 2)
    # x_gtconv, shape = (batch size, channels(out), number of vertices, time sequence length(out))
    x_gtconv = gated_tconv(x)
    print('Input shape of Gated Temporal Convolution Layer:', x.shape)
    print('Input sequence length:', x.shape[3])
    print('Output shape of Gated Temporal Convolution Layer:', x_gtconv.shape)
    print('Output sequence length:', x_gtconv.shape[3])
