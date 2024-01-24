import math
import torch
import torch.nn as nn
import torch.nn.init as init
from utils import calculate_scaled_laplacian_torch, get_Tk_torch


class DiffusionConv(nn.Module):
    """
    - Params: max_diffusion_step, in_channels, out_channels, supports, device, bias
    - Input: x(b, n, c_in)
    - Output: x(b, n, c_out)
    """

    def __init__(self, max_diffusion_step, in_channels, out_channels, supports, device, bias):
        super(DiffusionConv, self).__init__()
        self.max_diffusion_step = max_diffusion_step
        self.supports = supports
        self.num_matrices = self.max_diffusion_step * len(self.supports) + 1
        self.Theta = nn.Parameter(torch.FloatTensor(self.num_matrices * in_channels, out_channels)).to(device)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels)).to(device)
        else:
            self.register_parameter('bias', None)
        self.init_parameters()

    def init_parameters(self):
        init.kaiming_uniform_(self.Theta, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.Theta)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        b, n, c = x.shape
        x0 = x.reshape(n, c * b)
        x = torch.unsqueeze(x0, 0)

        if self.max_diffusion_step == 0:
            pass
        else:
            for support in self.supports:
                x1 = torch.sparse.mm(support, x0)
                x = torch.cat([x, x1.unsqueeze(0)], dim=0)

                for i in range(2, self.max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = torch.cat([x, x2.unsqueeze(0)], dim=0)
                    x1, x0 = x2, x1

        x = x.reshape(self.num_matrices, n, c, b)
        x = x.permute(3, 1, 2, 0)  # x, shape = (b, n, c_in, num_matrices)
        x = x.reshape(b, n, c * self.num_matrices)
        x = torch.matmul(x, self.Theta)  # x, shape = (b, n, c_out)
        if self.bias is not None:
            x += self.bias
        else:
            x = x

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

    print('Diffusion Convolution Layer Usage:')
    L_tilde = calculate_scaled_laplacian_torch(distance_adj, device)
    K = 3
    Tks = get_Tk_torch(L_tilde, K, device)
    # x, shape = (batch size, time sequence length, number of vertices, channels(in))
    x = inputs.permute(0, 2, 3, 1)
    b, t, n, c = x.shape
    x_t = x[:, 0, :, :]
    dvonv = DiffusionConv(2, c, 64, Tks, device, False)
    x_dconv = dvonv(x_t)
    print('Output shape of Diffusion Convolution Layer(input: one time step of x):', x_dconv.shape)
