"""
Modified from paper
'Towards Spatio-Temporal Aware Traffic Time Series Forecasting' ICDE'20
https://arxiv.org/abs/2203.15737
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import LinearCustom, ParameterGenerator


class TemporalSelfAttention(nn.Module):
    """
    - Params: in_dim
    - Input: query(b, no_proxies, n, c), key, value, parameters(weights, bias)
    - Output: x
    """

    def __init__(self, in_dim):
        super(TemporalSelfAttention, self).__init__()
        self.K = 8
        self.head_size = int(in_dim // self.K)
        self.key_proj = LinearCustom()
        self.value_proj = LinearCustom()
        self.projection1 = nn.Linear(in_dim, in_dim)
        self.projection2 = nn.Linear(in_dim, in_dim)

    def forward(self, query, key, value, parameters):
        batch_size = query.shape[0]

        # [batch_size, num_step, N, K * head_size]
        key = self.key_proj(key, parameters[0])
        value = self.value_proj(value, parameters[1])

        # [K * batch_size, num_step, N, head_size]
        query = torch.cat(torch.split(query, self.head_size, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_size, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_size, dim=-1), dim=0)

        # query: [K * batch_size, N, -1, head_size]
        # key:   [K * batch_size, N, head_size, num_step]
        # value: [K * batch_size, N, num_step, head_size]
        query = query.permute((0, 2, 1, 3))
        key = key.permute((0, 2, 3, 1))
        value = value.permute((0, 2, 1, 3))

        attention = torch.matmul(query, key)  # [K * batch_size, N, num_step, num_step]
        attention /= (self.head_size ** 0.5)

        # normalize the attention scores
        # attention = self.mask * attention
        attention = F.softmax(attention, dim=-1)

        x = torch.matmul(attention, value)  # [batch_size * head_size, num_step, N, K]
        x = x.permute((0, 2, 1, 3))
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)

        # projection
        x = self.projection1(x)
        x = torch.tanh(x)
        x = self.projection2(x)
        return x


class SpatialSelfAttention(nn.Module):
    """
    - Params: in_dim
    - Input: x, parameters
    - Output: x
    """
    def __init__(self, in_dim):
        super(SpatialSelfAttention, self).__init__()
        self.K = 8
        self.head_size = int(in_dim // self.K)
        self.linear = LinearCustom()
        self.projection1 = nn.Linear(in_dim, in_dim)
        self.projection2 = nn.Linear(in_dim, in_dim)

    def forward(self, x, parameters):
        batch_size = x.shape[0]
        # [batch_size, 1, N, K * head_size]
        # query = self.linear(x, parameters[2])
        key = self.linear(x, parameters[0])
        value = self.linear(x, parameters[1])

        # [K * batch_size, num_step, N, head_size]
        query = torch.cat(torch.split(x, self.head_size, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_size, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_size, dim=-1), dim=0)

        attention = torch.matmul(query, key.transpose(2, 3))  # [K * batch_size, N, num_step, num_step]
        attention /= (self.head_size ** 0.5)

        attention = F.softmax(attention, dim=-1)
        x = torch.matmul(attention, value)  # [batch_size * head_size, num_step, N, K]
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)

        # projection
        x = self.projection1(x)
        x = F.relu(x)
        x = self.projection2(x)
        return x


class WindowAttention(nn.Module):
    """
    - Params: input_dim, num_nodes, cuts, cut_size, no_proxies, device
    - Input: x_in(b, t, n, c or input_dim)
    - Output: x_out(b, cuts, n, c or input dim)
    """

    def __init__(self, input_dim, num_nodes, cuts, cut_size, no_proxies, device):
        super(WindowAttention, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.cuts = cuts
        self.cut_size = cut_size
        self.no_proxies = no_proxies

        self.proxies = nn.Parameter(torch.randn(1, cuts * no_proxies, self.num_nodes, input_dim).to(device),
                                    requires_grad=True).to(device)
        self.temporal_att = TemporalSelfAttention(input_dim)
        self.spatial_att = SpatialSelfAttention(input_dim)

        self.temporal_parameter_generators = nn.ModuleList([
            ParameterGenerator(input_dim=input_dim, output_dim=input_dim, num_nodes=num_nodes,
                               device=device) for _ in range(2)
        ])

        self.spatial_parameter_generators = nn.ModuleList([
            ParameterGenerator(input_dim=input_dim, output_dim=input_dim, num_nodes=num_nodes,
                               device=device) for _ in range(2)
        ])

        self.aggregator = nn.Sequential(*[
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        ])

    def forward(self, x):
        # x, shape = (B, T, N, C)
        batch_size = x.size(0)

        temporal_parameters = [layer(x) for layer in self.temporal_parameter_generators]
        spatial_parameters = [layer(x) for layer in self.spatial_parameter_generators]

        data_concat = []
        out = 0
        for i in range(self.cuts):
            # t, shape = (B, cut_size, N, C)
            t = x[:, i * self.cut_size:(i + 1) * self.cut_size, :, :]

            proxies = self.proxies[:, i * self.no_proxies: (i + 1) * self.no_proxies]
            proxies = proxies.repeat(batch_size, 1, 1, 1) + out
            t = torch.cat([proxies, t], dim=1)

            out = self.temporal_att(t[:, :self.no_proxies, :, :], t, t, temporal_parameters)
            out = self.spatial_att(out, spatial_parameters)
            out = (self.aggregator(out) * out).sum(1, keepdim=True)
            data_concat.append(out)

        return torch.cat(data_concat, dim=1)


if __name__ == '__main__':
    device = 'cuda:0'
    # inputs, shape = (batch size, channels, time sequence length, number of vertices)
    inputs = torch.randn(32, 16, 72, 307).to(device)
    distance_adj = torch.randn(307, 307).to(device)
    neighbour_adj = torch.randn(307, 307).to(device)
    inout_adj = torch.randn(8, 307, 307).to(device)
    print('Shape of inputs(feature matrix):', inputs.shape)
    print('Shape of Distance-based adjacency matrix:', distance_adj.shape)
    print('Shape of Neighbours-based adjacency matrix:', neighbour_adj.shape)
    print('Shape of InOut-based adjacency matrix:', inout_adj.shape)
    print('------')

    print('Window Attention Layer Usage:')
    # x_in, shape = (batch size, time sequence length, number of vertices, channels)
    x_in = inputs.permute(0, 2, 3, 1)
    wa = WindowAttention(input_dim=16, num_nodes=307, cuts=12, cut_size=6, no_proxies=2, device=device)
    wa.to(device)
    x_out = wa(x_in)
    print('Output shape of Window Attention Layer:', x_out.shape)
