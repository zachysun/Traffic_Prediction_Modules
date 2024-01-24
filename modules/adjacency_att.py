import torch
import torch.nn as nn
import torch.nn.functional as F


class AdjAttention(nn.Module):
    """
    - Params: adj_input_dim, adj_hidden_dim(For weights)
    - Input: adj_list:[origin_adj(n, n), origin_adj_2(n, n), dynamic_adj(n, n), cur_io_adj(n, n)]
    - Output: adj_aggregated(n, n)
    """

    def __init__(self, adj_input_dim, adj_hidden_dim):
        super(AdjAttention, self).__init__()

        self.W = nn.Linear(adj_input_dim, adj_hidden_dim)
        self.V = nn.Linear(adj_hidden_dim, 1)

    def forward(self, adj_list):
        # adj_list: a list of adjacency matrices with shape (num_nodes, num_nodes)
        num_nodes, _ = adj_list[0].shape

        # Compute weights for each adjacency matrix
        weights = []
        for adj in adj_list:
            x = F.relu(self.W(adj))
            x = self.V(x)
            x = x.view(num_nodes, 1)
            alpha = x.mean()
            weights.append(alpha)

        weights = F.softmax(torch.Tensor(weights), dim=0)
        # Compute weighted sum of adjacency matrices
        adj_aggregated = torch.zeros_like(adj_list[0])
        for i in range(len(weights)):
            adj_aggregated += adj_list[i] * weights[i]

        return adj_aggregated


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

    print('Multi-adjacency relationship attention mechanism usage:')
    # x, shape = (batch size, time sequence length, number of vertices, channels)
    x = inputs.permute(0, 2, 3, 1)
    dynamic_adj = torch.randn(207, 207).to(device)
    distance_adj_2 = torch.matmul(distance_adj, distance_adj.T)
    adj_list = [distance_adj, distance_adj_2, dynamic_adj, inout_adj[0]]
    adj_att = AdjAttention(207, 128)
    adj_att.to(device)
    adj_aggregated = adj_att(adj_list)
    print('Shape of aggregated adjacency matrix:', adj_aggregated.shape)
