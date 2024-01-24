import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicAdjGen(nn.Module):
    """
    - Params: nodes_num, all_feature_dim, device
    - Input: x(b, t, n, c)
    - Output: dynamic_adj(n, n)
    """

    def __init__(self, nodes_num, all_feature_dim, device):
        super(DynamicAdjGen, self).__init__()
        self.all_feature_dim = all_feature_dim
        self.nodes_num = nodes_num
        self.node_vec1 = nn.Parameter(torch.FloatTensor(self.all_feature_dim, self.nodes_num // 10)).to(device)
        self.node_vec2 = nn.Parameter(torch.FloatTensor(self.all_feature_dim, self.nodes_num // 10)).to(device)
        self.node_vec3 = nn.Parameter(torch.FloatTensor(self.nodes_num, self.nodes_num)).to(device)
        nn.init.uniform_(self.node_vec1)
        nn.init.uniform_(self.node_vec2)
        nn.init.uniform_(self.node_vec3)

    def forward(self, x):
        b, t, n, c = x.shape
        x = x.reshape(b, n, t * c)
        x_emb_1 = x @ self.node_vec1
        x_emb_2 = x @ self.node_vec2
        x_emb_2_T = torch.einsum('bnc->bcn', x_emb_2)
        x_selfdot = torch.mean(torch.einsum('bnc,bcm->bnm', (x_emb_1, x_emb_2_T)), dim=0)
        x_selfdot = x_selfdot.reshape(n, n)
        dynamic_adj = x_selfdot @ self.node_vec3
        dynamic_adj = F.softmax(dynamic_adj, dim=1)

        return dynamic_adj


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

    print('Dynamic Adjacency Matrix Generation Layer usage:')
    # x, shape = (batch size, time sequence length, number of vertices, channels)
    x = inputs.permute(0, 2, 3, 1)
    b, t, n, c = x.shape
    dy_adj_gen = DynamicAdjGen(n, t*c, device)
    dynamic_adj = dy_adj_gen(x)
    print('Shape of dynamic adjacency matrix:', dynamic_adj.shape)
    distance_adj_2 = torch.matmul(distance_adj, distance_adj.T)
    adj_list = [distance_adj, distance_adj_2, dynamic_adj, inout_adj[0]]
    print('The length of adjacency list:', len(adj_list))
    print('Shape of matrix in adjacency list:', adj_list[0].shape)
