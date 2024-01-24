"""
Modified from paper
'Few-Sample Traffic Prediction with Graph Networks using Locale as Relational Inductive Biases' TITS'22
https://arxiv.org/abs/2203.03965
"""
import torch
import torch.nn as nn
import random


def find_edges_to_nodes(edges_list):
    result = {}
    for edge in edges_list:
        node = edge[1]
        if node in result:
            result[node].append(edge)
        else:
            result[node] = [edge]
    return result


def generate_edges(n):
    edges_list = set()
    while len(edges_list) < n:
        a = random.randint(0, n - 1)
        b = random.randint(0, n - 1)

        if a != b and (a, b) not in edges_list and (b, a) not in edges_list:
            edges_list.add((a, b))

    return list(edges_list)


class GUAN(nn.Module):
    """
    - Params: x_in_channels, x_out_channels, e_in_channels, e_out_channels, edges, edges_dict, device
    - Input: x(b, n, c_in), e(b, n_edge, c_in_edge)
    - Output: x(b, n, c_out), e(b, n_edge, c_out_edge)
    """

    def __init__(self, x_in_channels, x_out_channels, e_in_channels, e_out_channels, edges, edges_dict, device):
        super(GUAN, self).__init__()

        self.device = device
        self.n_edges = len(edges)
        self.edges = edges
        self.edges_dict = edges_dict

        self.ln1 = nn.Linear(e_in_channels + x_in_channels * 2, e_out_channels)
        self.ac1 = nn.ReLU()
        self.ln2 = nn.Linear(1 + x_in_channels, x_out_channels)
        self.ac2 = nn.ReLU()

        self.x_in_channels = x_in_channels
        self.x_out_channels = x_out_channels
        self.e_in_channels = e_in_channels
        self.e_out_channels = e_out_channels

    def forward(self, x, e):
        b, n, _ = x.shape
        # Update edge attributes
        new_e = torch.zeros(b, self.n_edges, self.e_in_channels + self.x_in_channels * 2).to(self.device)
        for edge in self.edges_dict:
            new_e[:, self.edges_dict[edge], :] = torch.cat([e[:, self.edges_dict[edge], :], x[:, edge[0], :],
                                                            x[:, edge[1], :]], dim=1)
        new_e = self.ln1(new_e)
        new_e = self.ac1(new_e)

        # Aggregate edge attributes to node
        node_attributes_add = torch.zeros(b, n, 1).to(self.device)
        edges_to_nodes = find_edges_to_nodes(self.edges)
        for node in edges_to_nodes:
            if edges_to_nodes[node]:
                value_list = [self.edges_dict.get(key) for key in edges_to_nodes[node]]
                node_attributes_add[:, node] = torch.mean(new_e[:, value_list, :], dim=(1, 2)).reshape(-1, 1)

        # Update node attributes
        new_x = torch.cat([x, node_attributes_add], dim=2)
        new_x = self.ln2(new_x)
        new_x = self.ac2(new_x)

        return new_x, new_e


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

    # edge attributes, shape = (batch size, time sequence length, number of vertices, channels(in))
    e = torch.randn(32, 12, 100, 4).to(device)
    # generate edges, (from node_x, to node_y):index
    num_edges = 100
    edges = generate_edges(num_edges)
    edges_dict = {edge: i for i, edge in enumerate(edges)}
    edges_shuffle = edges.copy()
    random.shuffle(edges_shuffle)
    print('Length of edges:', len(edges))

    print('Graph Update and Aggregate Network Usage:')
    # x, shape = (batch size, time sequence length, number of vertices, channels(in))
    x = inputs.permute(0, 2, 3, 1)
    x_t = x[:, 0, :, :]
    e_t = e[:, 0, :, :]
    guan = GUAN(x_in_channels=2, x_out_channels=2, e_in_channels=4, e_out_channels=4,
                edges=edges, edges_dict=edges_dict, device=device)
    guan.to(device)
    new_x_t, new_e_t = guan(x_t, e_t)
    print('Shape of new node attributes:', new_x_t.shape)
    print('Shape of new edge attributes:', new_e_t.shape)
