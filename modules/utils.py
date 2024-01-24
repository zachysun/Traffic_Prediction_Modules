import torch
import torch.nn as nn

from math import sqrt


# -------------------------- For GCN --------------------------
def calculate_scaled_laplacian_torch(adj, device):
    """
    torch version
    - Params: device
    - Input: adj
    - Output: L_tilde
    """
    D = torch.diag(torch.sum(adj, dim=1))
    L = D - adj

    eigenvalues, _ = torch.linalg.eig(L)
    lambda_max = eigenvalues.real.max()

    return (2 * L) / lambda_max - torch.eye(adj.shape[0]).to(device)


def get_Tk_torch(L_tilde, K, device):
    """
    torch version
    - Params: device
    - Input: L_tilde, K
    - Output: T_ks
    """
    T_ks = []
    N = L_tilde.shape[0]
    T_ks.append(torch.eye(N).to(device))
    T_ks.append(L_tilde)

    for i in range(2, K):
        T_ks.append(2 * L_tilde * T_ks[i - 1] - T_ks[i - 2])

    return T_ks


# -------------------------- For Window Attention Layer --------------------------
class LinearCustom(nn.Module):
    def __init__(self):
        super(LinearCustom, self).__init__()

    def forward(self, inputs, parameters):
        weights, biases = parameters[0], parameters[1]
        if len(weights.shape) > 3:
            return torch.matmul(inputs.unsqueeze(-2), weights.unsqueeze(1).repeat(1, inputs.shape[1], 1, 1, 1)).squeeze(
                -2) + biases.unsqueeze(1).repeat(1, inputs.shape[1], 1, 1)
        return torch.matmul(inputs, weights) + biases


class ParameterGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, num_nodes, device):
        super(ParameterGenerator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.device = device

    def forward(self, x):
        weights = nn.Parameter(torch.randn(x.shape[0], self.num_nodes, self.input_dim, self.output_dim)).to(self.device)
        biases = nn.Parameter(torch.randn(x.shape[0], self.num_nodes, self.output_dim)).to(self.device)

        return weights, biases


# -------------------------- For Spatial-Temporal Identity(STID) --------------------------
class MLPRes(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor):
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        hidden = hidden + input_data
        return hidden


# -------------------------- For Two-Stage Attention Layer --------------------------
class SelfAttentionLayer(nn.Module):
    """
    - Params: in_dim -> int, n_heads -> int, dim_keys -> bool, dim_values -> bool, dropout -> float
    - Input: q(b, l, in_dim), k(b, s, in_dim), v(b, s, in_dim)
    - Output: out(b, l, in_dim)
    """

    def __init__(self, in_dim, n_heads, dim_keys=None, dim_values=None, dropout=0.1):
        super(SelfAttentionLayer, self).__init__()

        dim_keys = dim_keys or int(in_dim // n_heads)
        dim_values = dim_values or int(in_dim // n_heads)

        self.n_heads = n_heads

        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(in_dim, dim_keys * n_heads)
        self.k_proj = nn.Linear(in_dim, dim_keys * n_heads)
        self.v_proj = nn.Linear(in_dim, dim_values * n_heads)

        self.out_proj = nn.Linear(dim_values * n_heads, in_dim)

    def forward(self, q, k, v):
        b, l, _ = q.shape
        _, s, _ = k.shape
        h = self.n_heads

        q = self.q_proj(q).reshape(b, l, h, -1)
        k = self.q_proj(k).reshape(b, s, h, -1)
        v = self.q_proj(v).reshape(b, s, h, -1)

        _, _, _, e = q.shape
        scale = 1. / sqrt(e)

        scores = torch.einsum("blhe,bshe->bhls", q, k)
        attention = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attention, v).contiguous()
        out = out.transpose(2, 1).contiguous()
        out = out.view(b, l, -1)
        out = self.out_proj(out)

        return out
