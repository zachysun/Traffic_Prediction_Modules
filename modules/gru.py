import math
import torch
import torch.nn as nn


class GRUCell(nn.Module):
    """
    - Params: input_size, hidden_size
    - Input: x(b, t, c*n), hidden(b, hidden_size)
    - Output: h_new(b, hidden_size)
    """

    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        # reset gate
        self.W_ir = nn.Parameter(torch.FloatTensor(input_size, hidden_size))
        self.b_ir = nn.Parameter(torch.FloatTensor(hidden_size))
        self.W_hr = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))

        # update gate
        self.W_iz = nn.Parameter(torch.FloatTensor(input_size, hidden_size))
        self.b_iz = nn.Parameter(torch.FloatTensor(hidden_size))
        self.W_hz = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))

        # candidate hidden state
        self.W_in = nn.Parameter(torch.FloatTensor(input_size, hidden_size))
        self.b_in = nn.Parameter(torch.FloatTensor(hidden_size))
        self.W_hn = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        h_t = hidden
        r = torch.sigmoid(x @ self.W_ir + h_t @ self.W_hr + self.b_ir)
        z = torch.sigmoid(x @ self.W_iz + h_t @ self.W_hz + self.b_iz)
        h_tilde = torch.tanh(x @ self.W_in + r * (h_t @ self.W_hn) + self.b_in)
        h_new = (1 - z) * h_t + z * h_tilde

        return h_new


class GRU(nn.Module):
    """
    - Params: input_size, hidden_size, dropout
    - Input: x(b, t, c*n), hidden(b, hidden_size)
    - Output: output(t, b, hidden_size), final_hidden(b, hidden_size)
    """

    def __init__(self, input_size, hidden_size, dropout):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Linear(input_size, hidden_size)
        self.gru = GRUCell(hidden_size, hidden_size)

    def forward(self, x, hidden):
        output = []
        batch_size, seq_len, input_size = x.size()
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        for i in range(seq_len):
            hidden = self.dropout(hidden)
            hidden = self.gru(embedded[:, i, :], hidden)
            output.append(hidden)

        output = torch.stack(output, dim=0)
        final_hidden = hidden

        return output, final_hidden


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

    print('GRU usage:')
    b, c, t, n = inputs.shape
    hidden_size = 64
    x = inputs.reshape(b, t, c * n)
    init_hidden = torch.zeros(b, hidden_size).to(device)
    gru = GRU(c * n, hidden_size, 0.1)
    gru.to(device)
    x_gru, final_hidden = gru(x, init_hidden)
    print('Shape of output:', x_gru.shape)
    print('Shape of final hidden state:', final_hidden.shape)
