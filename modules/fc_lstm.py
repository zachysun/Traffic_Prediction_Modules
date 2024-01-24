"""
Unofficial implementation of paper
'Generating Sequences With Recurrent Neural Networks'
https://arxiv.org/abs/1308.0850
"""
import torch
import torch.nn as nn


class FC_LSTM(nn.Module):
    """
    - Params: input_size(c*n), hidden_size
    - Input: x(b, t, c*n), init_hidden(b, hidden_size), init_memory(hidden_size)
    - Output: hiddens(t, b, hidden_size), memorys(t, b, hidden_size)
    """

    def __init__(self, input_size, hidden_size):
        super(FC_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fc_x = nn.Linear(input_size, 4 * hidden_size)
        self.fc_h = nn.Linear(hidden_size, 4 * hidden_size)
        self.W_ci = nn.Parameter(torch.FloatTensor(hidden_size))
        self.W_cf = nn.Parameter(torch.FloatTensor(hidden_size))
        self.W_co = nn.Parameter(torch.FloatTensor(hidden_size))

    def forward(self, x, init_hidden, init_memory):
        batch_size, seq_len, input_size = x.size()
        hiddens = []
        memorys = []
        hiddens.append(init_hidden)
        memorys.append(init_memory)

        for t in range(seq_len):
            x_t = x[:, t, :]

            x_t_w = self.fc_x(x_t)
            h_t_w = self.fc_h(hiddens[t])
            y = x_t_w + h_t_w
            i, f, o, g = torch.split(y, self.hidden_size, dim=1)

            Ci = torch.mul(self.W_ci, memorys[t])
            Cf = torch.mul(self.W_cf, memorys[t])

            i = torch.sigmoid(i + Ci)
            f = torch.sigmoid(f + Cf)
            g = torch.tanh(g)

            memory = torch.mul(f, memorys[t]) + torch.mul(i, g)
            Co = torch.mul(self.W_co, memory)
            o = torch.sigmoid(o + Co)
            hidden = torch.mul(o, torch.tanh(memory))

            hiddens.append(hidden)
            memorys.append(memory)

        hiddens = torch.stack(hiddens, dim=0)
        memorys = torch.stack(memorys, dim=0)

        return hiddens[1:], memorys[1:]


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

    print('FC-LSTM Layer usage:')
    b, c, t, n = inputs.shape
    x = inputs.reshape(b, t, c * n)
    hidden_size = 64
    init_hidden = torch.zeros(b, hidden_size).to(device)
    init_memory = torch.zeros(b, hidden_size).to(device)
    fc_lstm = FC_LSTM(c*n, hidden_size)
    fc_lstm.to(device)
    x_fc_lstm, memorys = fc_lstm(x, init_hidden, init_memory)
    print('Shape of output(all hidden states):', x_fc_lstm.shape)
    print('Shape of all memory states:', memorys.shape)
