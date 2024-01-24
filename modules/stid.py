"""
Modified from paper
'Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting' CIKM'22
https://arxiv.org/abs/2208.05233
"""
import torch
from torch import nn
from utils import MLPRes


class STID(nn.Module):
    """
    - Params: num_nodes, node_dim, input_length, input_dim, embed_dim, output_length, num_layer,
              temp_dim_tid, temp_dim_diw, time_of_day_size, day_of_week_size, if_T_i_D,
              if_D_i_W, if_node
    - Input: x(b, t, n, c_in) c_in(maybe with tid and diw data)
    - Output: x(b, t, n, c_out) c_out(without tid and diw data)
    """

    def __init__(self, num_nodes, node_dim, input_length, input_dim, embed_dim, output_length, num_layer,
                 temp_dim_tid, temp_dim_diw, time_of_day_size, day_of_week_size, if_T_i_D,
                 if_D_i_W, if_node):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.input_length = input_length
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.output_length = output_length
        self.num_layer = num_layer
        self.temp_dim_tid = temp_dim_tid
        self.temp_dim_diw = temp_dim_diw
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size

        self.if_time_in_day = if_T_i_D
        self.if_day_in_week = if_D_i_W
        self.if_spatial = if_node

        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        self.time_series_emb_layer = nn.Conv2d(in_channels=self.input_dim * self.input_length, out_channels=self.embed_dim,
                                               kernel_size=(1, 1), bias=True)
        self.hidden_dim = self.embed_dim + self.node_dim * int(self.if_spatial) + \
                          self.temp_dim_tid * int(self.if_time_in_day) + \
                          self.temp_dim_diw * int(self.if_day_in_week)
        self.encoder = nn.Sequential(
            *[MLPRes(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )
        self.regression_layer = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.output_length,
                                          kernel_size=(1, 1), bias=True)

    def forward(self, x):
        input_data = x[..., range(self.input_dim)]

        if self.if_time_in_day:
            t_i_d_data = x[..., 1]
            time_in_day_emb = self.time_in_day_emb[
                (t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)
            ]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = x[..., 2]
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        node_emb = []
        if self.if_spatial:
            node_emb.append(self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))

        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)

        hidden = self.encoder(hidden)

        out = self.regression_layer(hidden)

        return out


if __name__ == '__main__':
    device = 'cuda:0'
    # inputs, shape = (batch size, channels, time sequence length, number of vertices)
    inputs = torch.rand(32, 3, 12, 307).to(device)
    distance_adj = torch.randn(307, 307).to(device)
    neighbour_adj = torch.randn(307, 307).to(device)
    inout_adj = torch.randn(8, 307, 307).to(device)
    print('Shape of inputs(feature matrix):', inputs.shape)
    print('Shape of Distance-based adjacency matrix:', distance_adj.shape)
    print('Shape of Neighbours-based adjacency matrix:', neighbour_adj.shape)
    print('Shape of InOut-based adjacency matrix:', inout_adj.shape)
    print('------')

    print('Spatial-Temporal Identity Usage:')
    # x_history, shape = (batch size, time sequence length, number of vertices, channels(including tid and diw data))
    x_history = inputs.permute(0, 2, 3, 1)
    stid = STID(num_nodes=307, node_dim=32, input_length=12, input_dim=3, embed_dim=32, output_length=12, num_layer=3,
                temp_dim_tid=32, temp_dim_diw=32, time_of_day_size=288, day_of_week_size=7, if_T_i_D=True,
                if_D_i_W=True, if_node=True)
    stid.to(device)
    x_out = stid(x_history)
    print('Output shape of STID:', x_out.shape)
