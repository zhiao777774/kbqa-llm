import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
import math
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_max, scatter_add
from torch_geometric.nn import  RGCNConv, SAGEConv, GATConv


def softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] +
                 1e-16)

    return out


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


class GTrans(MessagePassing):
    def __init__(self,
                 n_heads=2,
                 d_input=6,
                 d_input_edge=6,
                 d_k=6,
                 dropout=0.1,
                 **kwargs):
        super(GTrans, self).__init__(aggr='add', **kwargs)

        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        self.d_input = d_input
        self.d_k = d_k // n_heads
        self.d_q = d_k // n_heads
        self.d_e = d_k // n_heads
        self.d_sqrt = math.sqrt(d_k // n_heads)

        # Attention Layer Initialization
        self.w_k_list = nn.ModuleList([
            nn.Linear(self.d_input, self.d_k, bias=True)
            for i in range(self.n_heads)
        ])
        self.w_q_list = nn.ModuleList([
            nn.Linear(self.d_input, self.d_q, bias=True)
            for i in range(self.n_heads)
        ])
        self.w_v_list = nn.ModuleList([
            nn.Linear(self.d_input, self.d_e, bias=True)
            for i in range(self.n_heads)
        ])
        self.w_transfer = nn.ModuleList([
            nn.Linear(self.d_input + d_input_edge, self.d_k, bias=True)
            for i in range(self.n_heads)
        ])

        # Normalization
        self.layer_norm = nn.LayerNorm(d_input)

    def forward(self, x, edge_index, edge_type, edge_vector):
        '''

        :param x: [b,d], node feature
        :param edge_index: [2,num_edge], edge indexes
        :param edge_type: [num_edge,1], edge type vector
        :param edge_value: [num_edge,d], relation vector
        :return:
        '''
        num_nodes = x.shape[0]
        residual = x
        x = self.layer_norm(x)
        return self.propagate(edge_index,
                              x=x,
                              edge_type=edge_type,
                              edge_vector=edge_vector,
                              residual=residual,
                              num_nodes=num_nodes)

    def message(self, x_j, x_i, edge_index_i, edge_type, edge_vector,
                num_nodes):
        '''
           :param x_j: [num_edge, d] sender
           :param x_i: [num_edge,d]  receiver
           :param edge_index_i:  receiver node list [num_edge]
           :param edges_temporal: [num_edge,d]
           :return:
        '''

        messages = []
        edge_value = edge_type.view(
            -1, 1)  #[num_edge,1] , learnable scalar for each relation type
        for i in range(self.n_heads):
            k_linear = self.w_k_list[i]
            q_linear = self.w_q_list[i]
            v_linear = self.w_v_list[i]
            w_transfer = self.w_transfer[i]

            if not x_j.shape[0] == edge_type.shape[0]:
                print(x_j.shape[0])
                print(edge_type.shape[0])

            x_j_transfer = F.gelu(
                w_transfer(torch.cat([x_j, edge_vector], dim=1)))

            attention = self.each_head_attention(x_j_transfer, k_linear,
                                                 q_linear, x_i,
                                                 edge_value)  # [b,1]

            attention = torch.div(attention, self.d_sqrt)

            attention_norm = softmax(attention, edge_index_i,
                                     num_nodes)  # [num_edge,num_edge]

            sender = v_linear(x_j_transfer)

            message = attention_norm * sender  # [4,3]
            messages.append(message)

        message_all_head = torch.cat(messages, 1)

        return message_all_head

    def each_head_attention(self, x_j_transfer, w_k, w_q, x_i, edge_value):
        x_i = w_q(x_i)  # receiver #[num_edge,d*heads]
        sender = w_k(x_j_transfer)

        # Calculate attention score
        attention = torch.bmm(torch.unsqueeze(sender, 1),
                              torch.unsqueeze(x_i, 2))  #[b,1,1]
        # multiplied by the prior
        edge_value = torch.unsqueeze(edge_value, dim=2)

        attention = attention * edge_value
        return torch.squeeze(attention, 1)  #[b,1]

    def update(self, aggr_out, residual):
        x_new = residual + F.gelu(aggr_out)

        return self.dropout(x_new)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class GeneralConvLayer(nn.Module):
    '''
    general layers
    '''
    def __init__(self, conv_name, in_hid, in_edge, out_hid, n_heads, dropout, num_relations):
        super(GeneralConvLayer, self).__init__()
        self.conv_name = conv_name
        if self.conv_name == 'GTrans':
            self.base_conv = GTrans(n_heads, in_hid, in_edge, out_hid, dropout)
        elif self.conv_name == 'SAGE':
            self.base_conv = SAGEConv(in_hid, out_hid, in_edge)
        elif self.conv_name == 'RGCN':
            self.base_conv = RGCNConv(in_hid, out_hid, num_relations=num_relations)
        elif self.conv_name == 'GAT':
            self.base_conv = GATConv(in_hid, out_hid, heads=n_heads)
        else:
            self.base_conv = GCNConv(in_hid, out_hid)

    def forward(self, x, edge_index, edge_type, edge_vector):
        if self.conv_name == 'GTrans':
            return self.base_conv(x, edge_index, edge_type, edge_vector)
        elif self.conv_name == 'RGCN':
            x = x.to(torch.long)
            return self.base_conv(x, edge_index.unsqueeze(2), edge_type)
        elif self.conv_name == 'GAT':
            return self.base_conv(x, edge_index)
        else:
            return self.base_conv(x, edge_index)
