import torch
import torch.nn as nn

from .layers import GeneralConvLayer
from utils.kg_util import nodes_to_graph


class KGEncoder(nn.Module):
    def __init__(self, args, num_entity, num_relation):
        super(KGEncoder, self).__init__()

        self.args = args
        self.batch_size = args.batch_size
        self.total_num_entity = num_entity

        self.num_relations = num_relation

        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.device = args.device
        self.criterion_KG = nn.MarginRankingLoss(margin=args.transe_margin,
                                                 reduction='mean')

        # 1. Embedding initialization
        self.entity_embedding_layer = nn.Embedding(self.total_num_entity,
                                                   self.entity_dim)
        nn.init.xavier_uniform_(self.entity_embedding_layer.weight)

        self.rel_embedding_layer = nn.Embedding(self.num_relations,
                                                self.relation_dim)
        nn.init.xavier_uniform_(self.rel_embedding_layer.weight)

        self.relation_prior = nn.Embedding(self.num_relations, 1)
        nn.init.xavier_uniform_(self.relation_prior.weight)

        # 2. GNN initialization
        self.encoder_KG = GNN(in_dim=args.entity_dim,
                              in_edge_dim=args.relation_dim,
                              n_hid=args.encoder_hdim_kg,
                              out_dim=args.entity_dim,
                              n_heads=args.n_heads,
                              n_layers=args.n_layers_KG,
                              dropout=args.dropout,
                              conv_name=args.model_name,
                              num_relations=self.num_relations)

    def forward_GNN_embedding(self, graph_input, GNN):
        # Original GNN implementation
        x_features = self.entity_embedding_layer(
            graph_input.x)  # [num_nodes,d]
        edge_index = graph_input.edge_index
        edge_type_vector = self.relation_prior(
            graph_input.edge_attr)  # [num_edge]

        edge_vector_embedding = self.rel_embedding_layer(graph_input.edge_attr)
        x_gnn_kg_output = GNN(x_features, edge_index, edge_type_vector,
                              edge_vector_embedding, graph_input.y,
                              graph_input.num_size)  # [N,d]

        return x_gnn_kg_output

    def forward_kg(self, h_graph, sample, t_graph, t_neg_graph):
        h = self.forward_GNN_embedding(h_graph,
                                       self.encoder_KG).unsqueeze(1)  #[b,1,D]

        r = self.rel_embedding_layer(sample[:, 1]).unsqueeze(1)  #[b,1,D]

        t = self.forward_GNN_embedding(t_graph,
                                       self.encoder_KG).unsqueeze(1)  #[b,1,D]

        t_neg = self.forward_GNN_embedding(t_neg_graph,
                                           self.encoder_KG).unsqueeze(1)

        projected_t = self.project_t([h, r])
        pos_loss = self.define_loss([t, projected_t])  ## [b,1]

        neg_losses = self.define_loss([t, t_neg])  # [b,num_neg]
        neg_loss = torch.mean(neg_losses, dim=-1)  # TransE

        target = torch.tensor(
            [-1], dtype=torch.long,
            device=self.device)  # -1 means we expect x1 - x2 < 0
        total_loss = self.criterion_KG(pos_loss, neg_loss, target)

        return total_loss

    def project_t(self, hr):
        """calculate the projected tail vector (use in TransE)

        Args:
            hr (_type_): embeddings of head and relation

        Returns:
            _type_: _description_
        """

        return hr[0] + hr[1]

    def predict(self, h_emb, r):
        # Support batching.
        # (h,r) (r is index_only) -> projected t vector
        entity_dim = h_emb.shape[1]
        h = h_emb.view(-1, entity_dim).unsqueeze(1)
        r = self.rel_embedding_layer(r).unsqueeze(1)
        projected_t = self.project_t([h, r])

        return projected_t

    def define_loss(self, t_true_pred):
        t_true = t_true_pred[0]
        t_pred = t_true_pred[1]
        return torch.norm(t_true - t_pred + 1e-8,
                          dim=2)  # input shape: (None, 1, dim)

    def get_kg_embeddings_matrix(self, kg, batch_size, device, is_kg=True):
        '''
        Compute the entity_embedding matrixes in advance, based on self.encoder_GNN/ align_GNN
        :param kg:
        :param batch_size:
        :param device:
        :return:
        '''
        with torch.no_grad():
            node_index_tensor = torch.LongTensor(
                [i for i in range(kg.num_entity)])
            graphs = nodes_to_graph(kg.subgraph_list_kg, node_index_tensor,
                                    batch_size)

            embedding_list = []
            if is_kg:
                for graph_batch in graphs:
                    assert graph_batch.edge_index.shape[
                        1] == graph_batch.edge_attr.shape[0]
                    graph_batch = graph_batch.to(
                        device)  # only used to retrive relations
                    node_embeddings = self.forward_GNN_embedding(
                        graph_batch, self.encoder_KG)
                    embedding_list.append(node_embeddings)
            else:
                for graph_batch in graphs:
                    assert graph_batch.edge_index.shape[
                        1] == graph_batch.edge_attr.shape[0]
                    graph_batch = graph_batch.to(
                        device)  # only used to retrive relations
                    node_embeddings = self.forward_GNN_embedding(
                        graph_batch, self.encoder_align)
                    embedding_list.append(node_embeddings)

            embedding_table = torch.cat(embedding_list,
                                        dim=0).to(device)  # [n,d]

        return embedding_table


class GNN(nn.Module):
    '''
    wrap up multiple layers
    '''
    def __init__(self,
                 in_dim,
                 in_edge_dim,
                 n_hid,
                 out_dim,
                 n_heads,
                 n_layers,
                 num_relations,
                 dropout=0.2,
                 conv_name='GTrans'):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.in_dim = in_dim
        self.in_edge_dim = in_edge_dim
        self.n_hid = n_hid
        self.drop = nn.Dropout(dropout)

        for l in range(n_layers):
            self.gcs.append(
                GeneralConvLayer(conv_name,
                                 n_hid,
                                 in_edge_dim,
                                 n_hid,
                                 n_heads,
                                 dropout,
                                 num_relations=num_relations))

    def forward(self,
                x,
                edge_index,
                edge_type,
                edge_vector,
                y=None,
                s=None):  # aggregation part

        h_0 = x
        h_t = self.drop(h_0)

        for gc in self.gcs:
            h_t = gc(h_t, edge_index, edge_type, edge_vector)  # [num_nodes,d]

        if y != None:
            true_indexes = self.get_real_index(y, s)
            h_t_real = torch.index_select(h_t, 0, true_indexes)

        return h_t_real

    def get_real_index(self, y, s):
        total_graph = y.shape[0]
        node_base = torch.LongTensor([0]).to(y.device)
        true_index = []

        for i in range(total_graph):
            true_index_each = node_base + y[i]
            node_base += s[i]
            true_index.append(true_index_each.view(-1, 1))
        true_index = torch.cat(true_index).to(y.device)
        true_index = true_index.view(-1)

        return true_index