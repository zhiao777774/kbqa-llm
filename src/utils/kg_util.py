import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm


def ranking_all_batch(predicted_t: torch.Tensor,
                      embedding_matrix: torch.Tensor,
                      k: int = None):
    """Compute k-nearest neighbors in a batch
    If k== None, return ranked all candidatees
    otherwise, return top_k candidates

    Args:
        predicted_t (torch.Tensor): predicted tail embedding [b,d]
        embedding_matrix (torch.Tensor): entity embedding matrix [n,d]
        k (int, optional): top k candidates. Defaults to None.

    Returns:
        _type_: _description_
    """

    total_entity = embedding_matrix.shape[0]
    predicted_t = torch.unsqueeze(predicted_t, dim=1)  # [b,1,d]

    # TODO: CUDA out of memory on validation part
    # mabey this is better?
    diff = predicted_t - embedding_matrix.unsqueeze(0)
    distance = torch.norm(diff, dim=2)
    # predicted_t = predicted_t.repeat(1, total_entity, 1)  # [b,n,d]
    # distance = torch.norm(predicted_t - embedding_matrix, dim=2)  # [b,n]

    if not k:
        k = total_entity

    top_k_scores, top_k_indices = torch.topk(-distance, k=k)
    return top_k_indices, top_k_scores


def nodes_to_graph(sub_graph_list, node_index, batch_size=-1):
    one_batch = False
    if batch_size == -1:
        # get embeddings together without batch
        batch_size = node_index.shape[0]
        one_batch = True

    graphs = [sub_graph_list[i.item()] for i in node_index]

    graph_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    if one_batch:
        for one_batch in graph_loader:
            assert one_batch.edge_index.shape[1] == one_batch.edge_attr.shape[
                0]
            return one_batch
    else:
        return graph_loader


def get_negative_samples_graph(batch_size_each, num_entity):
    rand_negs = torch.randint(high=num_entity,
                              size=(batch_size_each, ))  # [b,1]

    return rand_negs


def create_subgraph_list(edge_index, edge_value, total_num_nodes, num_hops, k,
                         node_base, relation_base):
    # Adding self-loop for nodes without edges
    # TODO: here k is for restricting the total number of edges in a subgrap. Wether to remove?
    # TODO padding self=edges for those do not have edges
    sub_graph_list = []
    num_edges = []

    node_list = [i for i in range(total_num_nodes)]

    for i in tqdm(node_list):

        [node_ids, edge_index_each, node_position,
         edge_masks] = k_hop_subgraph([i],
                                      num_hops,
                                      edge_index,
                                      num_nodes=total_num_nodes,
                                      relabel_nodes=True)
        x = node_ids + node_base  # global indexing
        edge_index_each = edge_index_each[:, :k]
        # edge value can be zero!!!!!!!!!!!!!
        edge_value_masked = (edge_value + 1) * edge_masks
        edge_attr = edge_value_masked[edge_value_masked.nonzero(
            as_tuple=True)] - 1
        edge_attr = edge_attr[:k] + relation_base  # global indexing

        assert edge_attr.shape[0] == edge_index_each.shape[1]

        node_position = torch.LongTensor([node_position])
        num_size = torch.LongTensor([len(node_ids)])
        graph_each = Data(x=x,
                          edge_index=edge_index_each,
                          edge_attr=edge_attr,
                          y=node_position,
                          num_size=num_size)
        sub_graph_list.append(graph_each)
        num_edges.append(edge_index_each.shape[1])

    print('Average subgraph edges %.2f' % np.mean(num_edges))

    return sub_graph_list


def get_kg_edges_for_each(train_data):
    train_df = pd.DataFrame(train_data.numpy(),
                            columns=['v1', 'relation', 'v2'])

    # Training data graph construction
    sender_node_list = train_df['v1'].values.astype(np.int64).tolist()
    sender_node_list += train_df['v2'].values.astype(np.int64).tolist()

    receiver_node_list = train_df['v2'].values.astype(np.int64).tolist()
    receiver_node_list += train_df['v1'].values.astype(np.int64).tolist()

    edge_weight_list = train_df['relation'].values.astype(
        np.int64).tolist() + train_df['relation'].values.astype(
            np.int64).tolist()

    edge_index = torch.LongTensor(
        np.vstack((sender_node_list, receiver_node_list)))
    edge_weight = torch.LongTensor(np.asarray(edge_weight_list))

    return edge_index, edge_weight


def get_subgraph_list(train_data, num_entity, num_hop, k, node_base,
                      relation_base):
    edge_index, edge_type = get_kg_edges_for_each(train_data)

    sub_graph_list = create_subgraph_list(edge_index, edge_type, num_entity,
                                          num_hop, k, node_base, relation_base)

    return sub_graph_list


def k_hop_subgraph(node_idx,
                   num_hops,
                   edge_index,
                   relabel_nodes=False,
                   num_nodes=None,
                   flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = num_nodes

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return [subset, edge_index, inv, edge_mask]