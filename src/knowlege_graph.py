import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.kg_util import get_subgraph_list


class KnowledgeGraph(nn.Module):
    def __init__(self,
                 kg_train_data,
                 kg_val_data,
                 kg_test_data,
                 num_entity,
                 num_relation,
                 num_hop,
                 k,
                 device,
                 n_neg_pos=1):
        super(KnowledgeGraph, self).__init__()

        self.train_data = kg_train_data
        self.val_data = kg_val_data
        self.test_data = kg_test_data

        self.num_relation = num_relation  # TODO: check total number of relations does need to +1?
        self.num_entity = num_entity

        self.n_neg_pos = n_neg_pos
        self.device = device
        self.num_hop = num_hop
        self.k = k

        self.computed_entity_embedidng_KG = None

        self.true_tail = self.get_true_tail(self.train_data)

        self.h_train, self.r_train, self.t_train = self.train_data[:,
                                                                   0], self.train_data[:,
                                                                                       1], self.train_data[:,
                                                                                                           2]
        self.h_val, self.r_val, self.t_val = self.val_data[:,
                                                           0], self.val_data[:,
                                                                             1], self.val_data[:,
                                                                                               2]
        self.h_test, self.r_test, self.t_test = self.test_data[:,
                                                               0], self.test_data[:,
                                                                                  1], self.test_data[:,
                                                                                                     2]

        subgraph_list_self = get_subgraph_list(self.train_data,
                                               self.num_entity, self.num_hop,
                                               self.k, 0, 0)
        self.subgraph_list_kg = subgraph_list_self

    def get_true_tail(self, triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        triples_np = triples.numpy()

        true_tail = {}

        for head, relation, tail in triples_np:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)

        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(
                list(set(true_tail[(head, relation)])))

        return true_tail

    def generate_batch_data(self,
                            h_all,
                            r_all,
                            t_all,
                            batch_size,
                            shuffle=True):

        h_all = torch.unsqueeze(h_all, dim=1)
        r_all = torch.unsqueeze(r_all, dim=1)
        t_all = torch.unsqueeze(t_all, dim=1)

        # generate negative samples
        total_num = h_all.shape[0]
        t_neg = self.get_negative_samples(total_num).to(h_all.device)

        triplets_all = torch.cat([h_all, r_all, t_all, t_neg], dim=-1)  #[B,3]
        triplets_dataloader = DataLoader(triplets_all,
                                         batch_size=batch_size,
                                         shuffle=shuffle)

        return triplets_dataloader

    def get_negative_samples(self, batch_size_each):
        rand_negs = torch.randint(high=self.num_entity,
                                  size=(batch_size_each, ),
                                  device=self.device)  # [b, num_neg = 1]

        rand_negs = rand_negs.view(-1, 1)

        return rand_negs


class MultiLingualKnowledgeGraph(KnowledgeGraph):
    def __init__(self,
                 kg_train_data,
                 kg_val_data,
                 kg_test_data,
                 num_entity,
                 num_relation,
                 is_supporter_kg,
                 entity_id_base,
                 relation_id_base,
                 device,
                 n_neg_pos=1):
        super(MultiLingualKnowledgeGraph,
              self).__init__(kg_train_data, kg_val_data, kg_test_data,
                             num_entity, num_relation, device, n_neg_pos)

        self.entity_id_base = entity_id_base
        self.relation_id_base = relation_id_base

        self.is_supporter_kg = is_supporter_kg


if __name__ == '__main__':
    import os
    import pandas as pd

    triplets_df_mapped_pickle_path = 'src/data/KQAPro/kg/KQAPro_kb_triplets_index_mapped_py36.pkl'

    if os.path.exists(triplets_df_mapped_pickle_path):
        triplets_df_mapped = pd.read_pickle(triplets_df_mapped_pickle_path)

        num_entity = len(
            pd.read_json('src/data/KQAPro/kg/KQAPro_kb_entities.json')[0])
        num_relation = len(
            pd.read_json('src/data/KQAPro/kg/KQAPro_kb_relations.json')[0])
    else:
        triplets_df = pd.read_json(
            'src/data/KQAPro/kg/KQAPro_kb_triplets.json')
        triplets_df.columns = ['v1', 'relation', 'v2']
        triplets_df['v2'] = triplets_df['v2'].values.astype(str)

        entitiy_indices = pd.read_json(
            'src/data/KQAPro/kg/KQAPro_kb_entities.json')[0].to_dict()
        entitiy_indices = {v: k for k, v in entitiy_indices.items()}

        relation_indices = pd.read_json(
            'src/data/KQAPro/kg/KQAPro_kb_relations.json')[0].to_dict()
        relation_indices = {v: k for k, v in relation_indices.items()}

        num_entity = len(entitiy_indices)
        num_relation = len(relation_indices)

        triplets_df_mapped = triplets_df.copy()

        triplets_df_mapped['v1'] = triplets_df_mapped['v1'].map(
            lambda x: entitiy_indices[x])
        triplets_df_mapped['relation'] = triplets_df_mapped['relation'].map(
            lambda x: relation_indices[x])
        triplets_df_mapped['v2'] = triplets_df_mapped['v2'].map(
            lambda x: entitiy_indices[x])

        triplets_df_mapped.to_pickle(triplets_df_mapped_pickle_path)

    triplets_train, triplets_val, triplets_test = np.split(
        triplets_df_mapped.sample(frac=1, random_state=42),
        [int(.6 * len(triplets_df_mapped)),
         int(.8 * len(triplets_df_mapped))])

    print(f'number of All triplets: {len(triplets_df_mapped)}')

    print(f'number of Train triplets: {len(triplets_train)}')
    assert len(triplets_train) == len(triplets_df_mapped) * 0.6

    print(f'number of Test triplets: {len(triplets_test)}')
    assert len(triplets_test) == len(triplets_df_mapped) * 0.2

    print(f'number of Validation triplets: {len(triplets_val)}')
    assert len(triplets_val) == len(triplets_df_mapped) * 0.2

    triplets_train = triplets_train.values.astype(np.int64)
    triplets_val = triplets_val.values.astype(np.int64)
    triplets_test = triplets_test.values.astype(np.int64)

    kg = KnowledgeGraph(torch.LongTensor(triplets_train),
                        torch.LongTensor(triplets_val),
                        torch.LongTensor(triplets_test), num_entity,
                        num_relation, 2, 10, torch.device('cuda:0'))

    batch_size = 32
    kg_batch_generator = kg.generate_batch_data(kg.h_train, kg.r_train,
                                                kg.t_train, batch_size)

    assert len(kg_batch_generator) == len(kg.h_train) // batch_size + 1