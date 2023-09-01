import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple


def topk_arg_validation(k):
    k = int(k)

    if k < 10:
        raise argparse.ArgumentTypeError('"k" must be greater or equal to 10')

    return k


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_model(model: nn.Module, output_dir: str, filename: str,
               args: argparse.Namespace):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save the weights for the whole model
    ckpt_path = os.path.join(output_dir, filename)
    torch.save({
        'state_dict': model.state_dict(),
        'args': args,
    }, ckpt_path)


def load_data(
    seed: int, pickle_path: str, entity_json_path: str,
    relation_json_path: str, triplets_json_path: str
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, int, int]:
    """Load triplets data and split to train / validation / test.

    Args:
        seed (int): random seed
        pickle_path (str): already mapped triplets data path
        entity_json_path (str): only entity data path
        relation_json_path (str): only relation data path
        triplets_json_path (str): all triplets data path

    Returns:
        Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, int, int]:
            train triplets, validation triplets, test triplets, number of entity, number of relation
    """

    if os.path.exists(pickle_path):
        triplets_df_mapped = pd.read_pickle(pickle_path)

        num_entity = len(pd.read_json(entity_json_path)[0])
        num_relation = len(pd.read_json(relation_json_path)[0])
    else:
        triplets_df = pd.read_json(triplets_json_path)
        triplets_df.columns = ['v1', 'relation', 'v2']
        triplets_df['v2'] = triplets_df['v2'].values.astype(str)

        entitiy_indices = pd.read_json(entity_json_path)[0].to_dict()
        entitiy_indices = {v: k for k, v in entitiy_indices.items()}

        relation_indices = pd.read_json(relation_json_path)[0].to_dict()
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

        triplets_df_mapped.to_pickle(pickle_path)

    # TODO: change to K-fold cross validation
    triplets_train, triplets_val, triplets_test = np.split(
        triplets_df_mapped.sample(frac=1, random_state=seed),
        [int(.8 * len(triplets_df_mapped)),
         int(.9 * len(triplets_df_mapped))])

    print(f'number of All triplets: {len(triplets_df_mapped)}')

    print(f'number of Train triplets: {len(triplets_train)}')
    assert len(triplets_train) == len(triplets_df_mapped) * 0.8

    print(f'number of Test triplets: {len(triplets_test)}')
    assert len(triplets_test) == len(triplets_df_mapped) * 0.1

    print(f'number of Validation triplets: {len(triplets_val)}')
    assert len(triplets_val) == len(triplets_df_mapped) * 0.1

    triplets_train = triplets_train.values.astype(np.int64)
    triplets_val = triplets_val.values.astype(np.int64)
    triplets_test = triplets_test.values.astype(np.int64)

    print(num_entity)
    print(num_relation)

    return torch.LongTensor(triplets_train), torch.LongTensor(
        triplets_val), torch.LongTensor(
            triplets_test), num_entity, num_relation
