import time
import pandas as pd
import numpy as np
import logging
import torch
from os.path import join
from tqdm import tqdm

from utils.kg_util import ranking_all_batch


class Tester:
    def __init__(self, target_kg, model, device):
        """
        :param target_kg: KnowledgeGraph object
        :param support_kgs: list[KnowledgeGraph]
        """
        self.target_kg = target_kg
        self.device = device
        self.model = model

    def get_hit_mrr(self, topk_indices_all, ground_truth):

        # ground_truth = ground_truth.repeat(1,kg.num_entity) #[n_test,n]
        zero_tensor = torch.tensor([0]).to(ground_truth.device)
        one_tensor = torch.tensor([1]).to(ground_truth.device)

        # Calculate Hit@1, Hit@10
        hits_1 = torch.where(ground_truth == topk_indices_all[:, [0]],
                             one_tensor, zero_tensor).sum().item()
        hits_10 = torch.where(ground_truth == topk_indices_all[:, :10],
                              one_tensor, zero_tensor).sum().item()

        # Calculate MRR
        gt_expanded = ground_truth.expand_as(topk_indices_all)
        hits = (gt_expanded == topk_indices_all).nonzero()
        ranks = hits[:, -1] + 1
        ranks = ranks.float()
        rranks = torch.reciprocal(ranks)
        mrr = torch.sum(rranks).data / ground_truth.size(0)

        return hits_1, hits_10, mrr

    def test(self, batch_size, k=None, is_val=True):
        """
        # for validation set!!

        Compute Hits@10 on first param.n_test test samples
        :param supporter_kg: needed when mode == KG1 or LINK_REDIRECT. None for KG0 or VOTING
        :param voting_function: used when mode==VOTING. Default: vote by count
        :return:
        """

        time0 = time.time()
        if is_val:
            samples = self.target_kg.h_val.shape[0]
            ground_truth = self.target_kg.t_val.view(-1, 1).to(self.device)
            output_text = "Val:"
            kg_batch_generator = self.target_kg.generate_batch_data(
                self.target_kg.h_val,
                self.target_kg.r_val,
                self.target_kg.t_val,
                batch_size=batch_size,
                shuffle=False)

        else:
            samples = self.target_kg.h_test.shape[0]
            ground_truth = self.target_kg.t_test.view(-1, 1).to(self.device)
            output_text = "Test:"
            kg_batch_generator = self.target_kg.generate_batch_data(
                self.target_kg.h_test,
                self.target_kg.r_test,
                self.target_kg.t_test,
                batch_size=batch_size,
                shuffle=False)

        # if not k: k = self.target_kg.num_entity
        self.pre_compute_all_embeddings(batch_size)  # compute embeddings

        topk_indices_all = []

        for kg_batch_each in tqdm(kg_batch_generator,
                                  desc=f' {"Val" if is_val else "Test"}',
                                  leave=False):
            h_batch = kg_batch_each[:, 0].view(-1)
            r_batch = kg_batch_each[:, 1].to(self.device)  # global index
            h_embedding = self.target_kg.computed_entity_embedidng_KG[
                h_batch, :]
            model_predictions = self.model.predict(h_embedding, r_batch)
            model_predictions = torch.squeeze(model_predictions, dim=1)
            ranking_indices, ranking_scores = ranking_all_batch(
                model_predictions, self.target_kg.computed_entity_embedidng_KG, k)

            topk_indices_all.append(ranking_indices)

        pred_all = torch.cat(topk_indices_all, dim=0)
        # assert pred_all.shape[1] == k
        # assert pred_all.shape[0] == samples

        hits_1_compute, hits_10_compute, mrr = self.get_hit_mrr(
            pred_all, ground_truth)

        hits_1_ratio = hits_1_compute / samples
        hits_10_ratio = hits_10_compute / samples

        logging.info('%s Hits@%d (%d triples): %f' %
                     (output_text, 1, samples, hits_1_ratio))
        logging.info('%s Hits@%d (%d triples): %f' %
                     (output_text, 10, samples, hits_10_ratio))
        logging.info('%s MRR (%d triples): %f' % (output_text, samples, mrr))
        print('time: %s' % (time.time() - time0))

        return hits_1_ratio, hits_10_ratio, mrr

    def pre_compute_all_embeddings(self, batch_size):
        # no gradient compute!
        with torch.no_grad():
            self.target_kg.computed_entity_embedidng_KG = self.model.get_kg_embeddings_matrix(
                self.target_kg, batch_size, self.device, is_kg=True)
