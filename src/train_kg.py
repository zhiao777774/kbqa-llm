import os
import logging
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from knowlege_graph import KnowledgeGraph
from model import KGEncoder
from validate import Tester
from utils.data_util import topk_arg_validation, get_device, load_data, save_model
from utils.kg_util import nodes_to_graph, get_negative_samples_graph


def train_kg_batch(args, kg: KnowledgeGraph, model: KGEncoder,
                   optimizer: optim.Optimizer):
    kg_batch_generator = kg.generate_batch_data(kg.h_train,
                                                kg.r_train,
                                                kg.t_train,
                                                batch_size=args.batch_size,
                                                shuffle=True)

    for n_epoch in tqdm(range(args.epoch)):
        kg_loss = []
        for kg_batch_each in tqdm(kg_batch_generator,
                                  desc=f' train {n_epoch}',
                                  leave=False):
            h_graph = nodes_to_graph(kg.subgraph_list_kg,
                                     kg_batch_each[:, 0]).to(args.device)
            t_graph = nodes_to_graph(kg.subgraph_list_kg,
                                     kg_batch_each[:, 2]).to(args.device)
            batch_size = kg_batch_each.shape[0]
            t_neg_index = get_negative_samples_graph(batch_size, kg.num_entity)
            t_neg_graph = nodes_to_graph(kg.subgraph_list_kg,
                                         t_neg_index).to(args.device)

            kg_batch_each = kg_batch_each.to(args.device)

            optimizer.zero_grad()
            loss = model.forward_kg(h_graph, kg_batch_each, t_graph,
                                    t_neg_graph)
            loss.backward()
            optimizer.step()

            kg_loss.append(loss.item())

            del loss
            torch.cuda.empty_cache()

        logging.info('Epoch {:d} [Train KG Loss {:.6f}|'.format(
            n_epoch, np.mean(kg_loss)))


def validate(args: argparse.Namespace, model: KGEncoder, validator: Tester,
             best_test: float, best_val: float, i: int, experiment_id: int):
    model.eval()
    with torch.no_grad():
        metrics_val = validator.test(args.batch_size, k=args.k,
                                     is_val=True)  # validation set
        torch.cuda.empty_cache()
        metrics_test = validator.test(args.batch_size, k=args.k,
                                      is_val=False)  # Test set
        torch.cuda.empty_cache()
        if metrics_val[2] > best_val:
            best_val = metrics_val[2]
            message_best = 'BestVal! Epoch {:04d} [Test seq] | Best mrr {:.6f}| hits1 {:.6f}| hits10 {:.6f}|'.format(
                i, metrics_test[2], metrics_test[0], metrics_test[1])
            logging.info(message_best)

            # save model
            filename = "experiment_" + str(experiment_id)  + "_epoch_" + str(i) + "_MRR_" + str(metrics_test[2]) + "_Hit1_" + str(metrics_test[0]) +\
                       "_Hit10_" + str(metrics_test[1]) + '.ckpt'
            save_model(model, args.model_dir, filename, args)

            if metrics_test[
                    2] > best_test:  # both best test and best val, save one ckpt, but showing the message
                message_best = 'BestTest! Epoch {:04d} [Test seq] | Best mrr {:.6f}| hits1 {:.6f}| hits10 {:.6f}|'.format(
                    i, metrics_test[2], metrics_test[0], metrics_test[1])
                logging.info(message_best)

        elif metrics_test[2] > best_test:
            best_test = metrics_test[2]
            message_best = 'BestTest! Epoch {:04d} [Test seq] | Best mrr {:.6f}| hits1 {:.6f}| hits10 {:.6f}|'.format(
                i, metrics_test[2], metrics_test[0], metrics_test[1])
            logging.info(message_best)

            # save model
            filename = "experiment_" + str(experiment_id)  + "_epoch_" + str(i) + "_MRR_" + str(metrics_test[2]) + "_Hit1_" + str(metrics_test[0]) +\
                       "_Hit10_" + str(metrics_test[1]) + '.ckpt'
            save_model(model, args.model_dir, filename, args)

    return best_test, best_val


def train(args: argparse.Namespace, kg: KnowledgeGraph, model: KGEncoder,
          optimizer: optim.Optimizer, validator: Tester, experiment_id: int):
    best_test = 0
    best_val = 0

    for i in tqdm(range(args.round)):
        logging.info(f'round: {i}')

        model.train()
        train_kg_batch(args, kg, model, optimizer)

        if i % args.val_freq == 0:  # validation
            logging.info(f'=== validation at round {i} ===')

            # model.eval()
            best_test, best_val = validate(args, model, validator, best_test,
                                           best_val, i, experiment_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # training related
    parser.add_argument('--round',
                        default=25,
                        type=int,
                        help="how many rounds to train")
    parser.add_argument(
        '--epoch',
        default=3,
        type=int,
        help="how many knowledge model epochs for the target KG")

    parser.add_argument('--val_freq',
                        default=1,
                        type=int,
                        help="how many rounds to validate")
    parser.add_argument('--lr',
                        '--learning_rate',
                        default=1e-2,
                        type=float,
                        help="learning rate for knowledge model")
    parser.add_argument('-b',
                        '--batch_size',
                        default=200,
                        type=int,
                        help="batch size of queries")
    parser.add_argument('--dropout',
                        type=float,
                        default=0.1,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--l2', type=float, default=0, help='l2 regulazer')

    # KG model related
    parser.add_argument('--transe_margin', default=0.3, type=float)
    parser.add_argument('-d',
                        '--dim',
                        default=256,
                        type=int,
                        help='kg embedding table dimension')
    parser.add_argument('--k',
                        default=10,
                        type=topk_arg_validation,
                        help="how many nominations to consider")
    parser.add_argument('--num_hop', default=2, type=int, help="hop sampling")

    # GNN related
    parser.add_argument('--n_layers_KG',
                        default=2,
                        type=int,
                        help="GNN layer for KGE")
    parser.add_argument('--encoder_hdim_kg',
                        default=256,
                        type=int,
                        help='dimension of GNN for KGC')
    parser.add_argument('--n_heads',
                        default=1,
                        type=int,
                        help="GNN layer for KGE")
    parser.add_argument('--model_name', default='GTrans', type=str)

    # Data related
    parser.add_argument('--pickle_path', type=str, default='')
    parser.add_argument('--entity_json_path', type=str, default='')
    parser.add_argument('--relation_json_path', type=str, default='')
    parser.add_argument('--triplets_json_path', type=str, default='')
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--seed', default=42, type=int, help='random seed')

    return parser.parse_args()


def set_logger(experiment_dir: str):
    """Write logs to checkpoint and console

    Args:
        experiment_dir (str): experiment directory
    """

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    log_file = experiment_dir + 'train.log'
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def main(args: argparse.Namespace):
    device = get_device()
    args.device = device

    # TODO: revise dimension args
    args.entity_dim = args.dim
    args.relation_dim = args.dim

    triplets_train, triplets_val, triplets_test, num_entity, num_relation = load_data(
        args.seed, args.pickle_path, args.entity_json_path,
        args.relation_json_path, args.triplets_json_path)

    kg = KnowledgeGraph(triplets_train, triplets_val, triplets_test,
                        num_entity, num_relation, args.num_hop, args.k,
                        args.device)
    model = KGEncoder(args, num_entity, num_relation).to(args.device)
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.l2)

    print('-' * 30)
    print('model initialization done')

    validator = Tester(kg, model, args.device)

    experiment_id = int(random.SystemRandom().random() * 100000)
    args.model_dir = args.model_dir + str(experiment_id) + '/'
    set_logger(args.model_dir)

    logging.info(str(args))
    print('-' * 30)

    train(args, kg, model, optimizer, validator, experiment_id)


if __name__ == '__main__':
    main(parse_args())