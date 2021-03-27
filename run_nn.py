# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
import random
from utils import build_dataset, build_iterator, get_time_dif, set_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='BiLSTM_Att_Cons', type=str, 
                    choices=['BiLSTM_Att_Cons', 'CausalChain'], help='Choose a model')
parser.add_argument('--ratio', default=0.1, type=float, help='Split ratio of training and testing')
parser.add_argument('--charge', default='II-M-N', type=str, choices=['II-M-N', 'F-E', 'E-MPF', 'AP-DD', 'R-K-S'],
                    help='Name of the dataset')
parser.add_argument('--seed', default=1, type=int, help='Random seed of GCI')
parser.add_argument('--cons', default=0, type=float, help='Value of constraint')

args = parser.parse_args()

random.seed(1)

if __name__ == '__main__':
    embedding = 'embeddings.npz'
    model_name = args.model

    x = import_module('models.' + model_name)
    config = x.Config(embedding)
    config = set_parameters(config, args)

    if args.cons > 0:
        config.cons = args.cons

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print("Loading data...")

    vocab, train_data, dev_data, test_data = build_dataset(config, args.ratio, args.charge, model_name, args.seed)
    train_iter = build_iterator(train_data, model_name, config)
    dev_iter = build_iterator(dev_data, model_name, config)
    test_iter = build_iterator(test_data, model_name, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter, model_name, config.cons, args.seed, args.charge)
