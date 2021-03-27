# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
import time
from datetime import timedelta
import random

random.seed(1)
np.random.seed(1)
MAX_VOCAB_SIZE = 10000
UNK, PAD = '<UNK>', '<PAD>'


def build_dataset(config, ratio, charge, model_name, seed):
    vocab = pkl.load(open(config.vocab_path, 'rb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(text, labels, word_idx, word_key, chains, model_name):
        contents = []
        for i in range(len(text)):
            if model_name == 'BiLSTM_Att_Cons' or model_name == 'BiLSTM_Att' or model_name == 'BiLSTM':
                mask = [0] * pad_size
                token = text[i]
                label = np.argmax(labels[i])
                words_line = []
                seq_len = len(token)
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size

                for word in token:
                    idx = vocab.get(word, vocab.get(UNK))
                    words_line.append(idx)

                words_line_key = [[factor_num] * pad_size] * len(factor_list)
                att_weights_temp = [-10000.0] * pad_size

                assert len(word_key[i]) == len(word_idx[i])

                for j, key in enumerate(word_key[i]):
                    if word_idx[i][j] < pad_size:
                        for p in range(len(factor_list)):
                            if key in factor_list[p]:
                                l = factor_list[p]
                                num_l = -1
                                for k in range(len(l)):
                                    if key == l[k]:
                                        num_l = k
                                        break
                                words_line_key[p][word_idx[i][j]] = key
                                if int(label) == p:
                                    att_weights_temp[word_idx[i][j]] = strength[p][num_l]

                att_weights = []
                sum_int = 0.0
                for j in range(pad_size):
                    if att_weights_temp[j] > -10000 and att_weights_temp[j] < 10:
                        sum_int += np.exp(att_weights_temp[j])
                for j in range(pad_size):
                    if att_weights_temp[j] > -10000 and att_weights_temp[j] < 10:
                        att_weights.append(np.exp(att_weights_temp[j]) * 1. / sum_int)
                    else:
                        att_weights.append(0.0)
                if sum_int != 0.0:
                    sum_int = 1.0

                contents.append(
                    (words_line, words_line_key, att_weights, sum_int, int(label), seq_len))

            elif model_name == 'CausalChain':
                t = chains[i]
                chain_content = []
                scores = []
                for li, score in t:
                    scores.append(score)
                    chain_content.append(np.array(text[i])[li].tolist())
                label = np.argmax(labels[i])
                tokens = chain_content
                words_lines = []
                seq_lens = []
                it = 0
                chain_length = 8
                masks = []
                for token in tokens:
                    words_line = []
                    seq_len = len(token)
                    mask = [0] * chain_length
                    if pad_size:
                        if len(token) < chain_length:
                            token.extend([PAD] * (chain_length - len(token)))
                            if len(token) != 0:
                                mask[len(token) - 1] = 1
                        else:
                            token = token[:chain_length]
                            seq_len = chain_length
                            mask[chain_length - 1] = 1
                    seq_lens.append(seq_len)

                    for word in token:
                        idx = vocab.get(word, vocab.get(UNK))
                        words_line.append(idx)
                    words_lines.append(words_line)
                    masks.append(mask)
                    it += 1
                    if it >= chain_num:
                        break

                for t1 in range(chain_num - len(words_lines)):
                    words_lines.append([10001] * chain_length)
                    seq_lens.append(0)  
                    masks.append([0] * chain_length)
                    scores.append(0)

                if len(scores) > chain_num:
                    scores = scores[:chain_num]
                scores = np.array(scores)
                if np.sum(scores) > 0:
                    scores = scores / np.sum(scores)

                contents.append((words_lines, int(label), seq_lens, masks, scores))

        return contents

    prefix = 'nn_data'

    if ratio == 0.1:
        ratio_str = '19'
    elif ratio == 0.01:
        ratio_str = '199'
    elif ratio == 0.05:
        ratio_str = '119'
    elif ratio == 0.3:
        ratio_str = '37'
    elif ratio == 0.5:
        ratio_str = '55'

    filename = prefix + '_' + ratio_str + '_' + charge + '_' + str(seed) + '.pkl'

    chain_num = config.chain
    batch_size = config.batch_size
    factor_num = config.factor_num
    pad_size = config.pad_size

    with open('data/' + filename, 'rb') as f:
        data = pkl.load(f)

    strength = data['strength']
    factor_list_t = data['factor_list']

    factor_list = [[int(y[1:]) for y in x] for x in factor_list_t]

    text_train = data['text_train']
    labels_train = data['labels_train']
    text_test = data['text_test']
    labels_test = data['labels_test']
    word_idx_train = data['word_idx_train']
    word_idx_test = data['word_idx_test']
    word_key_train = data['word_key_train']
    word_key_test = data['word_key_test']
    chains_train = data['chains_train']
    chains_test = data['chains_test']

    cnt = int(len(text_train) / 10)
    train_split = cnt * 9
    random_idx = np.arange(len(text_train))
    np.random.shuffle(random_idx)
    train_idx = random_idx[:train_split]
    val_idx = random_idx[train_split:]

    train = load_dataset(text_train[train_idx], labels_train[train_idx], word_idx_train[train_idx],
                         word_key_train[train_idx], chains_train[train_idx], model_name)
    dev = load_dataset(text_train[val_idx], labels_train[val_idx], word_idx_train[val_idx],
                       word_key_train[val_idx], chains_train[val_idx], model_name)
    test = load_dataset(text_test, labels_test, word_idx_test, word_key_test, chains_test, model_name)

    if not os.path.exists('saved_dict/'):
        os.makedirs('saved_dict/')
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, model_name, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.model = model_name
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas, model_name):
        if model_name == "BiLSTM_Att_Cons" or model_name == 'BiLSTM_Att' or model_name == 'BiLSTM':
            x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
            key = torch.LongTensor([_[1] for _ in datas]).to(self.device)
            att_weights = torch.Tensor([_[2] for _ in datas]).to(self.device)
            flag = torch.Tensor([_[3] for _ in datas]).to(self.device)
            y = torch.LongTensor([_[4] for _ in datas]).to(self.device)
            seq_len = torch.LongTensor([_[5] for _ in datas]).to(self.device)
            return (x, key, att_weights, flag, seq_len), y
        else:
            x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
            y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
            seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
            mask = torch.Tensor([_[3] for _ in datas]).to(self.device)
            scores = torch.FloatTensor([_[4] for _ in datas]).to(self.device)
            return (x, mask, scores, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches, self.model)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches, self.model)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, model_name, config):
    iter = DatasetIterater(dataset, model_name, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def set_parameters(config, args):
    charge = args.charge
    ratio = args.ratio

    ratio2id = {0.01: 0, 0.05: 1, 0.1: 2, 0.3: 3, 0.5: 4}
    charge2id = {'F-E': 0, 'E-MPF': 1, 'AP-DD': 2, 'II-M-N': 3, 'R-K-S': 4}
    ratio_id = ratio2id[ratio]
    charge_id = charge2id[charge]

    factor_num_list = [20, 30, 30, 30, 60]
    batch_size_list = [[4, 8, 32, 64, 128], [4, 16, 32, 64, 128], [4, 8, 32, 64, 128],
                       [8, 8, 32, 64, 128], [4, 8, 32, 64, 128]]
    chain_list = [5, 5, 10, 10, 10]
    cons_list = [[0.5, 0.5, 0.25, 0.5, 0.5], [0.5, 0.25, 1, 0.25, 1], [0.5, 0.25, 0.25, 0.25, 0.25],
                 [0.25, 0.5, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.5, 0.1]]

    config.factor_num = factor_num_list[ratio_id]
    config.batch_size = batch_size_list[charge_id][ratio_id]
    config.chain = chain_list[ratio_id]
    config.cons = cons_list[charge_id][ratio_id]

    config.num_classes = len(charge.split('-'))
    config.class_list = [str(x) for x in range(config.num_classes)]
    return config
