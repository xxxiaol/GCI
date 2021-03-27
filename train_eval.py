# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
import pickle as pkl
import random

random.seed(1)


def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter, name, cons, seed, charge):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False
    for epoch in range(config.num_epochs):
        for i, (trains, labels) in enumerate(train_iter):
            if 'Att' in name:
                outputs, att = model(trains)
            else:
                outputs = model(trains)
            if name == 'BiLSTM_Att_Cons':
                model.zero_grad()
                sum1 = torch.sum((trains[2] - att) * (trains[2] - att), dim=-1)
                sum2 = torch.sum(sum1 * trains[3])
                loss_cons = sum2 / att.size()[0]
                loss = F.cross_entropy(outputs, labels) + cons * loss_cons
            else:
                model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            if total_batch % 10 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter, name, seed, charge)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()

            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

    test(config, model, test_iter, name, seed, charge)


def test(config, model, test_iter, name, seed, charge):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion, labels_all, predict_all = evaluate(config, model, test_iter, name,
                                                                                         seed, charge, test=True)

    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, name, seed, charge, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    att_all = []
    with torch.no_grad():
        for texts, labels in data_iter:
            if 'Att' in name:
                outputs, att = model(texts)
                att = att.cpu().numpy()
                att_all.append(att)
            else:
                outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion, labels_all, predict_all
    return acc, loss_total / len(data_iter)
