import pickle as pkl
import numpy as np
import re
import argparse
import queue
import pandas as pd
import random

import networkx as nx
from networkx.drawing.nx_pydot import write_dot
from networkx.drawing.nx_pydot import to_pydot

import os
import pydot

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import dowhy
from dowhy import CausalModel

parser = argparse.ArgumentParser()
parser.add_argument('--charge', default='II-M-N', type=str,
                    help='Name of the dataset')
parser.add_argument('--ratio', default=0.1, type=float,
                    help='Split ratio of training and testing')
parser.add_argument('--seed', default=1, type=int, help='Random seed')
parser.add_argument('--restart', default=False, type=bool,
                    help='Continue from dumped data')
parser.add_argument('--show_chains', default=False,
                    type=bool, help='Only show chains')
parser.add_argument('--sensitivity', default=False,
                    type=bool, help='Only check sensitivity')
parser.add_argument('--fix_graph', default=False,
                    type=bool, help='Fix graph')
parser.add_argument('--data_augment', default=False,
                    type=bool, help='Data augmentation')
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)
addr = 'data/'


def handle_args(args):
    if args.charge == 'II-M-N':
        accu_select = ['故意伤害', '故意杀人', '过失致人死亡']
    elif args.charge == 'R-K-S':
        accu_select = ['抢劫', '绑架', '抢夺']
    elif args.charge == 'F-E':
        accu_select = ['诈骗', '敲诈勒索']
    elif args.charge == 'AP-DD':
        accu_select = ['滥用职权', '玩忽职守']
    elif args.charge == 'E-MPF':
        accu_select = ['贪污', '挪用公款']

    if args.ratio <= 0.01:
        num_select1 = 12
        num_select2 = 3
        num_clusters = 20
    elif args.ratio >= 0.3:
        num_select1 = 30
        num_select2 = 10
        num_clusters = 60
    else:
        num_select1 = 20
        num_select2 = 5
        num_clusters = 30

    if args.ratio == 0.1:
        ratio_str = '19'
    elif args.ratio == 0.01:
        ratio_str = '199'
    elif args.ratio == 0.05:
        ratio_str = '119'
    elif args.ratio == 0.3:
        ratio_str = '37'
    elif args.ratio == 0.5:
        ratio_str = '55'

    suffix = '_' + ratio_str + '_' + args.charge + '_' + str(args.seed)

    return accu_select, num_select1, num_select2, num_clusters, ratio_str, suffix


def load_data(addr, args):
    print('load data...')
    with open(addr + 'preprocessed_data.pkl', 'rb') as f:
        data = pkl.load(f)
    fact_clean = data['fact_clean']
    accu_clean = data['accu_clean']
    fact_original = data['fact_original']

    random_idx = np.arange(len(accu_clean))
    np.random.shuffle(random_idx)
    fact_clean = np.array(fact_clean)[random_idx]
    accu_clean = np.array(accu_clean)[random_idx]
    fact_original_new = []
    for i in random_idx:
        fact_original_new.append(fact_original[i])
    fact_original = fact_original_new

    train_split = int(len(fact_clean) * args.ratio)

    with open(addr + 'used_wv.pkl', 'rb') as f:
        wv = pkl.load(f)

    return fact_clean, accu_clean, fact_original, random_idx, train_split, wv


def extract_keywords(addr, fact_original, accu_clean, train_split, num_select1=20, num_select2=5):
    import pke
    import spacy
    from yake_modified import YAKE

    def load_stop_words(stop_word_file):
        stop_words = []
        for line in open(stop_word_file, 'r', encoding='gbk'):
            if line.strip()[0:1] != "#":
                for word in line.split():  # in case more than one per line
                    stop_words.append(word)
        return stop_words

    def split_sentences(text):
        sentences = []
        sentences_last = []
        sentence_delimiters = re.compile(
            u'[。，；！？：.!?,;:\t\\\\"\\(\\)\\\'\u2019\u2013]|\\s\\-\\s')
        for t in text:
            sen = sentence_delimiters.split(t)
            sentences.extend(sen[3:7])
            sentences_last.extend(sen[7:])
        return sentences, sentences_last

    def count_df(fact_accu, stop_words):
        def count_df_accu(text, stop_words):
            nlp = spacy.load('zh_core_web_sm')
            extractor = YAKE()
            extractor.load_document("".join(text), language='zh', spacy_model=nlp)
            extractor.candidate_selection(n=1, stoplist=stop_words)
            return extractor.candidate_dict()

        from collections import Counter

        df = Counter()
        df_last = Counter()
        for i in fact_accu:
            sentences, sentences_last = split_sentences(i)
            df = df + count_df_accu(sentences, stop_words)
            df_last = df_last + count_df_accu(sentences_last, stop_words)

        return df, df_last

    def extract_yake(text, stop_words, num_select, df, N):
        nlp = spacy.load('zh_core_web_sm')
        extractor = YAKE()
        nlp.max_length = 4000000
        extractor.load_document("".join(text), language='zh', spacy_model=nlp)
        extractor.candidate_selection(n=1, stoplist=stop_words)
        extractor.candidate_weighting(df, N)
        return extractor.get_n_best(n=num_select)

    print('extract keyword with yake...')
    fact_train = fact_original[:train_split]
    accu_train = accu_clean[:train_split]
    accu_dict = {}
    fact_accu = []
    id2accu = {}
    for i in range(len(accu_train)):
        for j in accu_train[i]:
            if not j in accu_dict:
                id2accu[len(accu_dict)] = j
                accu_dict[j] = len(accu_dict)
                fact_accu.append([fact_train[i]])
            else:
                fact_accu[accu_dict[j]].append(fact_train[i])

    stop_word_file = addr + "ZHstopwords.txt"
    stop_words = load_stop_words(stop_word_file)

    df, df_last = count_df(fact_accu, stop_words)

    N = len(accu_dict)
    accu_keyword = []
    for i in accu_select:
        sentences, sentences_last = split_sentences(fact_accu[accu_dict[i]])
        cur_keyword = extract_yake(sentences, stop_words, num_select1, df, N)
        cur_keyword_last = extract_yake(sentences_last, stop_words, num_select2, df_last, N)
        cur_keyword.extend(cur_keyword_last)
        cur_keyword = [x for x, y in cur_keyword]
        accu_keyword.append(cur_keyword)

    original_combine_key = [x for k in accu_keyword for x in k]

    return original_combine_key


def cluster_keywords(wv, original_combine_key, num_clusters=30):
    print('cluster keyword...')
    km_cluster = KMeans(n_clusters=num_clusters, max_iter=300, n_init=40,
                        init='k-means++', n_jobs=1)
    original_combine_key_new = []
    key_wv = []
    for i in original_combine_key:
        if i in wv:
            key_wv.append(wv[i])
            original_combine_key_new.append(i)
    result = km_cluster.fit_predict(key_wv)

    clustered = {}
    for i in range(len(original_combine_key_new)):
        if not result[i] in clustered:
            clustered[result[i]] = [original_combine_key_new[i]]
        else:
            clustered[result[i]].append(original_combine_key_new[i])

    combine_key = [list(set(i)) for i in clustered.values()]
    return combine_key


def find_factors(fact_clean, accu_clean, accu_select, train_split, combine_key):
    print('find factors...')
    y = np.zeros((len(fact_clean), len(accu_select)), dtype=np.int8)
    cnt_pos = [0, 0]
    for i in range(len(accu_clean)):
        for j in range(len(accu_select)):
            if accu_select[j] in accu_clean[i]:
                y[i][j] = 1
        if np.sum(y[i]) == 1:
            if i < train_split:
                cnt_pos[0] += 1
            else:
                cnt_pos[1] += 1

    factor = [np.zeros((cnt_pos[0] * 2, len(combine_key)), dtype=np.int8),
              np.zeros((cnt_pos[1], len(combine_key)), dtype=np.int8)]
    idx = [np.zeros((cnt_pos[0] * 2), dtype=np.int64),
           np.zeros((cnt_pos[1]), dtype=np.int64)]
    word_idx = [[], []]
    word_key = [[], []]
    cnt_neg = [0, 0]
    for i in range(len(accu_clean)):
        if i % 1000 == 0:
            print(i, len(word_idx[0]), len(word_idx[1]))
        if i < train_split:
            c = 0
        else:
            c = 1
        if np.sum(y[i]) == 0:
            if c == 1 or cnt_neg[c] == cnt_pos[c]:
                continue
            else:
                cnt_neg[c] += 1
        elif np.sum(y[i]) >= 2:
            continue

        idx[c][len(word_idx[c])] = i
        cur = np.zeros((len(combine_key)), dtype=np.int8)
        word_idx_cur = []
        word_key_cur = []
        for k in range(len(combine_key)):
            f = False
            for p in combine_key[k]:
                if f:
                    break
                for j in range(len(fact_clean[i])):
                    d = np.sum(
                        np.abs(np.array(wv[fact_clean[i][j]]) - np.array(wv[p])))
                    if d < 20:
                        f = True
                        word_idx_cur.append(j)
                        word_key_cur.append(k)
                        break
            if f:
                cur[k] = 1
        factor[c][len(word_idx[c])] = cur
        word_idx[c].append(word_idx_cur)
        word_key[c].append(word_key_cur)

    return y, factor, idx, word_idx, word_key


def dump_data(combine_key, accu_select, y, factor, idx, word_idx, word_key, random_idx, fact_original, fact_clean,
              accu_clean):
    print('dump data...')
    with open(addr + 'factor_name' + suffix + '.txt', 'w') as f:
        for i in range(len(combine_key)):
            f.write('x' + str(i) + '\t')
            for j in combine_key[i]:
                f.write(j + ' ')
            f.write('\n')
        for i in range(len(accu_select)):
            f.write('y' + str(i) + '\t' + accu_select[i] + '\n')

    with open(addr + 'factor' + suffix + '.tsv', 'w') as f:
        for i in range(len(combine_key)):
            f.write('x' + str(i) + '\t')
        for i in range(len(accu_select)):
            if i < len(accu_select) - 1:
                f.write('y' + str(i) + '\t')
            else:
                f.write('y' + str(i) + '\n')

        for k in range(2):
            for i in range(len(word_idx[k])):
                for j in factor[k][i]:
                    f.write(str(j) + '\t')
                for j in range(len(accu_select)):
                    f.write(str(y[idx[k][i]][j]))
                    if j < len(accu_select) - 1:
                        f.write('\t')
                f.write('\n')

    new_train_split = len(word_idx[0])
    data_new = {
        'combine_key': combine_key,
        'word_idx': word_idx,
        'word_key': word_key,
        'idx': idx,
        'train_split': new_train_split,
        'random_idx': random_idx
    }
    with open(addr + 'data' + suffix + '.pkl', 'wb') as f:
        pkl.dump(data_new, f)

    with open(addr + 'text' + suffix + '.tsv', 'w') as f:
        for k in range(2):
            for i in range(len(word_idx[k])):
                f.write(str(idx[k][i]) + '\t'
                        + fact_original[idx[k][i]] + '\t')
                for j in accu_clean[idx[k][i]]:
                    f.write(j + ' ')
                f.write('\n')

    return new_train_split


def load_dumped_data(suffix):
    print('load dumped data...')
    with open(addr + 'data' + suffix + '.pkl', 'rb') as f:
        data = pkl.load(f)
    idx = data['idx']
    word_idx = data['word_idx']
    word_key = data['word_key']
    combine_key = data['combine_key']
    new_train_split = data['train_split']
    return idx, word_idx, word_key, combine_key, new_train_split


def time_constraint(fact_clean, idx, combine_key, suffix):
    def count_first_occur(fact_clean, idx, combine_key, suffix):
        fact_train = fact_clean[idx[0]]
        first_occur = np.zeros(
            (len(fact_train), len(combine_key)), dtype=np.int64) + 1000

        for i in range(len(fact_train)):
            if i % 1000 == 0:
                print(i)
            for k in range(len(combine_key)):
                f = False
                for j in range(len(fact_train[i])):
                    if f:
                        break
                    for p in combine_key[k]:
                        d = np.sum(
                            np.abs(np.array(wv[fact_train[i][j]]) - np.array(wv[p])))
                        if d < 20:
                            f = True
                            first_occur[i][k] = j
                            break
        np.save(addr + 'first_occur' + suffix + '.npy', first_occur)
        return first_occur

    print('time constraint...')
    first_occur_file = addr + 'first_occur' + suffix + '.npy'
    if not os.path.exists(first_occur_file):
        first_occur = count_first_occur(fact_clean, idx, combine_key, suffix)
    else:
        first_occur = np.load(first_occur_file)

    constraint = []  # forbidden
    alpha = 1.5
    for i in range(len(combine_key)):
        for j in range(i + 1, len(combine_key)):
            front = np.sum(((first_occur[:, i] < first_occur[:, j]) & (
                    first_occur[:, j] < 1000)).astype(int))
            back = np.sum(((first_occur[:, i] > first_occur[:, j]) & (
                    first_occur[:, i] < 1000)).astype(int))
            if front > alpha * back:
                constraint.append([j, i])
            if back > alpha * front:
                constraint.append([i, j])

    with open(addr + 'time_constraint' + suffix + '.pkl', 'wb') as f:
        pkl.dump(constraint, f)
    return constraint


def data_augmentation(suffix, accu_select, idx, word_idx, word_key, train_split):
    df = pd.read_csv(addr + 'factor' + suffix + '.tsv', sep='\t')
    df_train = df[:train_split]
    df_test = df[train_split:]
    new_df_train = df_train.copy()

    text = pd.read_csv(addr + 'text' + suffix + '.tsv', sep='\t')
    text_train = text[:train_split]
    text_test = text[train_split:]
    new_text_train = text_train.copy()

    new_idx = [list(idx[0]), idx[1]]
    new_word_idx = word_idx.copy()
    new_word_key = word_key.copy()

    y_cols = ['y' + str(i) for i in range(len(accu_select))]
    accu_cnt = df_train[y_cols].sum(axis=0)
    min_limit = accu_cnt.max() // 4

    for i in y_cols:
        if accu_cnt[i] < min_limit:
            idx_accu = list(df_train.loc[df_train[i] == 1].index)
            idx_chosen = np.random.choice(idx_accu, min_limit - accu_cnt[i], replace=True)
            new_df_train = new_df_train.append(df_train.iloc[idx_chosen])
            new_text_train = new_text_train.append(text_train.iloc[idx_chosen])
            new_idx[0].extend(list(idx[0][idx_chosen]))
            new_word_idx[0].extend(list(np.array(word_idx[0])[idx_chosen]))
            new_word_key[0].extend(list(np.array(word_key[0])[idx_chosen]))

    new_train_split = len(word_idx[0])
    print(train_split, new_train_split)
    new_idx[0] = np.asarray(new_idx[0])

    new_df = pd.concat([new_df_train, df_test])
    new_df.to_csv(addr + 'factor_aug' + suffix + '.tsv', sep='\t', index=0)
    new_text = pd.concat([new_text_train, text_test])
    new_text.to_csv(addr + 'text_aug' + suffix + '.tsv', sep='\t', index=0)
    return new_idx, new_word_idx, new_word_key, new_train_split


def build_causal_graph(suffix, train_split, constraint, accu_select, graph_samples_num, augment=False):
    from pycausal.pycausal import pycausal as pc
    from pycausal import prior as p
    from pycausal import search as s

    print('build causal graph...')
    if augment:
        df = pd.read_csv(addr + 'factor_aug' + suffix + '.tsv', sep='\t')
    else:
        df = pd.read_csv(addr + 'factor' + suffix + '.tsv', sep='\t')
    df_train = df[:train_split]
    pc = pc()
    pc.start_vm(java_max_heap_size='10000M')

    forbid = []
    for i in df_train.columns.values:
        for j in range(len(accu_select)):
            forbid.append(['y' + str(j), i])

    for i, j in constraint:
        xi = 'x' + str(i)
        xj = 'x' + str(j)
        forbid.append([xi, xj])
    prior = p.knowledge(forbiddirect=forbid)

    tetrad = s.tetradrunner()
    tetrad.run(algoId='gfci', dfs=df_train, testId='disc-bic-test', scoreId='bdeu-score',
               priorKnowledge=prior, dataType='discrete',
               maxDegree=3, maxPathLength=-1,
               completeRuleSetUsed=False, faithfulnessAssumed=True, verbose=False,
               numberResampling=5, resamplingEnsemble=1, addOriginalDataset=True)
    gdata = {}
    gdata['nodes'] = tetrad.getNodes()
    gdata['edges'] = tetrad.getEdges()
    gdata['edges_p'] = {}

    for i in gdata['edges']:
        print(i)
        t = i.find('[')
        cur_edge = i[t:]
        u, _, v = i[:t].strip().split(' ')
        p = [0., 0., 0.]  # u->v, v->u, no edge
        for j in cur_edge.split(';'):
            if not ':' in j:
                continue
            edge_str, edge_p = j.split(':')
            edge_p = float(edge_p)
            if '-->' in edge_str or '<--' in edge_str:
                if ('-->' in edge_str and u in edge_str.split(' ')[0]) or (
                        '<--' in edge_str and v in edge_str.split(' ')[0]):
                    direction = 0
                else:
                    direction = 1
                p[direction] += edge_p
            elif '---' in edge_str:
                p[0] += edge_p / 2
                p[1] += edge_p / 2
            elif 'o->' in edge_str or '<-o' in edge_str:
                if ('o->' in edge_str and u in edge_str.split(' ')[0]) or (
                        '<-o' in edge_str and v in edge_str.split(' ')[0]):
                    direction = 0
                else:
                    direction = 1
                p[direction] += edge_p / 2
                p[2] += edge_p / 2
            elif 'o-o' in edge_str:
                p[0] += edge_p / 3
                p[1] += edge_p / 3
                p[2] += edge_p / 3
            elif 'no edge' in edge_str or '<->' in edge_str:
                p[2] += edge_p
        p = p / np.sum(np.array(p))
        gdata['edges_p'][i] = p

    G_samples = []
    for k in range(graph_samples_num):
        G = nx.DiGraph()
        edge_list = []
        for i in gdata['nodes']:
            G.add_node(i)
        for i in gdata['edges']:
            t = i.find('[')
            u, _, v = i[:t].strip().split(' ')
            p = gdata['edges_p'][i]
            # avoid loop
            G1 = G.copy()
            G1.add_edge(u, v)
            if not nx.is_directed_acyclic_graph(G1):
                p[0] = 0
            G1 = G.copy()
            G1.add_edge(v, u)
            if not nx.is_directed_acyclic_graph(G1):
                p[1] = 0

            if np.sum(p) > 0:
                p = p / np.sum(p)
                choice = np.random.choice(a=3, p=p)
                if choice == 0:
                    G.add_edge(u, v)
                elif choice == 1:
                    G.add_edge(v, u)
        G_samples.append(G)
        with open(addr + 'graph' + suffix + '_' + str(k) + '.dot', 'w') as f:
            write_dot(G, f)

    with open(addr + 'graph_samples' + suffix + '.pkl', 'wb') as f:
        pkl.dump(G_samples, f)

    pc.stop_vm()
    return G_samples


def causal_strength(suffix, train_split, G_samples, accu_select, sensitivity=False, augment=False):
    import pgmpy
    from pgmpy.models import BayesianModel
    from pgmpy.estimators import BicScore
    print('calculate causal strength...')
    if augment:
        df = pd.read_csv(addr + 'factor_aug' + suffix + '.tsv', sep='\t')
    else:
        df = pd.read_csv(addr + 'factor' + suffix + '.tsv', sep='\t')
    for i in df.columns.values.tolist():
        df[i] = df[i].astype('bool')

    df_train = df[:train_split]
    y_cols = ['y' + str(i) for i in range(len(accu_select))]
    df_train_pos = df_train.loc[df_train[y_cols].sum(axis=1) > 0]
    df_test = df[train_split:]

    x_train = np.array(df_train_pos)
    x_test = np.array(df_test)

    model = BayesianModel([])
    bic = BicScore(data=df_train_pos)
    print(bic.score(model))

    column_dict = {}
    column_dict_rev = {}
    cnt = 0
    for i in df_train.columns:
        column_dict[i] = cnt
        column_dict_rev[cnt] = i
        cnt += 1
    outcomes = y_cols

    est_array_all = []
    G_scores = []
    for t in range(len(G_samples)):
        treatments = [[] for i in range(len(accu_select))]
        print(G_samples[t].edges())
        for u, v in G_samples[t].edges():
            if v[0] == 'y':
                treatments[int(v[1])].append(u)

        est = [{} for i in range(len(accu_select))]
        for k in range(len(outcomes)):
            for i in treatments[k]:
                model = CausalModel(
                    data=df_train_pos,
                    treatment=i,
                    outcome=outcomes[k],
                    graph=addr + 'graph' + suffix + '_' + str(t) + '.dot',
                    verbose=0
                )
                identified_estimand = model.identify_effect(
                    proceed_when_unidentifiable=True)
                confounders = identified_estimand.backdoor_variables
                if len(confounders) == 0:
                    causal_estimate = model.estimate_effect(identified_estimand,
                                                            method_name="backdoor.linear_regression",
                                                            test_significance=True)
                else:
                    causal_estimate = model.estimate_effect(identified_estimand,
                                                            method_name="backdoor.propensity_score_matching")
                est[k][i] = causal_estimate.value
                if sensitivity:
                    res_random = model.refute_estimate(identified_estimand, causal_estimate,
                                                       method_name="random_common_cause")
                    print(res_random)  # not change
                    res_placebo = model.refute_estimate(identified_estimand, causal_estimate,
                                                        method_name="placebo_treatment_refuter", placebo_type="permute")
                    print(res_placebo)  # 0
                    res_subset = model.refute_estimate(identified_estimand, causal_estimate,
                                                       method_name="data_subset_refuter", subset_fraction=0.8)
                    print(res_subset)  # not very significantly

        est_array = np.zeros((len(outcomes), len(column_dict) - len(outcomes)), dtype=np.float)
        for k in range(len(outcomes)):
            for i in est[k]:
                est_array[k, column_dict[i]] = est[k][i]
        est_array_all.append(est_array)

        # calculate BIC score
        model = BayesianModel(G_samples[t].edges())
        bic = BicScore(data=df_train_pos)
        G_scores.append(bic.score(model))

    est_array_all = np.array(est_array_all)

    G_scores = np.array(G_scores)
    print(G_scores)
    G_scores = np.abs(G_scores)
    G_scores = G_scores - np.min(G_scores)
    G_scores = (G_scores / np.sum(G_scores) + 1 / len(G_scores)) / 2
    print(G_scores)
    est_array_all = np.sum(np.multiply(np.reshape(G_scores, (-1, 1, 1)), est_array_all), 0)

    est_all = [{} for i in range(len(outcomes))]
    for k in range(len(outcomes)):
        for i in range(len(est_array_all[k])):
            if est_array_all[k][i] > 0:
                est_all[k][column_dict_rev[i]] = est_array_all[k][i]

    score_train = np.dot(x_train[:, 0:-len(outcomes)], est_array_all.T)

    y_train = x_train[:, -len(outcomes):]
    clf = RandomForestClassifier()
    _ = clf.fit(score_train, y_train)

    y_pred = clf.predict(score_train)
    print(classification_report(
        np.argmax(y_train, -1), np.argmax(y_pred, -1), labels=range(len(outcomes)), digits=4))

    score_test = np.dot(x_test[:, 0:-len(outcomes)], est_array_all.T)
    y_pred = clf.predict(score_test)
    y_true = x_test[:, -len(outcomes):]

    print(classification_report(np.argmax(y_true, -1), np.argmax(y_pred, -1), labels=range(len(outcomes)), digits=4))

    with open(addr + 'est' + suffix + '.pkl', 'wb') as f:
        pkl.dump([est_all, G_scores], f)
    return est_all, G_scores


cur_chains = []
max_length = 0


def chain(word_idx, word_key, fact_clean, idx, G_samples, G_scores, num_clusters, accu_select):
    print('find chains...')
    chains_all = []
    for s in range(len(G_samples)):
        G = G_samples[s]
        ancestor_key = {}
        for u, v in G.edges():
            if u[0] == 'x':
                u_int = int(u[1:])
            else:
                u_int = num_clusters + int(u[1:])
            if v[0] == 'x':
                v_int = int(v[1:])
            else:
                v_int = num_clusters + int(v[1:])
            if v_int in ancestor_key:
                ancestor_key[v_int].append(u_int)
            else:
                ancestor_key[v_int] = [u_int]

        global cur_chains, max_length
        max_num = 0
        chains = [[], []]
        for t in range(2):
            for i in range(len(word_idx[t])):
                key2idx = {}
                idx2key = {}
                for j in range(len(word_key[t][i])):
                    idx2key[word_idx[t][i][j]] = word_key[t][i][j]
                    if not word_key[t][i][j] in key2idx:
                        key2idx[word_key[t][i][j]] = [word_idx[t][i][j]]
                    else:
                        key2idx[word_key[t][i][j]].append(word_idx[t][i][j])
                q = queue.Queue()
                ancestor = {}
                tail = []
                used = set()
                for j in range(len(accu_select)):
                    if num_clusters + j in ancestor_key:
                        for k in ancestor_key[num_clusters + j]:
                            if k in key2idx:
                                for p in key2idx[k]:
                                    if not p in used:
                                        tail.append(p)
                                        used.add(p)
                                        q.put(p)
                while not q.empty():
                    j = q.get()
                    if idx2key[j] in ancestor_key:
                        for k in ancestor_key[idx2key[j]]:
                            if k in key2idx:
                                for p in key2idx[k]:
                                    if p < j:
                                        if not j in ancestor:
                                            ancestor[j] = []
                                        ancestor[j].append(p)
                                        if not p in used:
                                            used.add(p)
                                            q.put(p)
                cur_chains = []

                def find_chain(cur_node, tmp_chain):
                    global cur_chains
                    global max_length
                    if not cur_node in ancestor:
                        cur_chains.append((tmp_chain, G_scores[s]))
                        if len(tmp_chain) > max_length:
                            max_length = len(tmp_chain)
                    else:
                        for k in ancestor[cur_node]:
                            find_chain(k, [k] + tmp_chain)

                for j in tail:
                    find_chain(j, [j])
                if len(cur_chains) > max_num:
                    max_num = len(cur_chains)
                if s == 0:
                    if len(cur_chains) > 0 and i < 20:
                        for j, score in cur_chains:
                            for k in j:
                                print(fact_clean[idx[t][i]][k], end=' ')
                            print()
                        print()
                chains[t].append(cur_chains)
        chains_all.append(chains)
    # chains_all graph_num*2*n*chain_num -> 2*n*(graph_num*chain_num)
    chains_all_reshape = [[], []]
    for t in range(2):
        for i in range(len(word_idx[t])):
            cur_chains = [x for j in range(len(G_samples)) for x in chains_all[j][t][i]]
            chains_all_reshape[t].append(cur_chains)

    print(max_length, max_num)
    with open(addr + 'chains' + suffix + '.pkl', 'wb') as f:
        pkl.dump(chains_all_reshape, f)
    return chains_all_reshape


def prepare_nn_data(fact_clean, fact_original, idx, word_idx, word_key, train_split, est, chains, suffix, accu_select,
                    augment=False):
    print('prepare nn data...')
    if augment:
        df = pd.read_csv(addr + 'factor_aug' + suffix + '.tsv', sep='\t')
    else:
        df = pd.read_csv(addr + 'factor' + suffix + '.tsv', sep='\t')
    df_train = df[:train_split]
    y_cols = ['y' + str(i) for i in range(len(accu_select))]
    df_train_pos = df_train.loc[df_train[y_cols].sum(axis=1) > 0]
    idx_pos = np.array(
        df_train.loc[df_train[y_cols].sum(axis=1) > 0].index)
    print(idx_pos.shape, df_train_pos.shape)
    labels_train = np.array(df_train_pos[y_cols])
    df_test = df[train_split:]
    labels_test = np.array(df_test[y_cols])

    text_train = fact_clean[idx[0]]
    text_train_pos = text_train[idx_pos]
    text_test = fact_clean[idx[1]]

    word_idx_train = np.array(word_idx[0])[idx_pos]
    word_idx_test = np.array(word_idx[1])
    word_key_train = np.array(word_key[0])[idx_pos]
    word_key_test = np.array(word_key[1])

    pure_text_train = []
    for i in idx[0]:
        pure_text_train.append(fact_original[i])
    pure_text_train = np.asarray(pure_text_train)[idx_pos]
    pure_text_test = []
    for i in idx[1]:
        pure_text_test.append(fact_original[i])
    pure_text_test = np.asarray(pure_text_test)

    chains_train = np.array(chains[0])[idx_pos]
    chains_test = np.array(chains[1])

    strength = [[] for i in range(len(accu_select))]
    factor_list = [[] for i in range(len(accu_select))]
    for i in range(len(est)):
        for j in est[i]:
            factor_list[i].append(j)
            strength[i].append(est[i][j])

    data = {
        'text_train': text_train_pos,
        'text_test': text_test,
        'labels_train': labels_train,
        'labels_test': labels_test,
        'word_idx_train': word_idx_train,
        'word_idx_test': word_idx_test,
        'word_key_train': word_key_train,
        'word_key_test': word_key_test,
        'chains_train': chains_train,
        'chains_test': chains_test,
        'original_text_train': pure_text_train,
        'original_text_test': pure_text_test,
        'strength': strength,
        'factor_list': factor_list
    }

    with open(addr + 'nn_data' + suffix + '.pkl', 'wb') as f:
        pkl.dump(data, f)


accu_select, num_select1, num_select2, num_clusters, ratio_str, suffix = handle_args(
    args)
graph_samples_num = 5
fact_clean, accu_clean, fact_original, random_idx, train_split, wv = load_data(
    addr, args)
if not args.restart and not args.sensitivity:
    original_combine_key = \
        extract_keywords(addr, fact_original, accu_clean,
                         train_split, num_select1, num_select2)
    combine_key = cluster_keywords(wv, original_combine_key, num_clusters)
    y, factor, idx, word_idx, word_key = \
        find_factors(fact_clean, accu_clean, accu_select,
                     train_split, combine_key)
    new_train_split = \
        dump_data(combine_key, accu_select, y, factor, idx, word_idx, word_key, random_idx,
                  fact_original, fact_clean, accu_clean)
else:
    idx, word_idx, word_key, combine_key, new_train_split = load_dumped_data(
        suffix)
constraint = time_constraint(fact_clean, idx, combine_key, suffix)
if args.data_augment:
    idx, word_idx, word_key, new_train_split = data_augmentation(suffix, accu_select, idx, word_idx, word_key,
                                                                 new_train_split)
if not args.sensitivity:
    if not args.fix_graph:
        G_samples = build_causal_graph(suffix, new_train_split, constraint, accu_select, graph_samples_num,
                                       augment=args.data_augment)
    else:
        with open(addr + 'graph_samples' + suffix + '.pkl', 'rb') as f:
            G_samples = pkl.load(f)
    est, G_scores = causal_strength(suffix, new_train_split, G_samples, accu_select, augment=args.data_augment)
    chains = chain(word_idx, word_key, fact_clean, idx, G_samples, G_scores, num_clusters, accu_select)
    prepare_nn_data(fact_clean, fact_original, idx, word_idx,
                    word_key, new_train_split, est, chains, suffix, accu_select, augment=args.data_augment)
else:
    with open(addr + 'graph_samples' + suffix + '.pkl', 'rb') as f:
        G_samples = pkl.load(f)
    est, G_scores = causal_strength(suffix, new_train_split, G_samples, accu_select, augment=args.data_augment,
                                    sensitivity=True)
