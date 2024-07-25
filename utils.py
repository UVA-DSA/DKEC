import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from default_sets import *
# if dataset == 'RAA':
#     from default_sets import device, p_node, reverse_group_p_dict, ungroup_p_node, group_hier, ungroup_hier, p2hier
# elif dataset == 'MIMIC3':
#     from default_sets import ICD9_DIAG
import re
import default_sets
from visualize import plot_logits_histogram, plot_f1_p_r
from collections import OrderedDict, defaultdict
import os
import json
import pandas as pd
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import wandb
cahcedStopwords = stopwords.words('english')
import torch.distributed as dist


from torch.distributed.fsdp import (
    # FullyShardedDataParallel as FSDP,
    # CPUOffload,
    MixedPrecision,
    # BackwardPrefetch,
    # ShardingStrategy,
)

# requires grad scaler in main loop
fpSixteen = MixedPrecision(
    param_dtype=torch.float16,
    # Gradient communication precision.
    reduce_dtype=torch.float16,
    # Buffer precision.
    buffer_dtype=torch.float16,
)

bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
)

bfSixteen_working = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

fp32_policy = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)



class AttrDict(dict):
    def __getattr__(self, attr):
        return self[attr]['value']
    def __setattr__(self, attr, value):
        self[attr] = value

def reduce_tensor(tensor):
    rt = tensor.clone().detach()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    return rt

def cnt_instance_per_label(df):
    label_cnt = {}

    if dataset == 'RAA':
        column_name = 'Ungrouped Protocols'
    elif 'MIMIC3' in dataset:
        column_name = 'LABELS'
    else:
        raise Exception('check dataset in default_sets.py')

    for i in range(len(df)):
        if type(df[column_name][i]) == float:
            continue
        ps = df[column_name][i].strip()
        for p in ps.split(';'):
            p = p.strip()
            label_cnt[p] = label_cnt.get(p, 0) + 1
    return label_cnt

def segProtocols(label_cnt):
    head_p = []
    mid_p = []
    tail_p = []

    head_bound = 100
    mid_bound = 10
    tail_bound = 0

    for k, v in label_cnt.items():
        if v >= head_bound:
            head_p.append(k)
        elif mid_bound < v < head_bound:
            mid_p.append(k)
        elif v <= mid_bound:
            tail_p.append(k)
    return head_p, mid_p, tail_p

def segLogits(head_p, mid_p, tail_p, logits):
    head_p_idx = []
    mid_p_idx = []
    tail_p_idx = []

    p_list = label
    for i in range(len(head_p)):
        idx = p_list.index(head_p[i])
        head_p_idx.append(idx)

    for i in range(len(mid_p)):
        idx = p_list.index(mid_p[i])
        mid_p_idx.append(idx)

    for i in range(len(tail_p)):
        idx = p_list.index(tail_p[i])
        tail_p_idx.append(idx)

    head_p_logits = np.array(logits)[:, head_p_idx]
    mid_p_logits = np.array(logits)[:, mid_p_idx]
    tail_p_logits = np.array(logits)[:, tail_p_idx]
    return head_p_logits, mid_p_logits, tail_p_logits


def genSegReport(file, head_p, mid_p, tail_p, y_pred, y_true, topk_list):
    head_report = defaultdict(list)
    mid_report = defaultdict(list)
    tail_report = defaultdict(list)

    head_y_pred, mid_y_pred, tail_y_pred = segLogits(head_p, mid_p, tail_p, y_pred)
    head_y_true, mid_y_true, tail_y_true = segLogits(head_p, mid_p, tail_p, y_true)

    p = []
    r = []
    rprecision = []
    ndcg = []
    for topk in topk_list:
        head_p_topk = get_precision_at_k(head_y_true, head_y_pred, k=topk)
        head_r_topk = get_recall_at_k(head_y_true, head_y_pred, k=topk)
        head_ndcg_topk = get_ndcg_at_k(head_y_true, head_y_pred, k=topk)
        head_rprecision_topk = get_r_precision_at_k(head_y_true, head_y_pred, k=topk)

        mid_p_topk = get_precision_at_k(mid_y_true, mid_y_pred, k=topk)
        mid_r_topk = get_recall_at_k(mid_y_true, mid_y_pred, k=topk)
        mid_ndcg_topk = get_ndcg_at_k(mid_y_true, mid_y_pred, k=topk)
        mid_rprecision_topk = get_r_precision_at_k(mid_y_true, mid_y_pred, k=topk)

        tail_p_topk = get_precision_at_k(tail_y_true, tail_y_pred, k=topk)
        tail_r_topk = get_recall_at_k(tail_y_true, tail_y_pred, k=topk)
        tail_ndcg_topk = get_ndcg_at_k(tail_y_true, tail_y_pred, k=topk)
        tail_rprecision_topk = get_r_precision_at_k(tail_y_true, tail_y_pred, k=topk)

        p.append([head_p_topk, mid_p_topk, tail_p_topk])
        r.append([head_r_topk, mid_r_topk, tail_r_topk])
        ndcg.append([head_ndcg_topk, mid_ndcg_topk, tail_ndcg_topk])
        rprecision.append([head_rprecision_topk, mid_rprecision_topk, tail_rprecision_topk])

    filters = ['micro avg', 'macro avg', 'weighted avg', 'samples avg',
               "P@1", "R@1", "R-Precision@1", "nDCG@1",
               "P@3", "R@3", "R-Precision@3", "nDCG@3",
               "P@5", "R@5", "R-Precision@5", "nDCG@5",
               "P@6", "R@6", "R-Precision@6", "nDCG@6",
               "P@8", "R@8", "R-Precision@8", "nDCG@8",
               "P@10", "R@10", "R-Precision@10", "nDCG@10",
               "P@12", "R@12", "R-Precision@12", "nDCG@12"]
    for k, v in file.items():
        if k in filters:
            continue
        if k in head_p:
            head_report['precision'].append(v['precision'])
            head_report['recall'].append(v['recall'])
            head_report['f1-score'].append(v['f1-score'])
            head_report['TP'].append(v['TP'])
            head_report['FP'].append(v['FP'])
            head_report['FN'].append(v['FN'])
            head_report['TN'].append(v['TN'])

        elif k in mid_p:
            mid_report['precision'].append(v['precision'])
            mid_report['recall'].append(v['recall'])
            mid_report['f1-score'].append(v['f1-score'])
            mid_report['TP'].append(v['TP'])
            mid_report['FP'].append(v['FP'])
            mid_report['FN'].append(v['FN'])
            mid_report['TN'].append(v['TN'])


        elif k in tail_p:
            tail_report['precision'].append(v['precision'])
            tail_report['recall'].append(v['recall'])
            tail_report['f1-score'].append(v['f1-score'])
            tail_report['TP'].append(v['TP'])
            tail_report['FP'].append(v['FP'])
            tail_report['FN'].append(v['FN'])
            tail_report['TN'].append(v['TN'])

        else:
            print(k)
            raise Exception('wrong!')

    if (np.sum(head_report['TP']) + np.sum(head_report['FP'])) == 0:
        head_micro_p = 0.0
    else:
        head_micro_p = np.sum(head_report['TP']) / (np.sum(head_report['TP']) + np.sum(head_report['FP']))
    if (np.sum(head_report['TP']) + np.sum(head_report['FN'])) == 0:
        head_micro_r = 0.0
    else:
        head_micro_r = np.sum(head_report['TP']) / (np.sum(head_report['TP']) + np.sum(head_report['FN']))
    if head_micro_p == 0 and head_micro_r == 0:
        head_micro_f1 = 0.0
    else:
        head_micro_f1 = 2 * head_micro_p * head_micro_r / (head_micro_p + head_micro_r)


    if (np.sum(mid_report['TP']) + np.sum(mid_report['FP'])) == 0:
        mid_micro_p = 0.0
    else:
        mid_micro_p = np.sum(mid_report['TP']) / (np.sum(mid_report['TP']) + np.sum(mid_report['FP']))
    if (np.sum(mid_report['TP']) + np.sum(mid_report['FN'])) == 0:
        mid_micro_r = 0.0
    else:
        mid_micro_r = np.sum(mid_report['TP']) / (np.sum(mid_report['TP']) + np.sum(mid_report['FN']))
    if mid_micro_p == 0 and mid_micro_r == 0:
        mid_micro_f1 = 0.0
    else:
        mid_micro_f1 = 2 * mid_micro_p * mid_micro_r / (mid_micro_p + mid_micro_r)

    if (np.sum(tail_report['TP']) + np.sum(tail_report['FP'])) == 0:
        tail_micro_p = 0.0
    else:
        tail_micro_p = np.sum(tail_report['TP']) / (np.sum(tail_report['TP']) + np.sum(tail_report['FP']))

    if (np.sum(tail_report['TP']) + np.sum(tail_report['FN'])) == 0:
        tail_micro_r = 0.0
    else:
        tail_micro_r = np.sum(tail_report['TP']) / (np.sum(tail_report['TP']) + np.sum(tail_report['FN']))

    if tail_micro_p == 0 and tail_micro_r == 0:
        tail_micro_f1 = 0.0
    else:
        tail_micro_f1 = 2 * tail_micro_p * tail_micro_r / (tail_micro_p + tail_micro_r)


    file['head protocols'] = {
        'micro precision': head_micro_p,
        'micro recall': head_micro_r,
        'micro f1-score': head_micro_f1,
        'macro precision': np.mean(head_report['precision']),
        'macro recall': np.mean(head_report['recall']),
        'macro f1-score': np.mean(head_report['f1-score']),
        'P@1': p[0][0],
        'R@1': r[0][0],
        'R-Precision@1': rprecision[0][0],
        'nDCG@1': ndcg[0][0],

        'P@6': p[1][0],
        'R@6': r[1][0],
        'R-Precision@6': rprecision[1][0],
        'nDCG@6': ndcg[1][0],

        'P@8': p[2][0],
        'R@8': r[2][0],
        'R-Precision@8': rprecision[2][0],
        'nDCG@8': ndcg[2][0],

        'P@12': p[3][0],
        'R@12': r[3][0],
        'R-Precision@12': rprecision[3][0],
        'nDCG@12': ndcg[3][0]
    }

    file['middle protocols'] = {
        'micro precision': mid_micro_p,
        'micro recall': mid_micro_r,
        'micro f1-score': mid_micro_f1,
        'macro precision': np.mean(mid_report['precision']),
        'macro recall': np.mean(mid_report['recall']),
        'macro f1-score': np.mean(mid_report['f1-score']),
        'P@1': p[0][1],
        'R@1': r[0][1],
        'R-Precision@1': rprecision[0][1],
        'nDCG@1': ndcg[0][1],

        'P@6': p[1][1],
        'R@6': r[1][1],
        'R-Precision@6': rprecision[1][1],
        'nDCG@6': ndcg[1][1],

        'P@8': p[2][1],
        'R@8': r[2][1],
        'R-Precision@8': rprecision[2][1],
        'nDCG@8': ndcg[2][1],

        'P@12': p[3][1],
        'R@12': r[3][1],
        'R-Precision@12': rprecision[3][1],
        'nDCG@12': ndcg[3][1]
    }
    file['tail protocols'] = {
        'micro precision': tail_micro_p,
        'micro recall': tail_micro_r,
        'micro f1-score': tail_micro_f1,
        'macro precision': np.mean(tail_report['precision']),
        'macro recall': np.mean(tail_report['recall']),
        'macro f1-score': np.mean(tail_report['f1-score']),
        'P@1': p[0][2],
        'R@1': r[0][2],
        'R-Precision@1': rprecision[0][2],
        'nDCG@1': ndcg[0][2],

        'P@6': p[1][2],
        'R@6': r[1][2],
        'R-Precision@6': rprecision[1][2],
        'nDCG@6': ndcg[1][2],

        'P@8': p[2][2],
        'R@8': r[2][2],
        'R-Precision@8': rprecision[2][2],
        'nDCG@8': ndcg[2][2],

        'P@12': p[3][2],
        'R@12': r[3][2],
        'R-Precision@12': rprecision[3][2],
        'nDCG@12': ndcg[3][2]
    }
    return file


def coarseMap(output):
    '''
    :param output: b * ungrouped_p_node
    :return:
    '''
    b = output.shape[0]
    out_ohe = np.array([np.where(t.cpu().detach().numpy() > 0.5, 1, 0) for t in output])
    coordinates = list(zip(*np.where(out_ohe == 1)))
    coarse_ohe = torch.zeros((b, len(ungroup_hier))).to(device)
    for c in coordinates:
        cur_b = c[0]
        coarse_l = label2hier[label[c[1]]]
        cur_col = ungroup_hier.index(coarse_l)
        coarse_ohe[cur_b][cur_col] = 1
    return coarse_ohe


def res_merger(save_root, k):
    report_overall = defaultdict(list)
    report_head_mid_tail = {'head protocols': defaultdict(list),
                            'middle protocols': defaultdict(list),
                            'tail protocols': defaultdict(list)}
    # filters = ['support', 'TP', 'FP', 'FN', 'TN', 'train cnt', 'val cnt', 'test cnt']

    topk = 1 if dataset=='RAA' else 6
    thresh_metrics = ['micro precision', 'micro recall', 'micro f1-score',
                      'macro precision', 'macro recall', 'macro f1-score']
    ranking_metrics = ['P@{}'.format(topk), 'R@{}'.format(topk), 'nDCG@{}'.format(topk), 'R-Precision@{}'.format(topk)]

    for i in range(k):
        save_result_root = os.path.join(save_root, 'ungroup_test_ep{}.json'.format(i))
        with open(save_result_root, 'r') as f:
            cur_report = json.load(f)
        report_overall['micro precision'].append(cur_report['micro avg']['precision'])
        report_overall['micro recall'].append(cur_report['micro avg']['recall'])
        report_overall['micro f1'].append(cur_report['micro avg']['f1-score'])
        report_overall['macro precision'].append(cur_report['macro avg']['precision'])
        report_overall['macro recall'].append(cur_report['macro avg']['recall'])
        report_overall['macro f1'].append(cur_report['macro avg']['f1-score'])

        for rm in ranking_metrics:
            report_overall[rm].append(cur_report[rm])

        for distribution in ['head protocols', 'middle protocols', 'tail protocols']:
            for tm in thresh_metrics:
                report_head_mid_tail[distribution][tm].append(cur_report[distribution][tm])
            for rm in ranking_metrics:
                report_head_mid_tail[distribution][rm].append(cur_report[distribution][rm])

    report = {'head protocols': {},
              'middle protocols': {},
              'tail protocols': {}
              }
    for k, v in report_overall.items():
        report[k] = [np.mean(v), np.std(v)]

    for dis, dis_dict in report_head_mid_tail.items():
        for k, v in dis_dict.items():
           report[dis][k] = [np.mean(v), np.std(v)]

    with open(os.path.join(save_root, 'avg_report.json'), 'w') as f:
        json.dump(report, f, indent=4)




def sharpen(probabilities, T):
    if probabilities.ndim == 1:
        tempered = torch.pow(probabilities, 1 / T)
        tempered = (
            tempered
            / (torch.pow((1 - probabilities), 1 / T) + tempered)
        )
    else:
        tempered = torch.pow(probabilities, 1 / T)
        tempered = tempered / tempered.sum(dim=-1, keepdim=True)
    return tempered.cpu().numpy()


def convertListToStr(l):
    s = ""
    if len(l) == 0: return s
    for each in l:
        s += str(each) + ';'
    return s[:-1]

def checkOnehot(X):
    for x in X:
        if x != 0 and x != 1:
            return False
    return True

def sortby(p2tfidf):
    for p, values in p2tfidf.items():
        od = OrderedDict(sorted(values.items(), key=lambda x:x[1], reverse=True))
        p2tfidf[p] = od
    return p2tfidf

def ungroup(age, protocols):
    ungroup_protocols = []
    for p in protocols.split(';'):
        p = p.strip().lower()
        if p in reverse_group_label_dict:
            if int(age) <= 21:
                ungroup_protocols.append(reverse_group_label_dict[p][1])
            else:
                ungroup_protocols.append(reverse_group_label_dict[p][0])
        else:
            ungroup_protocols.append(p)
    return convertListToStr(ungroup_protocols)

def p2onehot(ps, ref_list):
    '''
    :param ps: list of protocols, e.g.: ['medical - altered mental status (protocol 3 - 15)']
    :param ref_list: p_node, ungroup_p_node
    :return: onehot encoding
    '''
    ohe = [0] * len(ref_list)
    for p in ps:
        idx = ref_list.index(p)
        ohe[idx] = 1
    return ohe


def onehot2p(onehot):
    pred = []
    for i in range(len(onehot)):
        if onehot[i] == 1:
            p_name = group_label[i]
            pred.append(p_name)
    return convertListToStr(pred)

def removePunctuation(sentence):
    sentence = re.sub(r'[?|!|\'|"|;|:|#|&|-]', r' ', sentence)
    sentence = re.sub(r'[.|,|)|(|\|/|_|~|<|>]', r' ', sentence)
    sentence = re.sub(r"[\([{})\]]", r' ', sentence)
    sentence = re.sub(r"[*]", r' ', sentence)
    sentence = re.sub(r"[%]", r' percentage', sentence)
    sentence = sentence.strip()
    sentence = sentence.replace("\n", " ")
    return sentence

def preprocess(text):
    text_ = removePunctuation(text.lower())
    text_tokens = word_tokenize(text_)
    # text_tokens_stem = [self.stemmer.stem(w) for w in text_tokens]
    tokens_without_sw = [word for word in text_tokens if not word in cahcedStopwords]
    new_text = ' '.join(tokens_without_sw)
    return new_text

def text_remove_double_space(text):
    text = text.lower()
    res = ''
    for word in text.split():
        res = res + word + ' '
    return res.strip()

'''
The following codes from https://github.com/MemoriesJ/KAMG/blob/6618c6633bbe40de7447d5ae7338784b5233aa6a/NeuralNLP-NeuralClassifier-KAMG/evaluate/classification_evaluate.py
'''

def ranking_precision_score(y_true, y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        return ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    return float(n_relevant) / k

def get_precision_at_k(y_true, y_score, k=10):
    """Mean precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    mean precision @k : float
    """

    p_ks = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            p_ks.append(ranking_precision_score(y_t, y_s, k=k))
    return np.mean(p_ks)

def ranking_recall_score(y_true, y_score, k=10):
    # https://ils.unc.edu/courses/2013_spring/inls509_001/lectures/10-EvaluationMetrics.pdf
    """Recall at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        return ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    return float(n_relevant) / n_pos

def get_recall_at_k(y_true, y_score, k=10):
    """Mean recall at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    mean recall @k : float
    """

    r_ks = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            r_ks.append(ranking_recall_score(y_t, y_s, k=k))

    return np.mean(r_ks)

def ranking_rprecision_score(y_true, y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        return ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
    return float(n_relevant) / min(k, n_pos)

def get_r_precision_at_k(y_true, y_score, k=10):
    """Mean precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    mean precision @k : float
    """

    p_ks = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            p_ks.append(ranking_rprecision_score(y_t, y_s, k=k))

    return np.mean(p_ks)

def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best

def get_ndcg_at_k(y_true, y_predict_score, k, gains="exponential"):

    """Normalized discounted cumulative gain (NDCG) at rank k
        Parameters
        ----------
        y_true : array-like, shape = [n_samples]
            Ground truth (true relevance labels).
        y_predict_score : array-like, shape = [n_samples]
            Predicted scores.
        k : int
            Rank.
        gains : str
            Whether gains should be "exponential" (default) or "linear".
        Returns
        -------
        Mean NDCG @k : float
        """

    ndcg_s = []
    for y_t, y_s in zip(y_true, y_predict_score):
        if np.sum(y_t == 1):
            ndcg_s.append(ndcg_score(y_t, y_s, k=k, gains=gains))

    return np.mean(ndcg_s)




if __name__ =='__main__':
    import yaml
    date = '2023-03-17-23_10_35'
    print('reading configuration file...')
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    # config = yaml.load(open(args.config, 'r'), Loader=loader)
    with open('./config.yaml', 'r') as f:
        config = yaml.load(f, Loader=loader)
        config_ = AttrDict(config['parameters'])
    res_merger(config_, date, k=3)


