import json

from default_sets import *
# if dataset == 'RAA':
#     from default_sets import p_node, ungroup_p_node, reverse_group_p_dict, group_hier, ungroup_hier, \
#         reverse_group_hier_dict, group_p_dict
# elif dataset == 'MIMIC3':
#     from default_sets import ICD9_DIAG, ICD9_DIAG_GROUP, group_ICD9_dict, reverse_group_ICD9_dict
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix
from utils import segProtocols, genSegReport, get_precision_at_k, get_recall_at_k, get_ndcg_at_k, get_r_precision_at_k
import pandas as pd

# val_indices, test_indices has again been shuffled in build_dataloader
def convert_label(labels, ages, logits=None):
    '''
    labels: [[ohe], [ohe], [ohe]]
    logits: [[logit], [logit], [logit]]
    ages: [tensor, tensor, tensor]
    convert clustered labels to exact labels
    p_node contains protocol names (the sequence order the same with labels).
    '''

    encodings = []
    encoding_logits = []
    # if metric == 'age':

    base_node = group_label
    refer_node = label
    convert_dict = reverse_group_label_dict
    map_dict = group_label_dict

    # ages = [age.numpy() for age in ages]
    for i, l in enumerate(labels):
        label_name = [base_node[j] for j in range(len(l)) if l[j] == 1]
        label_name_indices = [j for j in range(len(l)) if l[j] == 1]
        # convert the name 2 according to another sequence
        encoding = [0] * len(refer_node)
        encoding_logit = [0] * len(refer_node)
        for k, n in enumerate(label_name):
            if n in refer_node:
                idx = refer_node.index(n)
            # use age to determine
            else:
                age = int(ages[i])
                ### if it's pediatric ###
                if age < 18:
                    p_name = convert_dict[n][1]
                    if p_name in refer_node:
                        idx = refer_node.index(p_name)
                    else:
                        idx = refer_node.index(convert_dict[n][0])
                ### if it's adult ###
                else:
                    p_name = convert_dict[n][0]
                    if p_name in refer_node:
                        idx = refer_node.index(p_name)
                    else:
                        idx = refer_node.index(convert_dict[n][1])
            if type(logits) == np.ndarray:
                encoding_logit[idx] = logits[i][label_name_indices[k]]
            encoding[idx] = 1
        encodings.append(np.array(encoding))
        ##### convert logits #####
        if type(logits) == np.ndarray:
            for e in range(len(encoding_logit)):
                if encoding_logit[e] == 0:
                    if refer_node[e] in base_node:
                        q = base_node.index(refer_node[e])
                        encoding_logit[e] = logits[i][q]
                    else:
                        group_p_name = map_dict[refer_node[e]]
                        logit = logits[i][base_node.index(group_p_name)]
                        age = int(ages[i])
                        if age < 18:
                            if convert_dict[group_p_name][1] in refer_node:
                                encoding_logit[refer_node.index(convert_dict[group_p_name][1])] = logit
                            else:
                                encoding_logit[refer_node.index(convert_dict[group_p_name][0])] = logit
                        else:
                            if convert_dict[group_p_name][0] in refer_node:
                                encoding_logit[refer_node.index(convert_dict[group_p_name][0])] = logit
                            else:
                                encoding_logit[refer_node.index(convert_dict[group_p_name][1])] = logit
        encoding_logits.append(np.array(encoding_logit))
    return encodings, encoding_logits

def predZeros(onehot, logits):
    for i in range(len(onehot)):
        if not onehot[i].any():
            if torch.is_tensor(logits[i]):
                idx = torch.argmax(logits[i]).cpu().numpy()
            else:
                idx = np.argmax(logits[i])
            onehot[i][idx] = 1

def log_metrics(preds, labels, train_label_cnt, val_label_cnt, test_label_cnt):
    report = None
    print('begin generating evaluation reports...')
    report = gen_report(preds, labels, train_label_cnt, val_label_cnt, test_label_cnt)
    print('end generating evaluation reports')
    return report

def gen_report(preds, labels, train_label_cnt, val_label_cnt, test_label_cnt):
    y_score = np.array([t.detach().to(torch.float).cpu().numpy() for t in preds])
    OHE = [np.where(t.detach().to(torch.float).cpu().numpy() > 0.5, 1, 0) for t in preds]
    predZeros(OHE, preds)
    y_true = [np.where(t.detach().to(torch.float).cpu().numpy() > 0.5, 1, 0) for t in labels]

    confusion_matrix = multilabel_confusion_matrix(np.array(y_true), np.array(OHE))
    TP = [i[1][1] for i in confusion_matrix]
    FP = [i[0][1] for i in confusion_matrix]
    FN = [i[1][0] for i in confusion_matrix]
    TN = [i[0][0] for i in confusion_matrix]

    target_name = label
    report = metrics.classification_report(np.array(y_true), np.array(OHE), target_names=target_name, output_dict=True)

    for i in range(len(target_name)):
        pname = target_name[i]
        report[pname]['TP'] = int(TP[i])
        report[pname]['FP'] = int(FP[i])
        report[pname]['FN'] = int(FN[i])
        report[pname]['TN'] = int(TN[i])

    ## add train cnt, val cnt, test cnt
    filterings = ['micro avg', 'macro avg', 'weighted avg', 'samples avg']
    for k, vs in report.items():
        if k in filterings:
            continue
        if k in train_label_cnt:
            report[k]['train cnt'] = train_label_cnt[k]
        else:
            report[k]['train cnt'] = 0

        if k in val_label_cnt:
            report[k]['val cnt'] = val_label_cnt[k]
        else:
            report[k]['val cnt'] = 0

        if k in test_label_cnt:
            report[k]['test cnt'] = test_label_cnt[k]
        else:
            report[k]['test cnt'] = 0

    p1 = get_precision_at_k(np.array(y_true), y_score, k=1)
    r1 = get_recall_at_k(np.array(y_true), y_score, k=1)
    dcg1 = get_ndcg_at_k(np.array(y_true), y_score, k=1)
    rprecision1 = get_r_precision_at_k(np.array(y_true), y_score, k=1)
    report['P@1'] = p1
    report['R@1'] = r1
    report['nDCG@1'] = dcg1
    report['R-Precision@1'] = rprecision1

    p6 = get_precision_at_k(np.array(y_true), y_score, k=6)
    r6 = get_recall_at_k(np.array(y_true), y_score, k=6)
    dcg6 = get_ndcg_at_k(np.array(y_true), y_score, k=6)
    rprecision6 = get_r_precision_at_k(np.array(y_true), y_score, k=6)
    report['P@6'] = p6
    report['R@6'] = r6
    report['nDCG@6'] = dcg6
    report['R-Precision@6'] = rprecision6


    p8 = get_precision_at_k(np.array(y_true), y_score, k=8)
    r8 = get_recall_at_k(np.array(y_true), y_score, k=8)
    dcg8 = get_ndcg_at_k(np.array(y_true), y_score, k=8)
    rprecision8 = get_r_precision_at_k(np.array(y_true), y_score, k=8)
    report['P@8'] = p8
    report['R@8'] = r8
    report['nDCG@8'] = dcg8
    report['R-Precision@8'] = rprecision8


    p12 = get_precision_at_k(np.array(y_true), y_score, k=12)
    r12 = get_recall_at_k(np.array(y_true), y_score, k=12)
    dcg12 = get_ndcg_at_k(np.array(y_true), y_score, k=12)
    rprecision12 = get_r_precision_at_k(np.array(y_true), y_score, k=12)
    report['P@12'] = p12
    report['R@12'] = r12
    report['nDCG@12'] = dcg12
    report['R-Precision@12'] = rprecision12


    with open(os.path.join(DIR, '{}_Label_cnt.json'.format(dataset)), 'r') as f:
        label_cnt = json.load(f)
    head_p, mid_p, tail_p = segProtocols(label_cnt)
    final_report = genSegReport(report, head_p, mid_p, tail_p, y_score, np.array(y_true), topk_list=[1, 6, 8, 12])
    return final_report

def common_template(pred, label, target_name):
    confusion_matrix = multilabel_confusion_matrix(np.array(label), np.array(pred))
    TP = [i[1][1] for i in confusion_matrix]
    FP = [i[0][1] for i in confusion_matrix]
    FN = [i[1][0] for i in confusion_matrix]
    TN = [i[0][0] for i in confusion_matrix]
    report = metrics.classification_report(np.array(label), np.array(pred), target_names=target_name, output_dict=True)
    for i in range(len(target_name)):
        pname = target_name[i]
        report[pname]['TP'] = int(TP[i])
        report[pname]['FP'] = int(FP[i])
        report[pname]['FN'] = int(FN[i])
        report[pname]['TN'] = int(TN[i])
    return report

def postEval(report, y_score, y_true, ages, mode, topk_list):
    with open('./json files/{}/{}_Label_cnt.json'.format(dataset, dataset), 'r') as f:
        label_cnt = json.load(f)
    head_p, mid_p, tail_p = segProtocols(label_cnt)


    OHE = [np.where(t > 0.5, 1, 0) for t in y_score]
    predZeros(OHE, y_score)

    if mode == 'group':
        OHE, y_score = convert_label(OHE, ages, y_score, metric='age')

    y_true = [np.where(t > 0.5, 1, 0) for t in y_true]
    if mode == 'group':
        y_true, _ = convert_label(y_true, ages, logits=None, metric='age')

    for k in topk_list:
        pk = get_precision_at_k(np.array(y_true), y_score, k=k)
        rk = get_recall_at_k(np.array(y_true), y_score, k=k)
        dcgk = get_ndcg_at_k(np.array(y_true), y_score, k=k)
        rprecisionk = get_r_precision_at_k(np.array(y_true), y_score, k=k)
        report['P@{}'.format(k)] = pk
        report['R@{}'.format(k)] = rk
        report['nDCG@{}'.format(k)] = dcgk
        report['R-Precsion@{}'.format(k)] = rprecisionk


    final_report = genSegReport(report, head_p, mid_p, tail_p, y_score, y_true, topk_list=topk_list)
    return final_report


if __name__ == "__main__":
    import os
    import json

    root = '/home/xueren/Desktop/EMS/results/AAAI'
    dataset_name = 'MIMIC-III'
    model = 'BioMedLM (copy)'
    report_name = 'ungroup_test.json' if 'MIMIC' in dataset_name else 'ungroup_results.json'
    for root_, dir_, files_ in os.walk(os.path.join(root, dataset_name, model)):
        for file in files_:
            if file == report_name:
                with open(os.path.join(root_, file), 'r') as f:
                    report = json.load(f)
                print(root_)
                report.pop('head protocols')
                report.pop('middle protocols')
                report.pop('tail protocols')
                report.pop('P@1')
                report.pop('R@1')
                report.pop('nDCG@1')
                report.pop('P@3')
                report.pop('R@3')
                report.pop('nDCG@3')
                report.pop('R-Precsion@3')
                report.pop('P@5')
                report.pop('R@5')
                report.pop('nDCG@5')
                report.pop('R-Precsion@5')
                report.pop('P@10')
                report.pop('R@10')
                report.pop('nDCG@10')
                report.pop('R-Precsion@10')
                y_score_path = os.path.join(root_, 'Pred.npy')
                y_score = np.load(y_score_path)
                y_true_path = os.path.join(root_, 'GT.npy')
                y_true = np.load(y_true_path)
                age_path = os.path.join(root_, 'Ages.npy')
                age = np.load(age_path)

                if 'cluster:group' in root_:
                    mode = 'group'
                else:
                    mode = 'ungroup'
                print(mode)
                new_final_report = postEval(report, y_score, y_true, age, mode, topk_list=[1, 6])

                with open(os.path.join(root_, file), 'w') as f:
                    json.dump(new_final_report, f, indent=4)



