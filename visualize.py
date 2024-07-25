import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from default_sets import *
import os

def vis_train_details(train_dict, save_result_root):
    root = os.path.join(os.path.dirname(save_result_root), 'training details')
    if not os.path.exists(root):
        os.makedirs(root)

    for k, v in train_dict.items():
        plt.figure()
        plt.plot(np.arange(len(v)), v, label=k)
        plt.xlabel('epoch')
        plt.ylabel('{}'.format(k))
        plt.grid(True, which='major', linewidth=0.5)
        plt.grid(True, which='minor', linewidth=0.1)
        plt.legend()
        plt.savefig(os.path.join(root, '{}.jpg'.format(k)))



def plot_distribution(df, save_root):
    label_cnt = {}
    for i in range(len(df)):
        ps = df['Protocols'][i].strip()
        for p in ps.split(';'):
            p = p.strip()
            label_cnt[p] = label_cnt.get(p, 0) + 1
    cnt = list(label_cnt.values())
    name = list(label_cnt.keys())
    zipped = zip(name, cnt)
    sort_zipped = sorted(zipped, key=lambda x: (x[1], x[0]), reverse=True)
    result = zip(*sort_zipped)
    name, cnt = [list(x) for x in result]

    plt.figure(figsize=(15, 16))
    plt.barh(name, cnt, height=0.7, color='steelblue', alpha=0.8)
    plt.xticks(rotation=0, fontsize=8)

    plt.xlabel('count')
    for idx, v in enumerate(cnt):
        plt.text(v + 0.01, idx - 0.1, '{:.2f}'.format(v))
    plt.title('{} label-based data distribution'.format(len(name)))
    save_path = os.path.join(save_root, 'Figures')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, 'label_distribution.jpg'))


def sortBy(l1, l2, reverse=True):
    x_axis, y_axis = [], []
    if l1 and l2:
        zipped = zip(l1, l2)
        sort_zipped = sorted(zipped, key=lambda x:(x[1], x[0]), reverse=reverse)
        result = zip(*sort_zipped)
        x_axis, y_axis = [list(x) for x in result]
    return x_axis, y_axis


def plot_f1_p_r(report, path, fig_name):
    if not report:
        return
    f1 = []
    precision = []
    recall = []
    name = []
    filtering = ['micro avg', 'macro avg', 'weighted avg', 'samples avg', 'auc_micro', 'auc_macro', 'auc_macro',
                 'P@1', 'R@1', 'R-Precision@1', 'nDCG@1', 'P@3', 'R@3', 'R-Precision@3', 'nDCG@3',
                 'P@5', 'R@5', 'R-Precision@5', 'nDCG@5', 'P@6', 'R@6', 'R-Precision@6', 'nDCG@6',
                 'P@10', 'R@10', 'R-Precision@10', 'nDCG@10',
                 'head protocols', 'middle protocols', 'tail protocols']
    for k, v in report.items():
        if k in filtering:
            continue
        name.append(k)
        f1.append(v['f1-score'])
        precision.append(v['precision'])
        recall.append(v['recall'])
    ori_name = name
    f1, _ = sortBy(f1, ori_name, reverse=False)
    precision, _ = sortBy(precision, ori_name, reverse=False)
    recall, name = sortBy(recall, ori_name, reverse=False)

    fig = plt.figure(figsize=(25, 17))
    x = range(len(f1))
    ax1 = fig.add_subplot(311)
    ax1.bar(x, precision, width=0.7, alpha=0.8, color='y', label='precision')
    ax1.set_ylabel('precision', fontsize=14)
    for a, b in zip(x, precision):
        plt.text(a, b, np.round(b, 3), ha='center', va='bottom', fontsize=12)

    ax2 = fig.add_subplot(312)
    ax2.bar(x, recall, width=0.7, alpha=0.8, color='purple', label='recall')
    ax2.set_ylabel('recall', fontsize=14)
    for a, b in zip(x, recall):
        plt.text(a, b, np.round(b, 3), ha='center', va='bottom', fontsize=12)

    ax3 = fig.add_subplot(313)
    ax3.bar(name, f1, width=0.7, alpha=0.8, color='b', label='f1 score')
    ax3.set_ylabel('f1 score', fontsize=14)
    for a, b in zip(x, f1):
        plt.text(a, b, np.round(b, 3), ha='center', va='bottom', fontsize=12)
        plt.setp(ax3.get_xticklabels(), rotation=35, horizontalalignment='right', fontsize=14)

    plt.savefig(os.path.join(path, '{}_f1_precision_recall.jpg'.format(fig_name)))


def plot_trainCnt_f1_p_r(report, path, fig_name):
    if not report:
        return
    f1 = []
    precision = []
    recall = []
    train = []
    test = []
    name = []
    filtering = ['micro avg', 'macro avg', 'weighted avg', 'samples avg', 'auc_micro', 'auc_macro',
                 'auc_macro', 'P@1', 'R@1', 'R-Precsion@1', 'nDCG@1', 'P@3', 'R@3', 'R-Precsion@3', 'nDCG@3',
                 'P@5', 'R@5', 'nDCG@5', 'R-Precsion@5', 'P@6', 'R@6', 'R-Precision@6', 'nDCG@6',
                 'P@10', 'R@10', 'R-Precsion@10', 'nDCG@10',
                 'head protocols', 'middle protocols', 'tail protocols']
    for k, v in report.items():
        if k in filtering:
            continue

        name.append(k)
        f1.append(v['f1-score'])
        precision.append(v['precision'])
        recall.append(v['recall'])
        train.append(v['train cnt'])
        test.append(v['test cnt'])


    # ori_name = name
    # f1, _ = sortBy(f1, ori_name)
    # precision, _ = sortBy(precision, ori_name)
    # train, _ = sortBy(train, ori_name)
    # test, _ = sortBy(test, ori_name)
    # recall, name = sortBy(recall, ori_name)

    ori_train = train
    f1, _ = sortBy(f1, ori_train)
    precision, _ = sortBy(precision, ori_train)
    name, _ = sortBy(name, ori_train)
    test, _ = sortBy(test, ori_train)
    recall, train = sortBy(recall, ori_train)

    fig = plt.figure(figsize=(25, 17))
    x = range(len(f1))
    ax = fig.add_subplot(511)
    ax.bar(x, train, width=0.7, alpha=0.8, color='r', label='train cnt')
    ax.set_ylabel('train cnt', fontsize=14)
    ax.set_xticks([])
    for a, b in zip(x, train):
        ax.text(a, b + 1, int(b), ha='center', va='bottom', fontsize=12)

    ax2 = fig.add_subplot(512)
    ax2.bar(x, test, width=0.7, alpha=0.8, color='g', label='test cnt')
    ax2.set_ylabel('test cnt', fontsize=14)
    for a, b in zip(x, test):
        plt.text(a, b, int(b), ha='center', va='bottom', fontsize=12)

    ax3 = fig.add_subplot(513)
    ax3.bar(x, precision, width=0.7, alpha=0.8, color='y', label='precision')
    ax3.set_ylabel('precision', fontsize=14)
    for a, b in zip(x, precision):
        plt.text(a, b, np.round(b, 3), ha='center', va='bottom', fontsize=12)

    ax4 = fig.add_subplot(514)
    ax4.bar(x, recall, width=0.7, alpha=0.8, color='purple', label='recall')
    ax4.set_ylabel('recall', fontsize=14)
    for a, b in zip(x, recall):
        plt.text(a, b, np.round(b, 3), ha='center', va='bottom', fontsize=12)

    ax5 = fig.add_subplot(515)
    ax5.bar(name, f1, width=0.7, alpha=0.8, color='b', label='f1 score')
    ax5.set_ylabel('f1 score', fontsize=14)
    for a, b in zip(x, f1):
        plt.text(a, b, np.round(b, 3), ha='center', va='bottom', fontsize=12)
        plt.setp(ax5.get_xticklabels(), rotation=35, horizontalalignment='right', fontsize=14)

    plt.savefig(os.path.join(path, '{}_f1_precision_recall.jpg'.format(fig_name)))


def plot_loss_histogram(loss, path):
    plt.figure(figsize=(10, 7), tight_layout=True)
    plt.hist(np.array(loss))
    plt.savefig(os.path.join(path, 'test_loss_distribution.jpg'))


def plot_logits_histogram(values, unvalues, path):
    plt.figure(tight_layout=True)
    plt.hist(np.array(values), bins=5, color='b', alpha=0.3, label='correct prediction')
    plt.hist(np.array(unvalues), bins=5, color='r', alpha=0.3, label='incorrect predictions')
    plt.xlabel('logits')
    plt.ylabel('amount')
    plt.legend()
    save_path = os.path.join(path, 'Figures')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, 'test_logits_distribution.jpg'))


def plot_barchart(CNN_f1, BioBERT_f1, COReBERT_f1, GatorTron_f1, BioGPT_f1, metric, title, root):
    '''
    :param BioBERT_f1: tuple
    :param COReBERT_f1: tuple
    :param GatorTron_f1: tuple
    :return:
    '''
    model = ("base", "group", "la", "comb")
    f1_score = {
        'CNN': CNN_f1,
        'BioBERT': BioBERT_f1,
        'COReBERT': COReBERT_f1,
        'GatorTron': GatorTron_f1,
        'BioGPT': BioGPT_f1,
    }
    x = np.arange(len(model))
    width = 0.14 #width of bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(8, 7), layout='constrained')

    ax.axes.tick_params(which='both', direction='in', top=True, right=True)
    plt.minorticks_on()
    ax.set_facecolor((0, 0, 0, 0.02))


    for attribute, measurement in f1_score.items():
        offset = width * multiplier
        rects = ax.bar(x+offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=2, fontsize=8)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('{}'.format(metric), fontsize='17')
    ax.set_title('{} {}'.format(title, metric), fontsize=17)
    ax.set_xticks(x + width, model, fontsize='17')
    ax.yaxis.set_tick_params(labelsize=17)
    ax.legend(loc='upper left', fontsize="15", ncols=3)
    ax.set_ylim(0, 1)
    plt.grid(True, which='major', linewidth=0.5)
    plt.grid(True, which='minor', linewidth=0.1)
    # plt.show()
    plt.savefig(os.path.join(root, '{} of {}.jpg'.format(metric, title)), dpi=600)


### the following codes are used for visualize
if __name__ == '__main__':
    base_root = './results/EMNLP Experiments'
    # models = ['BioBERT', 'BioBERT+group', 'BioBERT+la', 'BioBERT+group+la',
    #           'COReBERT', 'COReBERT+group', 'COReBERT+la', 'COReBERT+group+la',
    #           'GatorTron', 'GatorTron+group', 'GatorTron+la', 'GatorTron+group+la']

    models = ['2023-04-24-10_15_35', '2023-04-24-10_21_20', '2023-04-24-10_10_27', '2023-04-24-10_33_34', #CNN
              '2023-04-05-10_19_58', '2023-04-06-15_08_10', '2023-04-05-15_53_23', '2023-04-06-20_05_21', #BioBERT
              '2023-04-05-11_20_28', '2023-04-06-16_03_46', '2023-04-05-16_47_21', '2023-04-06-20_59_50', #COReBERT
              '2023-04-05-12_36_04', '2023-04-06-17_17_09', '2023-04-05-18_19_35', '2023-04-06-21_54_18', #GatorTron
              '2023-04-19-17_36_29', '2023-04-19-20_21_41', '2023-04-20-09_21_28', '2023-04-20-14_17_41', #BioGPT
              '2023-04-20-15_43_57', '2023-04-21-14_08_20', '2023-04-20-15_50_48', '2023-04-21-14_13_01' #BioMedLM
              ]



    types = ['tail protocols', 'middle protocols', 'head protocols']
    for type in types:
        CNN = defaultdict(list)
        BioBERT = defaultdict(list)
        COReBERT = defaultdict(list)
        GatorTron = defaultdict(list)
        BioGPT = defaultdict(list)
        BioMedLM = defaultdict(list)

        for i, m in enumerate(models):
            with open(os.path.join(base_root, m, 'ungroup_results2.json'), 'r') as f:
                file = json.load(f)
            # CNN
            precision_ = 2
            if 0 <= i < 4:
                CNN['precision'].append(round(file[type]['precision'], precision_))
                CNN['recall'].append(round(file[type]['recall'], precision_))
                CNN['f1-score'].append(round(file[type]['f1-score'], precision_))
            # BioBERT
            elif 4 <= i < 8:
                BioBERT['precision'].append(round(file[type]['precision'], precision_))
                BioBERT['recall'].append(round(file[type]['recall'], precision_))
                BioBERT['f1-score'].append(round(file[type]['f1-score'], precision_))
            # COReBERT
            elif 8 <= i < 12:
                COReBERT['precision'].append(round(file[type]['precision'], precision_))
                COReBERT['recall'].append(round(file[type]['recall'], precision_))
                COReBERT['f1-score'].append(round(file[type]['f1-score'], precision_))
            # GatorTron
            elif 12 <= i < 16:
                GatorTron['precision'].append(round(file[type]['precision'], precision_))
                GatorTron['recall'].append(round(file[type]['recall'], precision_))
                GatorTron['f1-score'].append(round(file[type]['f1-score'], precision_))
            # BioGPT
            elif 16 <= i < 20:
                BioGPT['precision'].append(round(file[type]['precision'], precision_))
                BioGPT['recall'].append(round(file[type]['recall'], precision_))
                BioGPT['f1-score'].append(round(file[type]['f1-score'], precision_))
            # BioMedLM
            elif 20 <= i < 24:
                BioMedLM['precision'].append(round(file[type]['precision'], precision_))
                BioMedLM['recall'].append(round(file[type]['recall'], precision_))
                BioMedLM['f1-score'].append(round(file[type]['f1-score'], precision_))

        for k, v in CNN.items():
            CNN[k] = tuple(v)
        for k, v in BioBERT.items():
            BioBERT[k] = tuple(v)
        for k, v in COReBERT.items():
            COReBERT[k] = tuple(v)
        for k, v in GatorTron.items():
            GatorTron[k] = tuple(v)
        for k, v in BioGPT.items():
            BioGPT[k] = tuple(v)
        for k, v in BioMedLM.items():
            BioMedLM[k] = tuple(v)

        plot_barchart(CNN['f1-score'], BioBERT['f1-score'], COReBERT['f1-score'],
                      GatorTron['f1-score'], BioGPT['f1-score'], BioMedLM['f1-score'], type, base_root)





