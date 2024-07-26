import re

from torch_geometric.data import HeteroData, Data
import torch
import torch.nn as nn
import pandas as pd
from collections import defaultdict
from utils import sortby, removePunctuation, preprocess
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import nltk
from default_sets import *
from logger import is_main_process
import os
import json
import nltk
nltk.download('stopwords')
nltk.download('punkt')

class bertEmbedding():
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = tokenizer.eos_token
        # if dataset == 'RAA':
        #     self.max_len = 256
        # elif dataset == 'MIMIC3':
        #     self.max_len = 32
        # else:
        #     raise Exception('check dataset in default_sets.py')
        # self.max_len = 256

    def tokenization(self, sentence):
        sentence = preprocess(sentence)
        # inputs = self.tokenizer.__call__(sentence,
        #                                  None,
        #                                  add_special_tokens=True,
        #                                  max_length=self.max_len,
        #                                  padding="max_length",
        #                                  truncation=True)
        inputs = self.tokenizer.__call__(sentence, None, add_special_tokens=True, padding=False)
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0), torch.tensor(mask, dtype=torch.long).unsqueeze(0)

    def fusion(self, token_embeddings, mode):
        if mode == 'SUM':
            '''
            sum up last 4 embeddings of biobert, and take 
            the mean in each dimension to get sentence embedding
            '''
            token_vecs_sum = []
            for token in token_embeddings:
                sum_vec = torch.sum(token[-4:], dim=0)
                token_vecs_sum.append(sum_vec.cpu().numpy())
            return np.array(token_vecs_sum).mean(axis=0)
        elif mode == 'MEAN':
            '''
            take the second 2 last layer and then take the mean
            '''
            token_vecs = token_embeddings[:, -2, :]
            sentence_embedding = torch.mean(token_vecs, dim=0)
            return sentence_embedding.numpy()
        else:
            raise Exception("check embedding mode")

    def getPreEmbedding(self, node, method='SUM'):
        ids, segs = self.tokenization(node)
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_ids=ids.to(self.device), attention_mask=segs.to(self.device))
        token_embeddings = torch.stack(output.hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)

        sentence_embedding = self.fusion(token_embeddings, method)
        return sentence_embedding.tolist()

class HeteroGraph(nn.Module):
    def __init__(self, config):
        super(HeteroGraph, self).__init__()

        if config.train.backbone == 'CNN' or config.train.backbone=='RNN':
            self.backbone = 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12'
        else:
            self.backbone = config.train.backbone

        self.p_node = label
        self.hier_node = list(hier2label.keys())
        self.ICD9_description = ICD9_description if "MIMIC3" in dataset else None
        self.sign_node = []
        self.treat_node = []
        self.h2idx = {}
        self.p2idx = {}
        self.s2idx = {}
        self.t2idx = {}
    @staticmethod
    def p2overview(signs_df):
        # create mapping (protocol --- overview)
        p2overview = {}
        for i in range(len(signs_df)):
            overview = signs_df['Overview'][i]
            if type(overview) == float:
                continue
            pname = signs_df['Protocols'][i] + ' ({})'.format(signs_df['Protocol ID'][i])
            pname = pname.lower()
            p2overview[pname] = overview
        return p2overview

    # the following mapping relations are generated from protocol guidelines
    @staticmethod
    def p2signs(signs_df):
        # create mapping (protocol --- signs)
        p2s = defaultdict(list)
        # p2s_d = defaultdict(dict)
        s2p = defaultdict(list)
        for i in range(len(signs_df)):
            ss = signs_df['Signs and Symptoms(in impression list)'][i]
            if type(ss) == float:
                continue
            pname = signs_df['Protocols'][i] + ' ({})'.format(signs_df['Protocol ID'][i])
            pname = pname.lower()
            for s in ss.split(';'):
                s = s.strip().capitalize()
                if pname not in s2p[s]:
                    s2p[s].append(pname)
                if s not in p2s[pname]:
                    p2s[pname].append(s)
        return p2s, s2p

    @staticmethod
    def p2impre(impre_df):
        p2impre = defaultdict(list)
        # p2impre_d = defaultdict(dict)
        impre2p = defaultdict(list)
        for i in range(len(impre_df)):
            ps = impre_df['Protocol_name'][i]
            candi_ps = impre_df['Indication/Considerations of Protocols'][i]
            impre = impre_df['Impression'][i].strip().capitalize()
            if type(impre) == float:
                continue
            if type(ps) != float:
                for p in ps.split(';'):
                    p = p.strip().lower()

                    if impre not in p2impre[p]:
                        p2impre[p].append(impre)
                    if p not in impre2p[impre]:
                        impre2p[impre].append(p)
            if type(candi_ps) != float:
                for c_p in candi_ps.split(';'):
                    c_p = c_p.strip().lower()

                    if impre not in p2impre[c_p]:
                        p2impre[c_p].append(impre)
                    if c_p not in impre2p[impre]:
                        impre2p[impre].append(c_p)
        return p2impre, impre2p

    @staticmethod
    def p2med(med_df):
        # create mapping (protocol --- medication)
        p2m = defaultdict(list)
        # p2m_d = defaultdict(dict)
        m2p = defaultdict(list)

        for i in range(len(med_df)):
            med_name = med_df['medication'][i].strip().capitalize()
            med_name_idx = re.search(r'[(]\d+[)]', med_name).start()
            med_name = med_name[:med_name_idx].strip()
            pname = ''
            if type(med_df['protocols'][i]) != float:
                pname += med_df['protocols'][i] + '; '
            if type(med_df['considerations'][i]) != float:
                pname += med_df['considerations'][i]
            if type(pname) != float:
                pname = pname.lower()
            for p in pname.split(';'):
                p = p.lower().strip()
                if p == '':
                    continue
                # p2m_d[p][med_name] = p2m_d[p].get(med_name, 0) + 1
                if med_name not in p2m[p]:
                    p2m[p].append(med_name)
                if p not in m2p[med_name]:
                    m2p[med_name].append(p)
        return p2m, m2p

    @staticmethod
    def p2proc(proc_df):
        # create mapping (protocol --- procedure)
        p2proc = defaultdict(list)
        # p2proc_d = defaultdict(dict)
        proc2p = defaultdict(list)
        proc_filter = ['Assess airway', 'Ecg', 'Iv', 'Move patient', 'Bvm', 'Assess - pain assessment',
                       'Im/sq injection']
        for i in range(len(proc_df)):
            procedure = proc_df['Grouping'][i].strip().capitalize()
            if procedure in proc_filter:
                continue

            if type(proc_df['Protocols'][i]) == float:
                continue
            for p in proc_df['Protocols'][i].split(';'):
                p = p.lower().strip()

                # p2proc_d[p][procedure] = p2proc_d[p].get(procedure, 0) + 1
                if procedure not in p2proc[p]:
                    p2proc[p].append(procedure)
                if p not in proc2p[procedure]:
                    proc2p[procedure].append(p)
        return p2proc, proc2p

    @staticmethod
    def p2treatment(p2proc, proc2p, p2m, m2p):
        p2treat = defaultdict(list)
        treat2p = defaultdict(list)

        for p, proc in p2proc.items():
            p2treat[p].extend(proc)
        for p, m in p2m.items():
            p2treat[p].extend(m)

        for proc, p in proc2p.items():
            treat2p[proc].extend(p)
        for m, p in m2p.items():
            treat2p[m].extend(p)
        return p2treat, treat2p

    @staticmethod
    def protocol2sym():
        p2s_path = '%s/protocol2symptom.json' % DIR
        s2p_path = '%s/symptom2protocol.json' % DIR
        with open(p2s_path, 'r') as f:
            p2s = json.load(f)
        with open(s2p_path, 'r') as f:
            s2p = json.load(f)
        return p2s, s2p
    @staticmethod
    def protocol2treat():
        p2t_path = '%s/protocol2treatment.json' % DIR
        t2p_path = '%s/treatment2protocol.json' % DIR
        with open(p2t_path, 'r') as f:
            p2t = json.load(f)
        with open(t2p_path, 'r') as f:
            t2p = json.load(f)
        return p2t, t2p

    @staticmethod
    def diag2overview(ICD9_description):
        return ICD9_description

    @staticmethod
    def diag2sign(ICD9_description):
        # d2s_path = '%s/diag2sign.json' % DIR
        d2s_path = '%s/icd9code2symptom.json' % DIR
        # d2s_path = '%s/wikipedia/diag2sign.json' % DIR
        with open(d2s_path, 'r') as f:
            d2s_raw = json.load(f)
        d2s = defaultdict(list)
        for k, vs in d2s_raw.items():
            for v in vs:
                # d2s[k].append(ICD9_description[v])
                d2s[k].append(v)

        # s2d_path = '%s/sign2diag.json' % DIR
        s2d_path = '%s/symptom2icd9code.json' % DIR
        # s2d_path = '%s/wikipedia/sign2diag.json' % DIR
        with open(s2d_path, 'r') as f:
            s2d_raw = json.load(f)
        s2d = defaultdict(list)
        for k, vs in s2d_raw.items():
            for v in vs:
                # s2d[ICD9_description[k]].append(v)
                s2d[k].append(v)
        return d2s, s2d

    @staticmethod
    def diag2treament(ICD9_description):
        d2t_path = '%s/icd9code2treatment.json' % DIR
        # d2t_path = '%s/wikipedia/diag2treatment.json' % DIR
        with open(d2t_path, 'r') as f:
            d2t_raw = json.load(f)
        d2t = defaultdict(list)
        for k, vs in d2t_raw.items():
            for v in vs:
                d2t[k].append(v)

        t2d_path = '%s/treatment2icd9code.json' % DIR
        # t2d_path = '%s/wikipedia/treatment2diag.json' % DIR
        with open(t2d_path, 'r') as f:
            t2d_raw = json.load(f)
        t2d = defaultdict(list)
        for k, vs in t2d_raw.items():
            for v in vs:
                t2d[k].append(v)
        return d2t, t2d

    @staticmethod
    def diag2hier():
        p2h = {}
        for k, v in label2hier.items():
            p2h[k.capitalize()] = [v]
        return p2h, hier2label

    def _genNode(self, p2overview, s2p, t2p):
        for k, v in s2p.items():
            # check if sign-mapped protocols is in p_node
            for i in v:
                if i in self.p_node and k not in self.sign_node:
                    self.sign_node.append(k)
                    break

        for k, v in t2p.items():
            for i in v:
                if i in self.p_node and k not in self.treat_node:
                    self.treat_node.append(k)
                    break


        node = []
        node.extend(self.hier_node)
        node.extend(self.p_node)
        node.extend(self.sign_node)
        node.extend(self.treat_node)

        pre_trained_embed_root = os.path.join(ROOT, 'pre-trained embedding')
        if not os.path.exists(pre_trained_embed_root):
            os.makedirs(pre_trained_embed_root)
        backbone_name = self.backbone.split('/')[-1]
        dataset_name = 'MIMIC3' if 'MIMIC3' in dataset else 'RAA'
        nodes_attr_cache = os.path.join(pre_trained_embed_root,
                                        '{}_node_embedding_{}.json'.format(dataset_name, backbone_name))

        if not os.path.exists(nodes_attr_cache):
            cache_dict = {}
            nodes_attr = []
            tokenizer = AutoTokenizer.from_pretrained(self.backbone, do_lower_Case=True)
            model = AutoModel.from_pretrained(self.backbone, output_hidden_states=True)
            b_embed = bertEmbedding(tokenizer, model)
            h, q, w, e = 0, 0, 0, 0
            for i, n in enumerate(tqdm(node, desc='build heterogeneous graph')):
                attr = {}
                if n in self.hier_node:
                    n_type = 'hierarchy'
                    n_feature = b_embed.getPreEmbedding(preprocess(n))
                    self.h2idx[i] = h
                    h += 1
                elif n in self.p_node:
                    n_type = 'protocol'
                    n_feature = b_embed.getPreEmbedding(preprocess(p2overview[n]))
                    self.p2idx[i] = q
                    q += 1
                elif n in self.sign_node:
                    n_type = 'sign'
                    n_feature = b_embed.getPreEmbedding(preprocess(n))
                    self.s2idx[i] = w
                    w += 1
                elif n in self.treat_node:
                    n_type = 'treatment'
                    n_feature = b_embed.getPreEmbedding(preprocess(n))
                    self.t2idx[i] = e
                    e += 1
                else:
                    raise Exception('Node type is incorrect, recheck node type')
                attr[n_type] = n
                attr['node_type'] = n_type
                attr['node_feature'] = n_feature
                nodes_attr.append((i, attr))
                cache_dict[n] = n_feature
            if is_main_process():
                with open(nodes_attr_cache, 'w') as f:
                    json.dump(cache_dict, f, indent=4)
        else:
            with open(nodes_attr_cache, 'r') as f:
                nodes_attr_ = json.load(f)
            nodes_attr = []
            h, q, w, e = 0, 0, 0, 0
            for i, n in enumerate(tqdm(node, desc='build heterogeneous graph')):
                attr = {}
                if n in self.hier_node:
                    n_type = 'hierarchy'
                    self.h2idx[i] = h
                    h += 1
                elif n in self.p_node:
                    n_type = 'protocol'
                    self.p2idx[i] = q
                    q += 1
                elif n in self.sign_node:
                    n_type = 'sign'
                    self.s2idx[i] = w
                    w += 1
                elif n in self.treat_node:
                    n_type = 'treatment'
                    self.t2idx[i] = e
                    e += 1
                else:
                    raise Exception('Node type is incorrect, recheck node type')
                n_feature = nodes_attr_[n]
                attr[n_type] = n
                attr['node_type'] = n_type
                attr['node_feature'] = n_feature
                nodes_attr.append((i, attr))
        return node, nodes_attr

    def __checkEdges(self, node, src_node, dic):
        edges = []
        if src_node in dic:
            for i, dst_node in enumerate(dic[src_node]):

                src_node_idx = node.index(src_node)
                dst_node_idx = node.index(dst_node)

                if dst_node in self.hier_node:
                    edge_type = 'is children of'
                elif dst_node in self.sign_node:
                    edge_type = 'has'
                elif dst_node in self.treat_node:
                    edge_type = 'suggests'
                else:
                    raise Exception('recheck edge type')
                edges.append((src_node_idx, dst_node_idx, {'edge_type': edge_type}))
        return edges

    def _genEdge(self, node, p2h, p2s, p2t):
        edges = []
        for i, n in enumerate(node):
            edge_h = self.__checkEdges(node, n, p2h)
            if edge_h:
                edges.extend(edge_h)
            edge_s = self.__checkEdges(node, n, p2s)
            if edge_s:
                edges.extend(edge_s)
            edge_t = self.__checkEdges(node, n, p2t)
            if edge_t:
                edges.extend(edge_t)
        return edges

    @staticmethod
    def transIdx(idx, hier_idx_map, protocol_idx_map, sign_idx_map, treat_idx_map):
        if idx in hier_idx_map:
            return hier_idx_map[idx], 'hierarchy'
        elif idx in protocol_idx_map:
            return protocol_idx_map[idx], 'protocol'
        elif idx in sign_idx_map:
            return sign_idx_map[idx], 'sign'
        elif idx in treat_idx_map:
            return treat_idx_map[idx], 'treatment'
        else:
            raise Exception('node indexing is incorrect')

    @staticmethod
    def checkEdgeWeight(src, dst, node, dst_name):
        with open('%s/wikipedia/tf-idf_diag2sign.json' % DIR, 'r') as f:
            tf_idf_diag2sign = json.load(f)
        with open('%s/wikipedia/tf-idf_diag2treatment.json' % DIR, 'r') as f:
            tf_idf_diag2treatment = json.load(f)

        if dst_name == 'sign':
            weight = tf_idf_diag2sign[node[src]][node[dst]]
        elif dst_name == 'treatment':
            weight = tf_idf_diag2treatment[node[src]][node[dst]]
        else:
            raise Exception('check edge type')
        return weight


    def _genGraph(self, node, nodes_attr, edges):
        # convert graph data into class HeteroData
        label_hetero = HeteroData()
        # add node features
        h_feat = []
        p_feat = []
        s_feat = []
        t_feat = []

        for (index, node_attr) in nodes_attr:
            if node_attr['node_type'] == 'hierarchy':
                h_feat.append(node_attr['node_feature'])
            elif node_attr['node_type'] == 'protocol':
                p_feat.append(node_attr['node_feature'])
            elif node_attr['node_type'] == 'sign':
                s_feat.append(node_attr['node_feature'])
            elif node_attr['node_type'] == 'treatment':
                t_feat.append(node_attr['node_feature'])
            else:
                raise Exception('incorrect node type')

        label_hetero['hierarchy'].x = torch.tensor(h_feat, dtype=torch.float)
        label_hetero['protocol'].x = torch.tensor(p_feat, dtype=torch.float)
        label_hetero['impression'].x = torch.tensor(s_feat, dtype=torch.float)
        label_hetero['treatment'].x = torch.tensor(t_feat, dtype=torch.float)

        ph_src, ph_dst = [], []
        phs_src, phs_dst = [], []
        pst_src, pst_dst = [], []
        for h in self.hier_node:
            if h in self.p_node:
                print('p_node', h)
            elif h in self.sign_node:
                print('s_node', h)
            elif h in self.treat_node:
                print('t_node', h)

        for (src, dst, _) in edges:
            src_idx, src_name = self.transIdx(src, self.h2idx, self.p2idx, self.s2idx, self.t2idx)
            dst_idx, dst_name = self.transIdx(dst, self.h2idx, self.p2idx, self.s2idx, self.t2idx)
            if dst_name == 'hierarchy':
                ph_src.append(src_idx)
                ph_dst.append(dst_idx)
            elif dst_name == 'sign':
                phs_src.append(src_idx)
                phs_dst.append(dst_idx)
            elif dst_name == 'treatment':
                pst_src.append(src_idx)
                pst_dst.append(dst_idx)

            else:
                raise Exception('check transIdx')

        label_hetero['protocol', 'is children of', 'hierarchy'].edge_index = torch.tensor([ph_src, ph_dst], dtype=torch.int64)
        label_hetero['protocol', 'has', 'impression'].edge_index = torch.tensor([phs_src, phs_dst], dtype=torch.int64)
        label_hetero['protocol', 'suggests', 'treatment'].edge_index = torch.tensor([pst_src, pst_dst], dtype=torch.int64)
        label_hetero['hierarchy', 'is parent of', 'protocol'].edge_index = torch.tensor([ph_dst, ph_src], dtype=torch.int64)
        label_hetero['impression', 'indicates', 'protocol'].edge_index = torch.tensor([phs_dst, phs_src], dtype=torch.int64)
        label_hetero['treatment', 'is suggested by', 'protocol'].edge_index = torch.tensor([pst_dst, pst_src], dtype=torch.int64)
        return label_hetero

    def forward(self, signs_df=None, impre_df=None, med_df=None, proc_df=None):
        if dataset == 'RAA':
            p2overview = self.p2overview(signs_df)
            # p2sign, sign2p = self.p2impre(impre_df)
            # p2med, med2p = self.p2med(med_df)
            # p2proc, proc2p = self.p2proc(proc_df)
            # p2treatment, treatment2p = self.p2treatment(p2proc, proc2p, p2med, med2p)
            p2sign, sign2p = self.protocol2sym()
            p2treatment, treatment2p = self.protocol2treat()
            p2hier, hier2p = self.diag2hier()
        elif 'MIMIC3' in dataset:
            p2overview = self.diag2overview(self.ICD9_description)
            p2sign, sign2p = self.diag2sign(self.ICD9_description)
            p2treatment, treatment2p = self.diag2treament(self.ICD9_description)
            p2hier, hier2p = self.diag2hier()
        else:
            raise Exception('check dataset in default_sets.py')

        node, nodes_attr = self._genNode(p2overview, sign2p, treatment2p)
        edges = self._genEdge(node, p2hier, p2sign, p2treatment)
        graph = self._genGraph(node, nodes_attr, edges)
        return graph


class HeteroGraphwoHier(nn.Module):
    def __init__(self, config):
        super(HeteroGraphwoHier, self).__init__()

        if config.train.backbone == 'CNN':
            self.backbone = 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12'
        else:
            self.backbone = config.train.backbone

        self.p_node = label
        self.ICD9_description = ICD9_description if dataset == "MIMIC3" else None
        self.sign_node = []
        self.treat_node = []
        self.p2idx = {}
        self.s2idx = {}
        self.t2idx = {}
    @staticmethod
    def p2overview(signs_df):
        # create mapping (protocol --- overview)
        p2overview = {}
        for i in range(len(signs_df)):
            overview = signs_df['Overview'][i]
            if type(overview) == float:
                continue
            pname = signs_df['Protocols'][i] + ' ({})'.format(signs_df['Protocol ID'][i])
            pname = pname.lower()
            p2overview[pname] = overview
        return p2overview

    # the following mapping relations are generated from protocol guidelines
    @staticmethod
    def p2signs(signs_df):
        # create mapping (protocol --- signs)
        p2s = defaultdict(list)
        # p2s_d = defaultdict(dict)
        s2p = defaultdict(list)
        for i in range(len(signs_df)):
            ss = signs_df['Signs and Symptoms(in impression list)'][i]
            if type(ss) == float:
                continue
            pname = signs_df['Protocols'][i] + ' ({})'.format(signs_df['Protocol ID'][i])
            pname = pname.lower()
            for s in ss.split(';'):
                s = s.strip().capitalize()
                if pname not in s2p[s]:
                    s2p[s].append(pname)
                if s not in p2s[pname]:
                    p2s[pname].append(s)
        return p2s, s2p

    @staticmethod
    def p2impre(impre_df):
        p2impre = defaultdict(list)
        # p2impre_d = defaultdict(dict)
        impre2p = defaultdict(list)
        for i in range(len(impre_df)):
            ps = impre_df['Protocol_name'][i]
            candi_ps = impre_df['Indication/Considerations of Protocols'][i]
            impre = impre_df['Impression'][i].strip().capitalize()
            if type(impre) == float:
                continue
            if type(ps) != float:
                for p in ps.split(';'):
                    p = p.strip().lower()

                    if impre not in p2impre[p]:
                        p2impre[p].append(impre)
                    if p not in impre2p[impre]:
                        impre2p[impre].append(p)
            if type(candi_ps) != float:
                for c_p in candi_ps.split(';'):
                    c_p = c_p.strip().lower()

                    if impre not in p2impre[c_p]:
                        p2impre[c_p].append(impre)
                    if c_p not in impre2p[impre]:
                        impre2p[impre].append(c_p)
        return p2impre, impre2p

    @staticmethod
    def p2med(med_df):
        # create mapping (protocol --- medication)
        p2m = defaultdict(list)
        # p2m_d = defaultdict(dict)
        m2p = defaultdict(list)

        for i in range(len(med_df)):
            med_name = med_df['medication'][i].strip().capitalize()
            med_name_idx = re.search(r'[(]\d+[)]', med_name).start()
            med_name = med_name[:med_name_idx].strip()
            pname = ''
            if type(med_df['protocols'][i]) != float:
                pname += med_df['protocols'][i] + '; '
            if type(med_df['considerations'][i]) != float:
                pname += med_df['considerations'][i]
            if type(pname) != float:
                pname = pname.lower()
            for p in pname.split(';'):
                p = p.lower().strip()
                if p == '':
                    continue
                # p2m_d[p][med_name] = p2m_d[p].get(med_name, 0) + 1
                if med_name not in p2m[p]:
                    p2m[p].append(med_name)
                if p not in m2p[med_name]:
                    m2p[med_name].append(p)
        return p2m, m2p

    @staticmethod
    def p2proc(proc_df):
        # create mapping (protocol --- procedure)
        p2proc = defaultdict(list)
        # p2proc_d = defaultdict(dict)
        proc2p = defaultdict(list)
        proc_filter = ['Assess airway', 'Ecg', 'Iv', 'Move patient', 'Bvm', 'Assess - pain assessment',
                       'Im/sq injection']
        for i in range(len(proc_df)):
            procedure = proc_df['Grouping'][i].strip().capitalize()
            if procedure in proc_filter:
                continue

            if type(proc_df['Protocols'][i]) == float:
                continue
            for p in proc_df['Protocols'][i].split(';'):
                p = p.lower().strip()

                # p2proc_d[p][procedure] = p2proc_d[p].get(procedure, 0) + 1
                if procedure not in p2proc[p]:
                    p2proc[p].append(procedure)
                if p not in proc2p[procedure]:
                    proc2p[procedure].append(p)
        return p2proc, proc2p

    @staticmethod
    def p2treatment(p2proc, proc2p, p2m, m2p):
        p2treat = defaultdict(list)
        treat2p = defaultdict(list)

        for p, proc in p2proc.items():
            p2treat[p].extend(proc)
        for p, m in p2m.items():
            p2treat[p].extend(m)

        for proc, p in proc2p.items():
            treat2p[proc].extend(p)
        for m, p in m2p.items():
            treat2p[m].extend(p)
        return p2treat, treat2p

    @staticmethod
    def protocol2sym():
        p2s_path = '%s/protocol2symptom.json' % DIR
        s2p_path = '%s/symptom2protocol.json' % DIR
        with open(p2s_path, 'r') as f:
            p2s = json.load(f)
        with open(s2p_path, 'r') as f:
            s2p = json.load(f)
        return p2s, s2p
    @staticmethod
    def protocol2treat():
        p2t_path = '%s/protocol2treatment.json' % DIR
        t2p_path = '%s/treatment2protocol.json' % DIR
        with open(p2t_path, 'r') as f:
            p2t = json.load(f)
        with open(t2p_path, 'r') as f:
            t2p = json.load(f)
        return p2t, t2p

    @staticmethod
    def diag2overview(ICD9_description):
        return ICD9_description

    @staticmethod
    def diag2sign(ICD9_description):
        # d2s_path = '%s/diag2sign.json' % DIR
        d2s_path = '%s/icd9code2symptom.json' % DIR
        # d2s_path = '%s/wikipedia/diag2sign.json' % DIR
        with open(d2s_path, 'r') as f:
            d2s_raw = json.load(f)
        d2s = defaultdict(list)
        for k, vs in d2s_raw.items():
            for v in vs:
                # d2s[k].append(ICD9_description[v])
                d2s[k].append(v)

        # s2d_path = '%s/sign2diag.json' % DIR
        s2d_path = '%s/symptom2icd9code.json' % DIR
        # s2d_path = '%s/wikipedia/sign2diag.json' % DIR
        with open(s2d_path, 'r') as f:
            s2d_raw = json.load(f)
        s2d = defaultdict(list)
        for k, vs in s2d_raw.items():
            for v in vs:
                # s2d[ICD9_description[k]].append(v)
                s2d[k].append(v)
        return d2s, s2d

    @staticmethod
    def diag2treament(ICD9_description):
        d2t_path = '%s/icd9code2treatment.json' % DIR
        # d2t_path = '%s/wikipedia/diag2treatment.json' % DIR
        with open(d2t_path, 'r') as f:
            d2t_raw = json.load(f)
        d2t = defaultdict(list)
        for k, vs in d2t_raw.items():
            for v in vs:
                d2t[k].append(v)

        t2d_path = '%s/treatment2icd9code.json' % DIR
        # t2d_path = '%s/wikipedia/treatment2diag.json' % DIR
        with open(t2d_path, 'r') as f:
            t2d_raw = json.load(f)
        t2d = defaultdict(list)
        for k, vs in t2d_raw.items():
            for v in vs:
                t2d[k].append(v)
        return d2t, t2d

    def _genNode(self, p2overview, s2p, t2p):
        for k, v in s2p.items():
            # check if sign-mapped protocols is in p_node
            for i in v:
                if i in self.p_node and k not in self.sign_node:
                    self.sign_node.append(k)
                    break

        for k, v in t2p.items():
            for i in v:
                if i in self.p_node and k not in self.treat_node:
                    self.treat_node.append(k)
                    break


        node = []
        node.extend(self.p_node)
        node.extend(self.sign_node)
        node.extend(self.treat_node)

        pre_trained_embed_root = os.path.join(ROOT, 'pre-trained embedding')
        if not os.path.exists(pre_trained_embed_root):
            os.makedirs(pre_trained_embed_root)
        backbone_name = self.backbone.split('/')[-1]
        dataset_name = 'MIMIC3' if 'MIMIC3' in dataset else 'RAA'
        nodes_attr_cache = os.path.join(pre_trained_embed_root,
                                        '{}_node_embedding_{}.json'.format(dataset_name, backbone_name))

        if not os.path.exists(nodes_attr_cache):
            cache_dict = {}
            nodes_attr = []
            tokenizer = AutoTokenizer.from_pretrained(self.backbone, do_lower_Case=True)
            model = AutoModel.from_pretrained(self.backbone, output_hidden_states=True)
            b_embed = bertEmbedding(tokenizer, model)
            q, w, e = 0, 0, 0
            for i, n in enumerate(tqdm(node, desc='build heterogeneous graph')):
                attr = {}
                if n in self.p_node:
                    n_type = 'protocol'
                    n_feature = b_embed.getPreEmbedding(preprocess(p2overview[n]))
                    self.p2idx[i] = q
                    q += 1
                elif n in self.sign_node:
                    n_type = 'sign'
                    n_feature = b_embed.getPreEmbedding(preprocess(n))
                    self.s2idx[i] = w
                    w += 1
                elif n in self.treat_node:
                    n_type = 'treatment'
                    n_feature = b_embed.getPreEmbedding(preprocess(n))
                    self.t2idx[i] = e
                    e += 1
                else:
                    raise Exception('Node type is incorrect, recheck node type')
                attr[n_type] = n
                attr['node_type'] = n_type
                attr['node_feature'] = n_feature
                nodes_attr.append((i, attr))
                cache_dict[n] = n_feature
            if is_main_process():
                with open(nodes_attr_cache, 'w') as f:
                    json.dump(cache_dict, f, indent=4)
        else:
            with open(nodes_attr_cache, 'r') as f:
                nodes_attr_ = json.load(f)
            nodes_attr = []
            h, q, w, e = 0, 0, 0, 0
            for i, n in enumerate(tqdm(node, desc='build heterogeneous graph')):
                attr = {}
                if n in self.p_node:
                    n_type = 'protocol'
                    self.p2idx[i] = q
                    q += 1
                elif n in self.sign_node:
                    n_type = 'sign'
                    self.s2idx[i] = w
                    w += 1
                elif n in self.treat_node:
                    n_type = 'treatment'
                    self.t2idx[i] = e
                    e += 1
                else:
                    raise Exception('Node type is incorrect, recheck node type')
                n_feature = nodes_attr_[n]
                attr[n_type] = n
                attr['node_type'] = n_type
                attr['node_feature'] = n_feature
                nodes_attr.append((i, attr))
        return node, nodes_attr

    def __checkEdges(self, node, src_node, dic):
        edges = []
        if src_node in dic:
            for i, dst_node in enumerate(dic[src_node]):

                src_node_idx = node.index(src_node)
                dst_node_idx = node.index(dst_node)

                if dst_node in self.sign_node:
                    edge_type = 'has'
                elif dst_node in self.treat_node:
                    edge_type = 'suggests'
                else:
                    raise Exception('recheck edge type')
                edges.append((src_node_idx, dst_node_idx, {'edge_type': edge_type}))
        return edges

    def _genEdge(self, node, p2s, p2t):
        edges = []
        for i, n in enumerate(node):
            edge_s = self.__checkEdges(node, n, p2s)
            if edge_s:
                edges.extend(edge_s)
            edge_t = self.__checkEdges(node, n, p2t)
            if edge_t:
                edges.extend(edge_t)
        return edges

    @staticmethod
    def transIdx(idx, protocol_idx_map, sign_idx_map, treat_idx_map):
        if idx in protocol_idx_map:
            return protocol_idx_map[idx], 'protocol'
        elif idx in sign_idx_map:
            return sign_idx_map[idx], 'sign'
        elif idx in treat_idx_map:
            return treat_idx_map[idx], 'treatment'
        else:
            raise Exception('node indexing is incorrect')

    @staticmethod
    def checkEdgeWeight(src, dst, node, dst_name):
        with open('%s/wikipedia/tf-idf_diag2sign.json' % DIR, 'r') as f:
            tf_idf_diag2sign = json.load(f)
        with open('%s/wikipedia/tf-idf_diag2treatment.json' % DIR, 'r') as f:
            tf_idf_diag2treatment = json.load(f)

        if dst_name == 'sign':
            weight = tf_idf_diag2sign[node[src]][node[dst]]
        elif dst_name == 'treatment':
            weight = tf_idf_diag2treatment[node[src]][node[dst]]
        else:
            raise Exception('check edge type')
        return weight


    def _genGraph(self, node, nodes_attr, edges):
        # convert graph data into class HeteroData
        label_hetero = HeteroData()
        # add node features
        p_feat = []
        s_feat = []
        t_feat = []

        for (index, node_attr) in nodes_attr:
            if node_attr['node_type'] == 'protocol':
                p_feat.append(node_attr['node_feature'])
            elif node_attr['node_type'] == 'sign':
                s_feat.append(node_attr['node_feature'])
            elif node_attr['node_type'] == 'treatment':
                t_feat.append(node_attr['node_feature'])
            else:
                raise Exception('incorrect node type')

        label_hetero['protocol'].x = torch.tensor(p_feat, dtype=torch.float)
        label_hetero['impression'].x = torch.tensor(s_feat, dtype=torch.float)
        label_hetero['treatment'].x = torch.tensor(t_feat, dtype=torch.float)


        phs_src, phs_dst = [], []
        pst_src, pst_dst = [], []

        for (src, dst, _) in edges:
            src_idx, src_name = self.transIdx(src, self.p2idx, self.s2idx, self.t2idx)
            dst_idx, dst_name = self.transIdx(dst, self.p2idx, self.s2idx, self.t2idx)

            if dst_name == 'sign':
                phs_src.append(src_idx)
                phs_dst.append(dst_idx)
            elif dst_name == 'treatment':
                pst_src.append(src_idx)
                pst_dst.append(dst_idx)

            else:
                raise Exception('check transIdx')

        label_hetero['protocol', 'has', 'impression'].edge_index = torch.tensor([phs_src, phs_dst], dtype=torch.int64)
        label_hetero['protocol', 'suggests', 'treatment'].edge_index = torch.tensor([pst_src, pst_dst], dtype=torch.int64)
        label_hetero['impression', 'indicates', 'protocol'].edge_index = torch.tensor([phs_dst, phs_src], dtype=torch.int64)
        label_hetero['treatment', 'is suggested by', 'protocol'].edge_index = torch.tensor([pst_dst, pst_src], dtype=torch.int64)
        return label_hetero

    def forward(self, signs_df=None, impre_df=None, med_df=None, proc_df=None):
        if dataset == 'RAA':
            p2overview = self.p2overview(signs_df)
            p2sign, sign2p = self.protocol2sym()
            p2treatment, treatment2p = self.protocol2treat()
        elif 'MIMIC3' in dataset:
            p2overview = self.diag2overview(self.ICD9_description)
            p2sign, sign2p = self.diag2sign(self.ICD9_description)
            p2treatment, treatment2p = self.diag2treament(self.ICD9_description)
        else:
            raise Exception('check dataset in default_sets.py')

        node, nodes_attr = self._genNode(p2overview, sign2p, treatment2p)
        edges = self._genEdge(node, p2sign, p2treatment)
        graph = self._genGraph(node, nodes_attr, edges)
        return graph


class heteroHierCooccur(nn.Module):
    def __init__(self, config):
        super(heteroHierCooccur, self).__init__()
        if config.train.backbone == 'CNN':
            self.backbone = 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12'
        else:
            self.backbone = config.train.backbone

        self.p_node = label
        if dataset == 'RAA':
            self.parent_node = ['emergency medical service protocol guidelines']
        elif dataset == 'MIMIC3':
            self.parent_node = ['international statistical classification of diseases and related health problems']
        else:
            raise Exception('check dataset in default_sets.py')


        self.child_node = self.p_node
        self.child2idx = {}
        self.parent2idx = {}
        # self.device = device

    @staticmethod
    def p2overview(signs_df):
        # create mapping (protocol --- overview)
        p2overview = defaultdict(list)
        for i in range(len(signs_df)):
            overview = signs_df['Overview'][i]
            if type(overview) == float:
                continue
            pname = signs_df['Protocols'][i] + ' ({})'.format(signs_df['Protocol ID'][i])
            pname = pname.lower()
            p2overview[pname].append(overview)
        return p2overview

    @staticmethod
    def diag2overview(ICD9_description):
        return ICD9_description

    @staticmethod
    def readCooccurMat():
        with open(os.path.join(DIR, 'co_occur_label.json'), 'r') as f:
            sim_mat = json.load(f)
        return np.array(sim_mat)

    def hierarchy_relation(self, raw_p2hier):
        p2hier = {}
        for p in self.p_node:
            hier = raw_p2hier[p]

            if hier not in self.parent_node:
                self.parent_node.append(hier)

            p2hier[p] = hier
        return p2hier

    def _genNode(self, p2overview):
        node = []
        node.extend(self.parent_node)
        node.extend(self.child_node)
        nodes_attr = []

        pre_trained_embed_root = os.path.join(ROOT, 'pre-trained embedding')
        if not os.path.exists(pre_trained_embed_root):
            os.makedirs(pre_trained_embed_root)
        backbone_name = self.backbone.split('/')[-1]
        nodes_attr_cache = os.path.join(pre_trained_embed_root,
                                        '{}_node_embedding_{}_{}.json'.format(dataset, backbone_name, self.mode))

        if not os.path.exists(nodes_attr_cache):
            cache_dict = {}
            nodes_attr = []
            tokenizer = AutoTokenizer.from_pretrained(self.backbone, do_lower_Case=True)
            model = AutoModel.from_pretrained(self.backbone, output_hidden_states=True)
            b_embed = bertEmbedding(tokenizer, model)
            for i, n in enumerate(tqdm(node, desc='Build Hierarchical Graph')):
                attr = {}
                n_features = []
                if n in self.child_node:
                    for view in p2overview[n]:
                        view_processed = preprocess(view)
                        n_features.append(b_embed.getPreEmbedding(view_processed))
                    n_feature = list(np.mean(n_features, axis=0))
                elif n in self.parent_node:
                    n_processed = preprocess(n)
                    n_features.append(b_embed.getPreEmbedding(n_processed))
                    n_feature = list(np.mean(n_features, axis=0))
                else:
                    raise Exception('node can only be in parent node or child node')
                attr['node_feature'] = n_feature
                nodes_attr.append((i, attr))
                cache_dict[n] = n_feature

                if is_main_process():
                    with open(nodes_attr_cache, 'w') as f:
                        json.dump(cache_dict, f, indent=4)
        else:
            with open(nodes_attr_cache, 'r') as f:
                nodes_attr_ = json.load(f)
            for i, n in enumerate(tqdm(node, desc='Build Graph')):
                attr = {}
                n_feature = nodes_attr_[n]
                attr['node_feature'] = n_feature
                nodes_attr.append((i, attr))
        return node, nodes_attr

    def _genEdge(self, node, p2hier, cooccur_mat):
        edges = []
        edge_type = 'hierarchical'
        for i, n in enumerate(node):
            if (i != 0) and (n in self.parent_node):
                src_node_idx = node.index(n) #hierarchy node
                dst_node_idx = 0 #root node
                edges.append((src_node_idx, dst_node_idx, {'edge_type': edge_type}))
            if n in self.child_node:
                hier = p2hier[n]
                src_node_idx = node.index(n)
                dst_node_idx = node.index(hier)
                edges.append((src_node_idx, dst_node_idx, {'edge_type': edge_type}))

        edge_type = 'co-occur'
        for i, n in enumerate(self.child_node):
            for j in range(len(self.child_node)):
                if j == i:
                    continue
                else:
                    if cooccur_mat[i][j] > 0:
                        src_node_idx = node.index(self.child_node[i])
                        dst_node_idx = node.index(self.child_node[j])
                        edges.append((src_node_idx, dst_node_idx, {'edge_type': edge_type}))
                        # edges.append([i, j])
                        # edges_attr.append([cooccur_mat[i][j]])
        return edges

    def _genGraph(self, nodes_attr, edges):
        label_hetero = HeteroData()

        # add node features
        l_feat = []
        for (index, node_attr) in nodes_attr:
            l_feat.append(node_attr['node_feature'])

        hier_src, hier_dst = [], []
        co_src, co_dst = [], []

        for (i, j, e_type) in edges:
            if e_type == 'hierarchical':
                hier_src.append(i)
                hier_dst.append(j)
            elif e_type == 'co-occur':
                co_src.append(i)
                co_dst.append(j)
            else:
                raise Exception('check edge types')
        label_hetero['protocol', 'hierarchical with', 'protocol'].edge_index = torch.tensor([hier_src, hier_dst])
        label_hetero['protocol', 'co-occurs with', 'protocol'].edge_index = torch.tensor([co_src, co_dst])
        return label_hetero

    def forward(self, signs_df, raw_p2hier):
        if dataset == 'RAA':
            p2overview = self.p2overview(signs_df)
        else:
            p2overview = self.diag2overview(self.ICD9_description)

        node, nodes_attr = self._genNode(p2overview)
        p2hier = self.hierarchy_relation(raw_p2hier)
        cooccur_mat = self.readCooccurMat()

        edges = self._genEdge(node, p2hier, cooccur_mat)
        graph = self._genGraph(nodes_attr, edges)
        return graph

class HierarchyGraph(nn.Module):
    def __init__(self, config):
        super(HierarchyGraph, self).__init__()
        # assert configbackbone == 'CNN'

        self.p_node = label
        if dataset == 'RAA':
            self.parent_node = ['emergency medical service protocol guidelines']
        elif 'MIMIC3' in dataset:
            self.parent_node = ['international statistical classification of diseases and related health problems']
        else:
            raise Exception('check dataset in default_sets.py')


        self.child_node = self.p_node
        self.child2idx = {}
        self.parent2idx = {}
        # self.device = device

    @staticmethod
    def p2overview(signs_df):
        # create mapping (protocol --- overview)
        p2overview = defaultdict(list)
        for i in range(len(signs_df)):
            overview = signs_df['Overview'][i]
            if type(overview) == float:
                continue
            pname = signs_df['Protocols'][i] + ' ({})'.format(signs_df['Protocol ID'][i])
            pname = pname.lower()
            p2overview[pname].append(overview)
        return p2overview

    @staticmethod
    def diag2overview(ICD9_description):
        return ICD9_description

    def hierarchy_relation(self, raw_p2hier):
        p2hier = {}
        for p in self.p_node:
            hier = raw_p2hier[p]

            if hier not in self.parent_node:
                self.parent_node.append(hier)

            p2hier[p] = hier
        return p2hier

    def _genNode(self, p2overview):
        node = []
        node.extend(self.parent_node)
        node.extend(self.child_node)
        nodes_attr = []

        pre_trained_embed_root = os.path.join(ROOT, 'pre-trained embedding')
        if not os.path.exists(pre_trained_embed_root):
            os.makedirs(pre_trained_embed_root)
        backbone_name = 'BioWordVec_PubMed_MIMICIII_d200'

        dataset_name = 'MIMIC3' if 'MIMIC3' in dataset else 'RAA'
        nodes_attr_cache = os.path.join(pre_trained_embed_root,
                                        '{}_label_embedding_{}.npy'.format(dataset_name, backbone_name))

        if not os.path.exists(nodes_attr_cache):
            from gensim.models.keyedvectors import KeyedVectors
            import nltk
            from utils import preprocess
            # pretrain_wv = '/home/xueren/Downloads/BioWordVec_PubMed_MIMICIII_d200.vec.bin'
            pretrain_wv = os.path.join(pre_trained_embed_root, 'BioWordVec_PubMed_MIMICIII_d200.vec.bin')
            model = KeyedVectors.load_word2vec_format(pretrain_wv, binary=True)

            cache_dict = {}
            q, w = 0, 0
            for i, n in enumerate(tqdm(node, desc='Build Hierarchical Graph')):
                attr = {}
                n_features = []
                if n in self.child_node:
                    for view in p2overview[n]:
                        view_processed = preprocess(view)
                        tokens = nltk.word_tokenize(view_processed)
                        for t in tokens:
                            if t in model.key_to_index:
                                n_features.append(model[t])
                    n_feature = list(np.mean(n_features, axis=0))
                    self.child2idx[i] = q
                    q += 1
                elif n in self.parent_node:
                    n_processed = preprocess(n)
                    tokens = nltk.word_tokenize(n_processed)
                    for t in tokens:
                        if t in model.key_to_index:
                            n_features.append(model[t])

                    n_feature = list(np.mean(n_features, axis=0))
                    self.parent2idx[i] = w
                    w += 1
                else:
                    raise Exception('node can only be in parent node or child node')
                attr['node_feature'] = n_feature
                nodes_attr.append((i, attr))
                cache_dict[n] = n_feature

                np.save(nodes_attr_cache, cache_dict)
        else:
            nodes_attr_ = np.load(nodes_attr_cache, allow_pickle=True)
            nodes_attr_ = dict(enumerate(nodes_attr_.flatten()))[0]
            q, w = 0, 0
            for i, n in enumerate(tqdm(node, desc='Build Hierarchical Graph')):
                attr = {}
                if n in self.child_node:
                    self.child2idx[i] = q
                    q += 1
                elif n in self.parent_node:
                    self.parent2idx[i] = w
                    w += 1
                else:
                    raise Exception('Node type is incorrect, recheck node type')
                n_feature = nodes_attr_[n]
                attr['node_feature'] = n_feature
                nodes_attr.append((i, attr))

        return node, nodes_attr

    def _genEdge(self, node, p2hier):
        edges = []
        edge_type = 'is children node of'
        for i, n in enumerate(node):
            if (i != 0) and (n in self.parent_node):
                src_node_idx = node.index(n) #hierarchy node
                dst_node_idx = 0 #root node
                edges.append([src_node_idx, dst_node_idx])
            if n in self.child_node:
                hier = p2hier[n]
                src_node_idx = node.index(n)
                dst_node_idx = node.index(hier)
                edges.append([src_node_idx, dst_node_idx])
        return edges

    def _genGraph(self, nodes_attr, edges):
        # add node features
        feat = []
        for (index, node_attr) in nodes_attr:
            feat.append(node_attr['node_feature'])

        feat = torch.tensor(np.array(feat), dtype=torch.float)
        edges = torch.tensor(edges)
        hierarchy_graph = Data(x=feat, edge_index=edges.t().contiguous())
        return hierarchy_graph

    def forward(self, signs_df=None, raw_p2hier=None):
        if dataset == 'RAA':
            p2overview = self.p2overview(signs_df)
        elif 'MIMIC3' in dataset:
            p2overview = self.diag2overview(signs_df)
        else:
            raise Exception('check dataset in default_sets.py')

        p2hier = self.hierarchy_relation(raw_p2hier)
        node, nodes_attr = self._genNode(p2overview)
        edges = self._genEdge(node, p2hier)
        graph = self._genGraph(nodes_attr, edges)
        return graph

class SematicGraph(nn.Module):
    def __init__(self, config):
        super(SematicGraph, self).__init__()
        assert config.train.backbone == 'CNN'
        # self.device = device

    @staticmethod
    def p2overview(signs_df):
        # create mapping (protocol --- overview)
        p2overview = defaultdict(list)
        for i in range(len(signs_df)):
            overview = signs_df['Overview'][i]
            if type(overview) == float:
                continue
            pname = signs_df['Protocols'][i] + ' ({})'.format(signs_df['Protocol ID'][i])
            pname = pname.lower()
            p2overview[pname].append(overview)
        return p2overview

    @staticmethod
    def diag2overview(ICD9_description):
        return ICD9_description

    def _genNode(self, p2overview):

        node = label
        nodes_attr = []

        pre_trained_embed_root = os.path.join(ROOT, 'pre-trained embedding')
        if not os.path.exists(pre_trained_embed_root):
            os.makedirs(pre_trained_embed_root)
        backbone_name = 'BioWordVec_PubMed_MIMICIII_d200'
        nodes_attr_cache = os.path.join(pre_trained_embed_root,
                                        '{}_label_embedding_{}.npy'.format(dataset, backbone_name))

        if not os.path.exists(nodes_attr_cache):
            from gensim.models.keyedvectors import KeyedVectors
            import nltk
            from utils import preprocess
            print("building...")
            pretrain_wv = os.path.join(pre_trained_embed_root, 'BioWordVec_PubMed_MIMICIII_d200.vec.bin')
            model = KeyedVectors.load_word2vec_format(pretrain_wv, binary=True)

            cache_dict = {}
            for i, n in enumerate(tqdm(node, desc='Build Semantic Graph')):
                attr = {}
                n_features = []
                for view in p2overview[n]:
                    view_processed = preprocess(view)
                    tokens = nltk.word_tokenize(view_processed)
                    for t in tokens:
                        if t in model.key_to_index:
                            n_features.append(model[t])

                n_feature = list(np.mean(n_features, axis=0))
                attr['node_feature'] = n_feature
                nodes_attr.append((i, attr))
                cache_dict[n] = n_feature
            np.save(nodes_attr_cache, cache_dict)
        else:
            nodes_attr_ = np.load(nodes_attr_cache, allow_pickle=True)
            nodes_attr_ = dict(enumerate(nodes_attr_.flatten()))[0]
            for i, n in enumerate(tqdm(node, desc='Build Semantic Graph')):
                attr = {}
                n_feature = nodes_attr_[n]
                attr['node_feature'] = n_feature
                nodes_attr.append((i, attr))

        return node, nodes_attr

    @staticmethod
    def readSimMat():
        with open(os.path.join(DIR, 'sim_semantics_label.json'), 'r') as f:
            sim_mat = json.load(f)
        return np.array(sim_mat)

    def _genEdge(self, node, sim_mat):
        edges = []
        edges_attr = []
        edge_type = 'is similar to'
        for i, n in enumerate(node):
            for j in range(len(node)):
                if j == i:
                    continue
                else:
                    edges.append([i, j])
                    edges_attr.append(sim_mat[i][j])
        return edges, edges_attr

    def _genGraph(self, nodes_attr, edges, edge_attr):
        # add node features
        feat = []
        for (index, node_attr) in nodes_attr:
            feat.append(node_attr['node_feature'])

        feat = torch.tensor(np.array(feat), dtype=torch.float)
        edges = torch.tensor(edges)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        sim_graph = Data(x=feat, edge_index=edges.t().contiguous(), edge_attr=edge_attr)
        return sim_graph

    def forward(self, signs_df):
        if dataset == "RAA":
            p2overview = self.p2overview(signs_df)
        elif dataset == 'MIMIC3':
            p2overview = self.diag2overview(signs_df)
        else:
            raise Exception('check dataset in default_sets.py')
        sim_mat = self.readSimMat()

        node, nodes_attr = self._genNode(p2overview)
        edges, edges_attr = self._genEdge(node, sim_mat)
        sim_graph = self._genGraph(nodes_attr, edges, edges_attr)
        return sim_graph

class CooccurGraph(nn.Module):
    def __init__(self, config):
        super(CooccurGraph, self).__init__()
        assert config.train.backbone == 'CNN'
        # self.device = device

    @staticmethod
    def p2overview(signs_df):
        # create mapping (protocol --- overview)
        p2overview = defaultdict(list)
        for i in range(len(signs_df)):
            overview = signs_df['Overview'][i]
            if type(overview) == float:
                continue
            pname = signs_df['Protocols'][i] + ' ({})'.format(signs_df['Protocol ID'][i])
            pname = pname.lower()
            p2overview[pname].append(overview)
        return p2overview

    @staticmethod
    def diag2overview(ICD9_description):
        return ICD9_description

    def _genNode(self, p2overview):
        node = label
        nodes_attr = []

        pre_trained_embed_root = os.path.join(ROOT, 'pre-trained embedding')
        if not os.path.exists(pre_trained_embed_root):
            os.makedirs(pre_trained_embed_root)
        backbone_name = 'BioWordVec_PubMed_MIMICIII_d200'
        nodes_attr_cache = os.path.join(pre_trained_embed_root,
                                        '{}_label_embedding_{}.npy'.format(dataset, backbone_name))

        if not os.path.exists(nodes_attr_cache):
            from gensim.models.keyedvectors import KeyedVectors
            import nltk
            from utils import preprocess
            pretrain_wv = os.path.join(pre_trained_embed_root, 'BioWordVec_PubMed_MIMICIII_d200.vec.bin')
            model = KeyedVectors.load_word2vec_format(pretrain_wv, binary=True)

            cache_dict = {}
            for i, n in enumerate(tqdm(node, desc='Build Co-occur Graph')):
                attr = {}
                n_features = []
                for view in p2overview[n]:
                    view_processed = preprocess(view)
                    tokens = nltk.word_tokenize(view_processed)
                    for t in tokens:
                        if t in model.key_to_index:
                            n_features.append(model[t])


                n_feature = list(np.mean(n_features, axis=0))
                attr['node_feature'] = n_feature
                nodes_attr.append((i, attr))
                cache_dict[n] = n_feature
            np.save(nodes_attr_cache, cache_dict)
        else:
            nodes_attr_ = np.load(nodes_attr_cache, allow_pickle=True)
            nodes_attr_ = dict(enumerate(nodes_attr_.flatten()))[0]
            for i, n in enumerate(tqdm(node, desc='Build Co-occur Graph')):
                attr = {}
                n_feature = nodes_attr_[n]
                attr['node_feature'] = n_feature
                nodes_attr.append((i, attr))

        return node, nodes_attr

    @staticmethod
    def readCooccurMat():
        with open(os.path.join(DIR, 'co_occur_label.json'), 'r') as f:
            sim_mat = json.load(f)
        return np.array(sim_mat)

    def _genEdge(self, node, cooccur_mat):
        edges = []
        edges_attr = []
        edge_type = 'co-occur with'
        for i, n in enumerate(node):
            for j in range(len(node)):
                if j == i:
                    continue
                else:
                    if cooccur_mat[i][j] > 0:
                        edges.append([i, j])
                        edges_attr.append([cooccur_mat[i][j]])
        return edges, edges_attr

    def _genGraph(self, nodes_attr, edges, edge_attr):
        # add node features
        feat = []
        for (index, node_attr) in nodes_attr:
            feat.append(node_attr['node_feature'])

        feat = torch.tensor(np.array(feat), dtype=torch.float)
        edges = torch.tensor(edges)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        cooccur_graph = Data(x=feat, edge_index=edges.t().contiguous(), edge_attr=edge_attr)
        return cooccur_graph

    def forward(self, signs_df):
        if dataset == 'RAA':
            p2overview = self.p2overview(signs_df)
        elif dataset == 'MIMIC3':
            p2overview = self.diag2overview(signs_df)
        else:
            raise Exception('check dataset in default_sets.py')
        cooccur_mat = self.readCooccurMat()

        node, nodes_attr = self._genNode(p2overview)
        edges, edges_attr = self._genEdge(node, cooccur_mat)
        cooccur_graph = self._genGraph(nodes_attr, edges, edges_attr)
        return cooccur_graph

if __name__ == '__main__':
    ### heterogeneous graph
    from default_sets import config

    if config.train.graph == 'HGT':
        HGraph = HeteroGraph(config)
        if dataset == 'RAA':
            signs_df = pd.read_excel(os.path.join(DIR, 'All Protocols Mapping.xlsx'))
            graph = HGraph(signs_df)
        elif 'MIMIC3' in dataset:
            graph = HGraph()
        else:
            raise Exception('check dataset in default_sets.py')
        print('Graph built')
    elif config.train.graph == 'GCN':
        if dataset == 'RAA':
            df = pd.read_excel(os.path.join(DIR, 'All Protocols Mapping.xlsx'))
        else:
            df = ICD9_description
        if config.model == 'KAMG':
            Graph1 = HierarchyGraph(config)
            G1 = Graph1(df, label2hier)
            Graph2 = SematicGraph(config)
            G2 = Graph2(df)
            Graph3 = CooccurGraph(config)
            G3 = Graph3(df)
            graph = [G1, G2, G3]
        else:
            Graph = HierarchyGraph(config)
            graph = Graph(df, label2hier)
        print('Graph built')
    else:
        if config.model == 'CAML' or config.model == 'DR_CAML':
            graph = np.load(os.path.join(ROOT, 'pre-trained embedding/'
                                               '{}_label_embedding_BioWordVec'
                                               '_PubMed_MIMICIII_d200.npy'.format(config.dataset)), allow_pickle=True)
            graph = torch.tensor(graph)
        else:
            graph = None

    print(graph)





