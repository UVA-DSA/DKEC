from torch_geometric.nn import HGTConv, GATConv, GCNConv, Linear, HeteroConv, SAGEConv, GraphConv
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform
from utils import preprocess
import nltk
from torch.autograd import Variable
import torch
import numpy as np
import math
from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification
from default_sets import *
import psutil

def pick_model(conf, vocab, rank):
    """
        Use args to initialize the appropriate model
    """
    if conf.model == "ZAGCNN":
        model = ZAGCNN(conf, vocab)
    elif conf.model == "CAML" or conf.model == 'DR_CAML':
        model = CAML(conf, vocab)
    elif conf.model == "KAMG":
        model = KAMG_ACNN(conf, vocab)
    elif conf.model == "DKEC":
        model = DKEC(conf, vocab, rank)
    elif conf.model == "MultiResCNN":
        model = MultiResCNN(conf, vocab)
    elif conf.model == 'BERT':
        model = BERTModel(conf, vocab)
    elif conf.model == 'BERT_LA':
        model = BERTLA(conf, vocab)
    elif conf.model == 'CNN':
        model = CNN(conf, vocab)
    else:
        raise Exception('check model name in the config')
    return model


class Label_wise_attention(torch.nn.Module):
    def __init__(self, text_feat_dim, label_feat_dim):
        super().__init__()
        self.linear = nn.Linear(text_feat_dim, label_feat_dim)
        self.tanh = nn.Tanh()

    def forward(self, last_hidden_state, label_feat):
        weights = self.tanh(self.linear(last_hidden_state))  # (batch_size, seq_length, hidden_size)
        b, _, _ = last_hidden_state.shape
        label_feat = label_feat.unsqueeze(0).repeat(b, 1, 1)  # (batch_size, class_num, hidden_size)
        weights = torch.bmm(label_feat, weights.permute(0, 2, 1))
        weights = nn.Softmax(dim=1)(weights)  # (batch_size, class_num, seq_length)
        attn_output = torch.bmm(weights, last_hidden_state)  # (batch_size, class_num, hidden_size)
        return weights, attn_output

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(-1, hidden_channels, heads=num_heads, dropout=0.3)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=0.3)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_weight
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        if x.shape[0] > len(label):
            return x[len(hier2label)+1:]
        return x

class HGT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, num_layers, use_bf16):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        self.use_bf16 = use_bf16
        node_types = ['hierarchy', 'protocol', 'impression', 'treatment']
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(in_channels, hidden_channels)

        self.convs = torch.nn.ModuleList()
        meta_data = (['hierarchy', 'protocol', 'impression', 'treatment'],
                     [('protocol', 'is children of', 'hierarchy'),
                      ('protocol', 'has', 'impression'),
                      ('protocol', 'suggests', 'treatment'),
                      ('hierarchy', 'is parent of', 'protocol'),
                      ('impression', 'indicates', 'protocol'),
                      ('treatment', 'is suggested by', 'protocol')]
                     )
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, meta_data, num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, G):
        if self.use_bf16:
            x_dict = {
                node_type: self.lin_dict[node_type](x.to(torch.bfloat16)).relu_()
                for node_type, x in G.x_dict.items()
            }
        else:
            x_dict = {
                node_type: self.lin_dict[node_type](x).relu_()
                for node_type, x in G.x_dict.items()
            }

        for conv in self.convs:
            x_dict = conv(x_dict, G.edge_index_dict)

        # graph_feat = {
        #     'protocols': self.lin(x_dict['protocol']),
        #     'impressions': self.lin(x_dict['impression']),
        #     'treatments': self.lin(x_dict['treatment']),
        # }
        return self.lin(x_dict['protocol'])
        # return graph_feat['protocols']

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        metadata = (['protocol', 'impression', 'treatment'],
                     [('protocol', 'has', 'impression'),
                      ('protocol', 'suggests', 'treatment'),
                      ('impression', 'indicates', 'protocol'),
                      ('treatment', 'is suggested by', 'protocol')]
                     )

        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: GraphConv((-1, -1), hidden_channels)
                # edge_type: GCN(-1, hidden_channels)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, G):

        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in G.x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, G.edge_index_dict, edge_weight_dict=G.edge_weight_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        return self.lin(x_dict['protocol'])

class CAML(nn.Module):
    def __init__(self, config, vocab):
        super(CAML, self).__init__()
        self.embed = nn.Embedding(len(vocab), config.CAML.text_embed_dim)
        self.dropout_embed = nn.Dropout(p=0.2)
        self.vocab = vocab
        self.lmbda = config.CAML.lmbda
        self.label = [ICD9_description[l] for l in label] if dataset == 'MIMIC3' else label
        self.textEncoder = nn.Conv1d(in_channels=config.CAML.text_embed_dim,
                                     out_channels=config.CAML.num_kernels,
                                     kernel_size=config.CAML.kernel_size,
                                     padding=config.CAML.kernel_size // 2)
        xavier_uniform(self.textEncoder.weight)
        self.label_wise_attn = Label_wise_attention(text_feat_dim=config.CAML.num_kernels,
                                                    label_feat_dim=config.CAML.label_embed_dim)


        #context vectors for computing attention as in 2.2
        self.U = nn.Linear(config.CAML.num_kernels, len(self.label))
        xavier_uniform(self.U.weight)

        # final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = nn.Linear(config.CAML.num_kernels, len(self.label))
        xavier_uniform(self.final.weight)


        if self.lmbda > 0:
            W = self.embed.weight.data
            self.desc_embedding = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.desc_embedding.weight.data = W.clone()

            self.label_conv = nn.Conv1d(config.CAML.text_embed_dim,
                                        config.CAML.num_kernels,
                                        kernel_size=config.CAML.kernel_size,
                                        padding=config.CAML.kernel_size // 2)
            xavier_uniform(self.label_conv.weight)

            self.label_fc1 = nn.Linear(config.CAML.num_kernels, config.CAML.num_kernels)
            xavier_uniform(self.label_fc1.weight)

    def embed_descriptions(self, desc_data):
        #label description embedding via convolutional layer
        #number of labels is inconsistent across instances, so have to iterate over the batch
        b_batch = []
        for inst in desc_data:

            label_ids = []
            for each in inst:
                each = preprocess(each)
                tokens = nltk.word_tokenize(each)

                for i, token in enumerate(tokens):
                    label_ids.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            if label_ids:
                d = Variable(torch.cuda.LongTensor(label_ids))
                d = self.desc_embedding(d).transpose(0, 1)
                # d = inst.transpose(1, 2)
                d = self.label_conv(d)
                d = F.max_pool1d(F.tanh(d), kernel_size=d.size()[1])
                d = d.squeeze(1)
                b_inst = self.label_fc1(d)
                b_batch.append(b_inst)
            else:
                b_batch.append([])
        return b_batch

    def _compare_label_embeddings(self, target, b_batch):
        #description regularization loss
        #b is the embedding from description conv
        #iterate over batch because each instance has different # labels
        diffs = []
        for i, bi in enumerate(b_batch):
            ti = target[i]
            inds = torch.nonzero(ti.data).squeeze().cpu().numpy()

            if not isinstance(bi, list):
                zi = self.final.weight[inds, :]
                diff = (zi - bi).mul(zi - bi).mean()

                #multiply by number of labels to make sure overall mean is balanced with regard to number of labels
                diffs.append(self.lmbda*diff*bi.size()[0])
        return diffs


    def _get_loss(self, yhat, target, diffs=None):
        #calculate the BCE
        loss = F.binary_cross_entropy_with_logits(yhat, target)

        #add description regularization loss if relevant
        if self.lmbda > 0 and diffs is not None:
            diff = torch.stack(diffs).mean()
            loss = loss + diff
        return loss

    def forward(self, ids, mask, G, target=None):
        text_embedding = self.dropout_embed(self.embed(ids)).transpose(1, 2)  # (batch size, hidden_size, seq_len)
        text_feats = torch.relu_(self.textEncoder(text_embedding)).transpose(1, 2)

        alpha = F.softmax(self.U.weight.matmul(text_feats.transpose(1, 2)), dim=2)
        attn_text_feats = alpha.matmul(text_feats)

        yhat = self.final.weight.mul(attn_text_feats).sum(dim=2).add(self.final.bias)

        if self.lmbda > 0:
            #run descriptions through description module
            desc_data = []
            for b in range(target.shape[0]):
                ind = torch.where(target[b]==1)[0].cpu().numpy()
                desc_data.append(np.array(self.label)[ind].tolist())

            b_batch = self.embed_descriptions(desc_data)
            #get l2 similarity loss
            diffs = self._compare_label_embeddings(target, b_batch)

        else:
            diffs = None

        loss = self._get_loss(yhat, target, diffs)
        return yhat, loss

class ZAGCNN(nn.Module):
    def __init__(self, config, vocab):
        super(ZAGCNN, self).__init__()
        # fname = '%s/Word2Vec.npy' % DIR if dataset == 'MIMIC3' else '%s/Word2Vec.npy' % DIR
        # word2vec = np.load(fname)
        # weights = torch.FloatTensor(word2vec)
        # self.embed = nn.Embedding.from_pretrained(weights)

        self.embed = nn.Embedding(len(vocab), config.ZAGCNN.text_embed_dim)
        self.dropout_embed = nn.Dropout(p=0.2)
        self.textEncoder = nn.Conv1d(in_channels=config.ZAGCNN.text_embed_dim,
                                     out_channels=config.ZAGCNN.num_kernels,
                                     kernel_size=config.ZAGCNN.kernel_size,
                                     padding=config.ZAGCNN.kernel_size // 2)
        self.graph_model = GCN(config.ZAGCNN.label_embed_dim,
                               config.ZAGCNN.gcn_hidden_features,
                               config.ZAGCNN.gcn_out_features)
        self.label_wise_attn = Label_wise_attention(text_feat_dim=config.ZAGCNN.num_kernels,
                                                    label_feat_dim=config.ZAGCNN.label_embed_dim)

        self.text_out_transform = torch.nn.Sequential(
            nn.Linear(
                in_features=config.ZAGCNN.num_kernels,
                out_features=config.ZAGCNN.gcn_in_features + config.ZAGCNN.gcn_out_features
            ),
            nn.ReLU()
        )

    def forward(self, ids, mask, G, target):
        # start = 10 if dataset == "RAA" else 19
        start = len(hier2label.keys()) + 1
        initial_graph_feat = G.x[start:]
        graph_feat = self.graph_model(G)
        text_embedding = self.dropout_embed(self.embed(ids)).transpose(1, 2)  # (batch size, hidden_size, seq_len)
        text_feats = torch.relu_(self.textEncoder(text_embedding)).transpose(1, 2)

        weights, attn_text_feats = self.label_wise_attn(text_feats, initial_graph_feat)
        all_graph_feat = torch.cat((initial_graph_feat, graph_feat), dim=-1)
        attn_text_feats_transform = self.text_out_transform(attn_text_feats)

        b = text_feats.shape[0]
        all_graph_feat = all_graph_feat.unsqueeze(0).repeat(b, 1, 1)
        out = torch.sum(attn_text_feats_transform * all_graph_feat, dim=-1)

        return out

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
            nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(outchannel)
        )

        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                        nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(outchannel)
                    )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out

class MultiResCNNOutputLayer(nn.Module):
    def __init__(self, Y, input_size):
        super(MultiResCNNOutputLayer, self).__init__()
        self.U = nn.Linear(input_size, Y)
        xavier_uniform(self.U.weight)
        self.final = nn.Linear(input_size, Y)
        xavier_uniform(self.final.weight)

    def forward(self, x):
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        m = alpha.matmul(x)
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        return y

class MultiResCNN(nn.Module):
    def __init__(self, conf, vocab):
        super(MultiResCNN, self).__init__()
        self.word_rep = nn.Embedding(len(vocab), config.MultiResCNN.text_embed_dim)
        self.dropout_embed = nn.Dropout(p=conf.MultiResCNN.embed_dropout)
        self.conv = nn.ModuleList()
        filter_sizes = conf.MultiResCNN.kernel_size

        self.filter_num = len(filter_sizes)
        self.conv_dict = {1: [conf.MultiResCNN.text_embed_dim, conf.MultiResCNN.num_filter_maps],
                          2: [conf.MultiResCNN.text_embed_dim, 100, conf.MultiResCNN.num_filter_maps],
                          3: [conf.MultiResCNN.text_embed_dim, 150, 100, conf.MultiResCNN.num_filter_maps],
                          4: [conf.MultiResCNN.text_embed_dim, 200, 150, 100, conf.MultiResCNN.num_filter_maps]
                          }


        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(conf.MultiResCNN.text_embed_dim,
                            conf.MultiResCNN.text_embed_dim,
                            kernel_size=filter_size,
                            padding=filter_size // 2)
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            conv_dimension = self.conv_dict[conf.MultiResCNN.conv_layer]
            for idx in range(conf.MultiResCNN.conv_layer):
                tmp = ResidualBlock(conv_dimension[idx],
                                    conv_dimension[idx + 1],
                                    filter_size, 1, True,
                                    conf.MultiResCNN.residual_dropout)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        self.U = nn.Linear(self.filter_num * conf.MultiResCNN.num_filter_maps, len(label))
        self.output_layer = MultiResCNNOutputLayer(len(label), self.filter_num * conf.MultiResCNN.num_filter_maps)

    def forward(self, ids, mask, G, target):
        x = self.dropout_embed(self.word_rep(ids)).transpose(1, 2)
        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)
        y = self.output_layer(x)
        return y

class KAMG_ACNN(nn.Module):
    def __init__(self, config, vocab):
        super(KAMG_ACNN, self).__init__()
        self.embed = nn.Embedding(len(vocab), config.KAMG.text_embed_dim)
        self.dropout_embed = nn.Dropout(p=0.2)
        self.textEncoder = nn.Conv1d(in_channels=config.KAMG.text_embed_dim,
                                     out_channels=config.KAMG.num_kernels,
                                     kernel_size=config.KAMG.kernel_size,
                                     padding=config.KAMG.kernel_size // 2)
        self.graph_model = GCN(config.KAMG.label_embed_dim,
                               config.KAMG.gcn_hidden_features,
                               config.KAMG.gcn_out_features)
        self.fusion_linear = torch.nn.Linear(in_features=config.fusion.in_features,
                                             out_features=config.fusion.out_features)

        self.label_wise_attn = Label_wise_attention(text_feat_dim=config.KAMG.num_kernels,
                                                    label_feat_dim=config.KAMG.label_embed_dim)

        self.text_out_transform = torch.nn.Sequential(
            nn.Linear(
                in_features=config.KAMG.num_kernels,
                out_features=config.KAMG.gcn_in_features + config.KAMG.gcn_out_features
            ),
            nn.ReLU()
        )

    def forward(self, ids, mask, G, target):
        start = 10 if dataset == "RAA" else 19
        initial_graph_feat = G[0].x[start:]

        graph_feat1 = self.graph_model(G[0])
        graph_feat2 = self.graph_model(G[1])
        graph_feat3 = self.graph_model(G[2])
        graph_feat = torch.cat((graph_feat1, graph_feat2, graph_feat3), dim=-1)
        graph_feat = self.fusion_linear(graph_feat)


        text_embedding = self.dropout_embed(self.embed(ids)).transpose(1, 2)  # (batch size, hidden_size, seq_len)
        text_embedding = self.dropout_embed(self.embed(ids)).transpose(1, 2)  # (batch size, hidden_size, seq_len)
        text_feats = torch.relu_(self.textEncoder(text_embedding)).transpose(1, 2)

        weights, attn_text_feats = self.label_wise_attn(text_feats, initial_graph_feat)
        all_graph_feat = torch.cat((initial_graph_feat, graph_feat), dim=-1)
        attn_text_feats_transform = self.text_out_transform(attn_text_feats)

        b = text_feats.shape[0]
        all_graph_feat = all_graph_feat.unsqueeze(0).repeat(b, 1, 1)
        out = torch.sum(attn_text_feats_transform * all_graph_feat, dim=-1)
        return out

class DKEC(nn.Module):
    def __init__(self, config, vocab, rank):
        super(DKEC, self).__init__()
        self.backbone = config.train.backbone
        self.window_size = config.train.window_size

        if self.backbone in ['stanford-crfm/BioMedLM', 'UFNLP/gatortron-medium']:
            self.hidden_size = 2560
        elif self.backbone == 'UFNLP/gatortron-large':
            self.hidden_size = 3584
        elif self.backbone in ['UFNLP/gatortron-base', 'microsoft/biogpt']:
            self.hidden_size = 1024
        elif self.backbone in ['google/mobilebert-uncased', 'nlpie/clinical-mobilebert', 'nlpie/bio-mobilebert']:
            self.hidden_size = 512
        elif self.backbone == 'nlpie/tiny-clinicalbert':
            self.hidden_size = 312
        elif self.backbone == 'CNN' or 'RNN':
            self.hidden_size = None
        else:
            self.hidden_size = 768

        self.label = label
        self.rank = rank
        self.num_class = len(self.label)
        self.seq_length = config.train.max_len
        self.graph_type = config.train.graph
        self.reduced_dim = config.train.reduced_dim

        if self.backbone == 'CNN':
            self.dropout_embed = nn.Dropout(p=0.2)
            self.embed = nn.Embedding(len(vocab), config.DKEC.text_embed_dim)

            self.list_kernel_size = config.DKEC.kernel_size
            self.num_kernel = config.DKEC.num_kernels
            self.textEncoder = nn.ModuleList([
                nn.Conv1d(config.DKEC.text_embed_dim, self.num_kernel,
                          kernel_size=kernel_size, padding=kernel_size // 2) for kernel_size in self.list_kernel_size
            ])

            self.text_out_transform = torch.nn.Sequential(
                nn.Linear(
                    in_features=self.num_kernel,
                    out_features=config.DKEC.label_embed_dim * 2
                ),
                nn.ReLU()
            )

            self.label_wise_attn = Label_wise_attention(text_feat_dim=self.num_kernel,
                                                        label_feat_dim=config.DKEC.label_embed_dim)
            # self.fc = nn.Linear(self.num_class, self.num_class)

            self.fc = nn.Linear(config.DKEC.label_embed_dim * 2, self.num_class)
            xavier_uniform(self.fc.weight)

        elif self.backbone == 'RNN':
            self.text_embed_dim = config.rnn.text_embed_dim
            self.bidirectional = config.rnn.bidirectional
            self.n_layers = config.rnn.n_layers
            self.hidden_size = config.rnn.hidden_size

            self.dropout_embed = nn.Dropout(p=0.2)
            self.embed = nn.Embedding(len(vocab), self.text_embed_dim)
            self.bidirectional = bool(self.bidirectional)
            self.n_directions = int(self.bidirectional) + 1

            self.textEncoder = nn.LSTM(self.text_embed_dim,
                                       self.hidden_size,
                                       num_layers=self.n_layers,
                                       bidirectional=self.bidirectional,
                                       dropout=0.2 if self.n_layers > 1 else 0)

            self.label_wise_attn = Label_wise_attention(text_feat_dim=self.n_layers * self.hidden_size,
                                                        label_feat_dim=config.rnn.label_embed_dim)


            self.text_out_transform = torch.nn.Sequential(
                nn.Linear(
                    in_features=self.n_directions * self.hidden_size,
                    out_features=config.rnn.label_embed_dim * 2
                ),
                nn.ReLU()
            )
            self.fc = nn.Linear(config.rnn.label_embed_dim * 2, self.num_class)
            xavier_uniform(self.fc.weight)
        else:
            self.textEncoder = AutoModel.from_pretrained(self.backbone,
                                                         output_hidden_states=True)
            self.label_wise_attn = Label_wise_attention(text_feat_dim=self.hidden_size,
                                                        label_feat_dim=self.hidden_size)
            #### sum_pool linear ####
            if self.reduced_dim:
                self.kernel_size = self.hidden_size // self.reduced_dim
                self.reduced_hidden_dim = math.floor((self.hidden_size - self.kernel_size) / self.kernel_size + 1)
                self.fc = nn.Linear(self.num_class * self.reduced_hidden_dim, self.num_class)

                # self.fc = nn.Linear(self.hidden_size, self.num_class)
                # xavier_uniform(self.fc.weight)

                # self.fc = nn.Linear(self.num_class, self.num_class)
            #### flatten linear ####
            else:
                self.fc = nn.Linear(self.num_class * self.hidden_size, self.num_class)


        if self.graph_type:
            use_bf16 = config.FSDP.enable and config.FSDP.mixed_precision and not config.FSDP.use_fp16
            num_heads = 8

            if self.backbone == 'CNN':
                num_channel = config.DKEC.label_embed_dim
            elif self.backbone == 'RNN':
                num_channel = config.rnn.label_embed_dim
            else:
                num_channel = self.hidden_size

            self.graph_model = HGT(in_channels=num_channel,
                                   hidden_channels=num_channel,
                                   out_channels=num_channel,
                                   num_heads=num_heads,
                                   num_layers=config.train.graph_layer,
                                   use_bf16=use_bf16)


    def init_hidden(self,
                    batch_size, rank):
        """
        Initialise the hidden layer
        :param batch_size: int
            The batch size
        :return: Variable
            The initialised hidden layer
        """
        # [(n_layers x n_directions) x batch_size x hidden_size]
        h = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(rank)
        c = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(rank)
        return h, c

    def forward(self, ids, mask, G, target):
        graph_feat = None
        out = None
        # aggregate information from heterogeneous graph
        initial_graph_feat = G.x_dict['protocol']
        if self.graph_type:
            graph_feat = self.graph_model(G)  # class_num * hidden_size(768)

        if ids != None:
            if self.backbone == 'CNN':
                text_embedding = self.dropout_embed(self.embed(ids)).transpose(1, 2)  # (batch size, hidden_size, seq_len)
                last_hidden_states = torch.cat([torch.relu_(conv(text_embedding)).transpose(1, 2) for conv in self.textEncoder], dim=1)
                out = last_hidden_states

            elif self.backbone == 'RNN':
                hidden = self.init_hidden(ids.shape[0], self.rank)
                text_embedding = self.dropout_embed(self.embed(ids)).permute(1, 0, 2)
                out, hidden = self.textEncoder(text_embedding, hidden)
                out = out.permute(1, 0, 2)
            else:
                # transformer-based models
                if self.window_size > 1:
                    out = []
                    for i in range(ids.shape[1]):
                        output = self.textEncoder(input_ids=ids[:, i, :], attention_mask=mask[:, i, :])
                        tmp_out = output[0] if self.graph_type else output[0][:, 0, :]
                        out.append(tmp_out)
                    out = torch.cat(out, dim=1)

                else:
                    output = self.textEncoder(input_ids=ids, attention_mask=mask)
                    out = output[0]

        if self.graph_type:
            if self.backbone == 'CNN' or self.backbone == 'RNN':
                weights, attn_text_feats = self.label_wise_attn(out, initial_graph_feat)
            else:
                weights, attn_text_feats = self.label_wise_attn(out, graph_feat)
            out = attn_text_feats

        if self.backbone == 'CNN' or self.backbone == 'RNN':
            all_graph_feat = torch.cat((initial_graph_feat, graph_feat), dim=-1)
            attn_text_feats_transform = self.text_out_transform(out)
            b = out.shape[0]
            all_graph_feat = all_graph_feat.unsqueeze(0).repeat(b, 1, 1)

            out = self.fc.weight.mul(attn_text_feats_transform * all_graph_feat).sum(dim=2).add(self.fc.bias)
        else:
            if self.reduced_dim:
                #### max_pool/avg_pool -> flatten -> linear ####
                out = F.max_pool1d(out, kernel_size=self.kernel_size)
                out = out.reshape(out.shape[0], -1)
                out = self.fc(out)

                # # ### sumpool -> fc ###
                # out = self.fc.weight.mul(out).sum(dim=2).add(self.fc.bias)

                ### torchsum -> fc ###
                # out = torch.sum(out, dim=-1)
                # out = self.fc(out)

            else:
                #### flatten -> linear ####
                out = out.reshape(out.shape[0], -1)
                out = self.fc(out)
        return out

class BERTLA(nn.Module):
    def __init__(self, config, vocab):
        super(BERTLA, self).__init__()
        self.backbone = config.train.backbone
        if self.backbone == 'stanford-crfm/BioMedLM':
            self.hidden_size = 2560
        elif self.backbone in ['UFNLP/gatortron-base', 'microsoft/biogpt']:
            self.hidden_size = 1024
        elif self.backbone in ['google/mobilebert-uncased', 'nlpie/clinical-mobilebert', 'nlpie/bio-mobilebert']:
            self.hidden_size = 512
        elif self.backbone == 'nlpie/tiny-clinicalbert':
            self.hidden_size = 312
        elif self.backbone == 'CNN':
            self.hidden_size = 200
        else:
            self.hidden_size = 768

        self.window_size = config.train.window_size
        self.textEncoder = AutoModel.from_pretrained(self.backbone, output_hidden_states=True)
        self.graph_model = GCN(config.BERTLA.label_embed_dim,
                               config.BERTLA.gcn_hidden_features,
                               config.BERTLA.gcn_out_features)
        self.label_wise_attn = Label_wise_attention(text_feat_dim=self.hidden_size,
                                                    label_feat_dim=config.BERTLA.label_embed_dim)

        self.text_out_transform = torch.nn.Sequential(
            nn.Linear(
                in_features=self.hidden_size,
                out_features=config.BERTLA.gcn_in_features + config.BERTLA.gcn_out_features
            ),
            nn.ReLU()
        )


    def forward(self, ids, mask, G, target):
        start = len(hier2label.keys()) + 1
        initial_graph_feat = G.x[start:]
        graph_feat = self.graph_model(G)

        # transformer-based models
        if self.window_size > 1:
            out = []
            for i in range(ids.shape[1]):
                output = self.textEncoder(input_ids=ids[:, i, :], attention_mask=mask[:, i, :])
                tmp_out = output[0]
                out.append(tmp_out)
            text_feats = torch.cat(out, dim=1)
        else:
            output = self.textEncoder(input_ids=ids, attention_mask=mask)
            text_feats = output[0]

        weights, attn_text_feats = self.label_wise_attn(text_feats, initial_graph_feat)
        all_graph_feat = torch.cat((initial_graph_feat, graph_feat), dim=-1)

        attn_text_feats_transform = self.text_out_transform(attn_text_feats)

        b = text_feats.shape[0]
        all_graph_feat = all_graph_feat.unsqueeze(0).repeat(b, 1, 1)
        out = torch.sum(attn_text_feats_transform * all_graph_feat, dim=-1)
        return out

class BERTModel(nn.Module):
    def __init__(self, config, vocab):
        super(BERTModel, self).__init__()
        self.backbone = config.train.backbone
        if self.backbone == 'stanford-crfm/BioMedLM':
            self.hidden_size = 2560
        elif self.backbone in ['UFNLP/gatortron-base', 'microsoft/biogpt']:
            self.hidden_size = 1024
        elif self.backbone in ['google/mobilebert-uncased', 'nlpie/clinical-mobilebert', 'nlpie/bio-mobilebert']:
            self.hidden_size = 512
        elif self.backbone == 'nlpie/tiny-clinicalbert':
            self.hidden_size = 312
        elif self.backbone == 'CNN':
            self.hidden_size = 200
        else:
            self.hidden_size = 768

        self.label = label
        self.num_class = len(self.label)
        self.seq_length = config.train.max_len
        self.textEncoder = AutoModel.from_pretrained(self.backbone, output_hidden_states=True)
        self.fc = nn.Linear(self.hidden_size, self.num_class)

    def forward(self, ids, mask, G, target):
        output = self.textEncoder(input_ids=ids, attention_mask=mask)
        if self.backbone in ['stanford-crfm/BioMedLM', 'microsoft/biogpt']:
            last_hidden_states = output[0]
            out = last_hidden_states
        elif self.backbone in ['distilbert-base-uncased', 'nlpie/clinical-distilbert']:
            last_hidden_states = output[0]
            pooled_output = last_hidden_states[:, 0]
            out = pooled_output
        else:
            last_hidden_states, pooled_output = output[0], output[1]
            out = pooled_output

        out_fine_grain = self.fc(out)
        if self.backbone in ['stanford-crfm/BioMedLM', 'microsoft/biogpt']:
            b = out_fine_grain.shape[0]
            out_fine_grain = out_fine_grain[torch.arange(b, device=out_fine_grain.device), -1]

        return out_fine_grain

class CNN(nn.Module):
    def __init__(self, config, vocab):
        super(CNN, self).__init__()

        self.num_class = len(label)
        self.dropout_embed = nn.Dropout(p=0.2)
        self.embed = nn.Embedding(len(vocab), config.CNN.text_embed_dim)

        self.list_kernel_size = config.CNN.kernel_size
        self.num_kernel = config.CNN.num_kernels
        self.textEncoder = nn.ModuleList([
            nn.Conv1d(config.CNN.text_embed_dim, self.num_kernel,
                      kernel_size=kernel_size, padding=kernel_size // 2) for kernel_size in self.list_kernel_size
        ])

        self.fc = nn.Linear(self.num_kernel * len(self.list_kernel_size), self.num_class)


    def forward(self, ids, mask, G, target):
        text_embedding = self.dropout_embed(self.embed(ids)).transpose(1, 2)  # (batch size, hidden_size, seq_len)
        last_hidden_states = [torch.relu_(conv(text_embedding))for conv in self.textEncoder]
        x = [F.max_pool1d(c, c.size(-1)).squeeze(dim=-1) for c in last_hidden_states]
        x = torch.cat(x, dim=1)
        x = self.fc(x)
        return x


