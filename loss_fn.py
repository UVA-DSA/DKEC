#!/usr/bin/env python
# coding:utf-8
"""
Tencent is pleased to support the open source community by making NeuralClassifier available.
Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the MIT License (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at
http://opensource.org/licenses/MIT
Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for thespecific language governing permissions and limitations under
the License.
"""

import torch
import torch.nn as nn

class Type(object):
    @classmethod
    def str(cls):
        raise NotImplementedError


class LossType(Type):
    """Standard names for loss type
    """
    SOFTMAX_CROSS_ENTROPY = "SoftmaxCrossEntropy"
    SOFTMAX_FOCAL_CROSS_ENTROPY = "SoftmaxFocalCrossEntropy"
    SIGMOID_FOCAL_CROSS_ENTROPY = "SigmoidFocalCrossEntropy"
    BCE_WITH_LOGITS = "BCEWithLogitsLoss"

    @classmethod
    def str(cls):
        return ",".join([cls.SOFTMAX_CROSS_ENTROPY,
                         cls.SOFTMAX_FOCAL_CROSS_ENTROPY,
                         cls.SIGMOID_FOCAL_CROSS_ENTROPY,
                         cls.BCE_WITH_LOGITS])


class ActivationType(Type):
    """Standard names for activation type
    """
    SOFTMAX = "Softmax"
    SIGMOID = "Sigmoid"

    @classmethod
    def str(cls):
        return ",".join([cls.SOFTMAX,
                         cls.SIGMOID])


class FocalLoss(nn.Module):
    """Softmax focal loss
    references: Focal Loss for Dense Object Detection
                https://github.com/Hsuxu/FocalLoss-PyTorch
    """

    def __init__(self, label_size, activation_type=ActivationType.SOFTMAX,
                 gamma=2.0, alpha=0.25, epsilon=1.e-9):
        super(FocalLoss, self).__init__()
        self.num_cls = label_size
        self.activation_type = activation_type
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, logits, target):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        if self.activation_type == ActivationType.SOFTMAX:
            idx = target.view(-1, 1).long()
            one_hot_key = torch.zeros(idx.size(0), self.num_cls,
                                      dtype=torch.float,
                                      device=idx.device)
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            logits = torch.softmax(logits, dim=-1)
            loss = -self.alpha * one_hot_key * \
                   torch.pow((1 - logits), self.gamma) * \
                   (logits + self.epsilon).log()
            loss = loss.sum(1)
        elif self.activation_type == ActivationType.SIGMOID:
            multi_hot_key = target
            logits = torch.sigmoid(logits)
            zero_hot_key = 1 - multi_hot_key
            loss = -self.alpha * multi_hot_key * \
                   torch.pow((1 - logits), self.gamma) * \
                   (logits + self.epsilon).log()
            loss += -(1 - self.alpha) * zero_hot_key * \
                    torch.pow(logits, self.gamma) * \
                    (1 - logits + self.epsilon).log()
        else:
            raise TypeError("Unknown activation type: " + self.activation_type
                            + "Supported activation types: " +
                            ActivationType.str())
        return loss.mean()


class ClassificationLoss(torch.nn.Module):
    def __init__(self, label_size, class_weight=None,
                 loss_type=LossType.SOFTMAX_CROSS_ENTROPY):
        super(ClassificationLoss, self).__init__()
        self.label_size = label_size
        self.loss_type = loss_type

        if loss_type == LossType.SOFTMAX_CROSS_ENTROPY:
            self.criterion = torch.nn.CrossEntropyLoss(class_weight)
        elif loss_type == LossType.SOFTMAX_FOCAL_CROSS_ENTROPY:
            self.criterion = FocalLoss(label_size, ActivationType.SOFTMAX)
        elif loss_type == LossType.SIGMOID_FOCAL_CROSS_ENTROPY:
            self.criterion = FocalLoss(label_size, ActivationType.SIGMOID)
        elif loss_type == LossType.BCE_WITH_LOGITS:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            raise TypeError(
                "Unsupported loss type: %s. Supported loss type is: %s" % (
                    loss_type, LossType.str()))

    def forward(self, logits, target,
                use_hierar=False,
                is_multi=False,
                *argvs):
        device = logits.device
        if use_hierar:
            assert self.loss_type in [LossType.BCE_WITH_LOGITS,
                                      LossType.SIGMOID_FOCAL_CROSS_ENTROPY]
            if not is_multi:
                target = torch.eye(self.label_size)[target].to(device)
            hierar_penalty, hierar_paras, hierar_relations = argvs[0:3]
            return self.criterion(logits, target) + \
                   hierar_penalty * self.cal_recursive_regularize(hierar_paras,
                                                                  hierar_relations,
                                                                  device)
        else:
            if is_multi:
                assert self.loss_type in [LossType.BCE_WITH_LOGITS,
                                          LossType.SIGMOID_FOCAL_CROSS_ENTROPY]
            else:
                if self.loss_type not in [LossType.SOFTMAX_CROSS_ENTROPY,
                                          LossType.SOFTMAX_FOCAL_CROSS_ENTROPY]:
                    target = torch.eye(self.label_size)[target].to(device)
            return self.criterion(logits, target)

    def cal_recursive_regularize(self, paras, hierar_relations, device="cpu"):
        """ Only support hierarchical text classification with BCELoss
        references: http://www.cse.ust.hk/~yqsong/papers/2018-WWW-Text-GraphCNN.pdf
                    http://www.cs.cmu.edu/~sgopal1/papers/KDD13.pdf
        """
        recursive_loss = 0.0
        for i in range(len(paras)):
            if i not in hierar_relations:
                continue
            children_ids = hierar_relations[i]
            if not children_ids:
                continue
            children_ids_list = torch.tensor(children_ids, dtype=torch.long).to(
                device)
            children_paras = torch.index_select(paras, 0, children_ids_list)
            parent_para = torch.index_select(paras, 0,
                                             torch.tensor(i).to(device))
            parent_para = parent_para.repeat(children_ids_list.size()[0], 1)
            diff_paras = parent_para - children_paras
            diff_paras = diff_paras.view(diff_paras.size()[0], -1)
            recursive_loss += 1.0 / 2 * torch.norm(diff_paras, p=2) ** 2
        return recursive_loss





# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from default_sets import device
# from utils import checkOnehot
#
# # class SimCLR(nn.Module):
# #     def __init__(self, temperature):
# #         super(SimCLR, self).__init__()
# #         self.temperature = temperature
# #
# #     def batch_sim_loss(self, text_feat, graph_feat, label):
# #         '''
# #         calculate one batch
# #         '''
# #         loss = 0
# #         total_sim = torch.sum(torch.exp(F.cosine_similarity(text_feat, graph_feat, dim=-1) / self.temperature))
# #         indices = (label == 1).nonzero().cpu().numpy()
# #         # calculate positive samples
# #         for k in indices:
# #             pos_sim = torch.exp(F.cosine_similarity(text_feat, graph_feat[k], dim=-1) / self.temperature)
# #             loss += -torch.log(pos_sim / total_sim)
# #         loss = loss / len(indices)
# #         return loss
# #
# #     def forward(self, text_feature, graph_feature, labels):
# #         # loop over batch size
# #         simLoss = 0
# #         batch_size = labels.shape[0]
# #         for i in range(batch_size):
# #             simLoss += self.batch_sim_loss(text_feature[i], graph_feature, labels[i])
# #         simLoss = simLoss / batch_size
# #         return simLoss
#
#
# class SimCLR(nn.Module):
#     def __init__(self, temperature=0.07):
#         '''
#         https://arxiv.org/pdf/2002.05709.pdf
#         '''
#         super(SimCLR, self).__init__()
#         self.temperature = temperature
#
#     def batch_sim_loss(self, text_feat, labels):
#         '''
#         calculate one batch
#         '''
#         num_class = labels.shape[1]
#         hidden_dim = text_feat.shape[2]
#
#         loss = torch.tensor(0, dtype=torch.float).to(device)
#         for i in range(num_class):
#             pos_set = []
#             neg_set = []
#             pos_cand, other_instance_neg = None, None
#             pos_cand = text_feat[torch.where(labels[:, i] == 1)].view(-1, hidden_dim)
#             other_instance_neg = text_feat[torch.where(labels[:, i] != 1)].view(-1, hidden_dim)
#             for j in range(pos_cand.shape[0]):
#                 if j % num_class == i:
#                     pos_set.append(pos_cand[j])
#                 else:
#                     neg_set.append(pos_cand[j])
#             if len(pos_set) > 1:
#                 pos_set_full = torch.stack(pos_set, dim=0)
#                 neg_set_ = torch.stack(neg_set, dim=0)
#                 neg_set_full = torch.cat((neg_set_, other_instance_neg), dim=0)
#
#                 row = pos_set_full.shape[0]
#                 pos_mat = (torch.ones(row, row) - torch.eye(row, row)).to(device)
#                 pos_mat_ = torch.zeros_like(pos_mat)
#                 neg_mat = torch.zeros(row).to(device)
#                 for k in range(row):
#                     pos_mat_[k] = torch.exp(F.cosine_similarity(pos_set[k], pos_set_full) / self.temperature) * pos_mat[k]
#                     neg_mat[k] = torch.sum(torch.exp(F.cosine_similarity(pos_set[k], neg_set_full) / self.temperature))
#
#                 sim_score = pos_mat_ / neg_mat
#                 sim_score_ = sim_score[sim_score.bool()]
#                 loss = loss + (-1 / (row - 1)) * torch.sum(torch.log(sim_score_))
#         return loss
#
#     def forward(self, text_feature, labels):
#         '''
#
#         :param text_feature: (b, class_num, hidden_size)
#         :param graph_feature: (b, class_num, hidden_size)
#         :return:
#         '''
#         # b = text_feature.shape[0]
#         # num_class = graph_feature.shape[0]
#         # text_feature = text_feature.view(b, -1) #(b, num_class * hidden_size)
#         # graph_feature = graph_feature.view(num_class, -1) # (num_class, num_class * hidden_size)
#         simLoss = self.batch_sim_loss(text_feature, labels)
#         return simLoss.to(device)
#
# # A customized version
# # with ref https://github.com/wutong16/DistributionBalancedLoss/blob/a3ecaa9021a920fcce9fdafbd7d83b51bf526af8/mllt/models/losses/resample_loss.py
# class ResampleLoss(nn.Module):
#     def __init__(self,
#                  use_sigmoid=True, partial=False,
#                  loss_weight=1.0, reduction='mean',
#                  reweight_func=None,  # None, 'inv', 'sqrt_inv', 'rebalance', 'CB'
#                  weight_norm=None,  # None, 'by_instance', 'by_batch'
#                  focal=dict(
#                      focal=True,
#                      alpha=0.5,
#                      gamma=2
#                  ),
#                  map_param=dict(
#                      alpha=10.0,
#                      beta=0.2,
#                      gamma=0.1
#                  ),
#                  CB_loss=dict(
#                      CB_beta=0.9,
#                      CB_mode='average_w'  # 'by_class', 'average_n', 'average_w', 'min_n'
#                  ),
#                  logit_reg=dict(
#                      neg_scale=5.0,
#                      init_bias=0.1
#                  ),
#                  class_freq=None,
#                  train_num=None):
#         super(ResampleLoss, self).__init__()
#
#         assert (use_sigmoid is True) or (partial is False)
#         self.use_sigmoid = use_sigmoid
#         self.partial = partial
#         self.loss_weight = loss_weight
#         self.reduction = reduction
#
#         if self.use_sigmoid:
#             if self.partial:
#                 self.cls_criterion = partial_cross_entropy
#             else:
#                 self.cls_criterion = binary_cross_entropy
#         else:
#             self.cls_criterion = cross_entropy
#
#         # reweighting function
#         self.reweight_func = reweight_func
#
#         # normalization (optional)
#         self.weight_norm = weight_norm
#
#         # focal loss params
#         self.focal = focal['focal']
#         self.gamma = focal['gamma']
#         self.alpha = focal['alpha']  # change to alpha
#
#         # mapping function params
#         self.map_alpha = map_param['alpha']
#         self.map_beta = map_param['beta']
#         self.map_gamma = map_param['gamma']
#
#         # CB loss params (optional)
#         self.CB_beta = CB_loss['CB_beta']
#         self.CB_mode = CB_loss['CB_mode']
#
#         self.class_freq = torch.from_numpy(np.asarray(class_freq)).float().cuda()
#         self.num_classes = self.class_freq.shape[0]
#         self.train_num = train_num  # only used to be divided by class_freq
#         # regularization params
#         self.logit_reg = logit_reg
#         self.neg_scale = logit_reg[
#             'neg_scale'] if 'neg_scale' in logit_reg else 1.0
#         init_bias = logit_reg['init_bias'] if 'init_bias' in logit_reg else 0.0
#         self.init_bias = - torch.log(
#             self.train_num / self.class_freq - 1) * init_bias  ########################## bug fixed https://github.com/wutong16/DistributionBalancedLoss/issues/8
#
#         self.freq_inv = torch.ones(self.class_freq.shape).cuda() / self.class_freq
#         self.propotion_inv = self.train_num / self.class_freq
#
#     def forward(self,
#                 cls_score,
#                 label,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None,
#                 **kwargs):
#
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#
#         weight = self.reweight_functions(label)
#
#         cls_score, weight = self.logit_reg_functions(label.float(), cls_score, weight)
#
#         ## not sure if it's correct
#         if self.focal:
#             logpt = self.cls_criterion(cls_score.clone(), label, weight=None, reduction='none', avg_factor=avg_factor)
#             # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
#             pt = torch.exp(-logpt)
#             wtloss = self.cls_criterion(cls_score, label.float(), weight=weight, reduction='none')
#             alpha_t = torch.where(label == 1, self.alpha, 1 - self.alpha)
#             loss = alpha_t * ((1 - pt) ** self.gamma) * wtloss  ####################### balance_param should be a tensor
#             loss = reduce_loss(loss, reduction)  ############################ add reduction
#         else:
#             loss = self.cls_criterion(cls_score, label.float(), weight,
#                                       reduction=reduction)
#
#         loss = self.loss_weight * loss
#         return loss
#
#     def reweight_functions(self, label):
#         if self.reweight_func is None:
#             return None
#         elif self.reweight_func in ['inv', 'sqrt_inv']:
#             weight = self.RW_weight(label.float())
#         elif self.reweight_func in 'rebalance':
#             weight = self.rebalance_weight(label.float())
#         elif self.reweight_func in 'CB':
#             weight = self.CB_weight(label.float())
#         else:
#             return None
#
#         if self.weight_norm is not None:
#             if 'by_instance' in self.weight_norm:
#                 max_by_instance, _ = torch.max(weight, dim=-1, keepdim=True)
#                 weight = weight / max_by_instance
#             elif 'by_batch' in self.weight_norm:
#                 weight = weight / torch.max(weight)
#
#         return weight
#
#     def logit_reg_functions(self, labels, logits, weight=None):
#         if not self.logit_reg:
#             return logits, weight
#         if 'init_bias' in self.logit_reg:
#             logits += self.init_bias
#         if 'neg_scale' in self.logit_reg:
#             logits = logits * (1 - labels) * self.neg_scale + logits * labels
#             if weight is not None:
#                 weight = weight / self.neg_scale * (1 - labels) + weight * labels
#         return logits, weight
#
#     def rebalance_weight(self, gt_labels):
#         repeat_rate = torch.sum(gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
#         pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
#         # pos and neg are equally treated
#         weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
#         return weight
#
#     def CB_weight(self, gt_labels):
#         if 'by_class' in self.CB_mode:
#             weight = torch.tensor((1 - self.CB_beta)).cuda() / \
#                      (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
#         elif 'average_n' in self.CB_mode:
#             avg_n = torch.sum(gt_labels * self.class_freq, dim=1, keepdim=True) / \
#                     torch.sum(gt_labels, dim=1, keepdim=True)
#             weight = torch.tensor((1 - self.CB_beta)).cuda() / \
#                      (1 - torch.pow(self.CB_beta, avg_n)).cuda()
#         elif 'average_w' in self.CB_mode:
#             weight_ = torch.tensor((1 - self.CB_beta)).cuda() / \
#                       (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
#             weight = torch.sum(gt_labels * weight_, dim=1, keepdim=True) / \
#                      torch.sum(gt_labels, dim=1, keepdim=True)
#         elif 'min_n' in self.CB_mode:
#             min_n, _ = torch.min(gt_labels * self.class_freq +
#                                  (1 - gt_labels) * 100000, dim=1, keepdim=True)
#             weight = torch.tensor((1 - self.CB_beta)).cuda() / \
#                      (1 - torch.pow(self.CB_beta, min_n)).cuda()
#         else:
#             raise NameError
#         return weight
#
#     def RW_weight(self, gt_labels, by_class=True):
#         if 'sqrt' in self.reweight_func:
#             weight = torch.sqrt(self.propotion_inv)
#         else:
#             weight = self.propotion_inv
#         if not by_class:
#             sum_ = torch.sum(weight * gt_labels, dim=1, keepdim=True)
#             weight = sum_ / torch.sum(gt_labels, dim=1, keepdim=True)
#         return weight
#
#
# def reduce_loss(loss, reduction):
#     """Reduce loss as specified.
#     Args:
#         loss (Tensor): Elementwise loss tensor.
#         reduction (str): Options are "none", "mean" and "sum".
#     Return:
#         Tensor: Reduced loss tensor.
#     """
#     reduction_enum = F._Reduction.get_enum(reduction)
#     # none: 0, elementwise_mean:1, sum: 2
#     if reduction_enum == 0:
#         return loss
#     elif reduction_enum == 1:
#         return loss.mean()
#     elif reduction_enum == 2:
#         return loss.sum()
#
#
# def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
#     """Apply element-wise weight and reduce loss.
#     Args:
#         loss (Tensor): Element-wise loss.
#         weight (Tensor): Element-wise weights.
#         reduction (str): Same as built-in losses of PyTorch.
#         avg_factor (float): Avarage factor when computing the mean of losses.
#     Returns:
#         Tensor: Processed loss values.
#     """
#     # if weight is specified, apply element-wise weight
#     if weight is not None:
#         loss = loss * weight
#
#     # if avg_factor is not specified, just reduce the loss
#     if avg_factor is None:
#         loss = reduce_loss(loss, reduction)
#     else:
#         # if reduction is mean, then average the loss by avg_factor
#         if reduction == 'mean':
#             loss = loss.sum() / avg_factor
#         # if reduction is 'none', then do nothing, otherwise raise an error
#         elif reduction != 'none':
#             raise ValueError('avg_factor can not be used with reduction="sum"')
#     return loss
#
#
# def binary_cross_entropy(pred,
#                          label,
#                          weight=None,
#                          reduction='mean',
#                          avg_factor=None):
#     # weighted element-wise losses
#     if weight is not None:
#         weight = weight.float()
#
#     # loss = F.binary_cross_entropy_with_logits(pred, label.float(), weight, reduction='none')
#     # loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)
#
#
#     b = pred.shape[0]
#     loss = 0
#     for i in range(b):
#         if checkOnehot(label[i]):
#             lamb = 1.0
#         else:
#             lamb = 0.3
#         loss += lamb * F.binary_cross_entropy_with_logits(pred[i], label[i].float(), weight, reduction=reduction)
#
#     loss = loss / b
#     return loss
#
#
# def reweight(cls_num_list, beta=0.9999):
#     '''
#     Implement reweighting by effective numbers
#     :param cls_num_list: a list containing # of samples of each class
#     :param beta: hyper-parameter for reweighting, see paper for more details
#     :return:
#     '''
#     per_cls_weights = None
#     #############################################################################
#     # TODO: reweight each class by effective numbers                            #
#     #############################################################################
#     per_cls_weights = torch.Tensor([(1 - beta) / (1 - pow(beta, num)) for num in cls_num_list]).cuda()
#     #############################################################################
#     #                              END OF YOUR CODE                             #
#     #############################################################################
#     return per_cls_weights
#
#
# class FocalLoss(nn.Module):
#     def __init__(self, class_freq=None, beta=0.9, gamma=0.5):
#         super(FocalLoss, self).__init__()
#         assert gamma >= 0
#         self.gamma = gamma
#         self.weight = reweight(class_freq, beta=beta)
#
#     def forward(self, input, target):
#         '''
#         Implement forward of focal loss
#         :param input: input predictions
#         :param target: labels
#         :return: tensor of focal loss in scalar
#         '''
#         loss = None
#         #############################################################################
#         # TODO: Implement forward pass of the focal loss                            #
#         #############################################################################
#         log_prob = F.log_softmax(input, dim=-1)
#         prob = torch.exp(log_prob)
#         _, target = torch.max(target, dim=1)
#         focal_weight = torch.clamp((1 - prob) ** self.gamma, min=1e-8, max=1.0-1e-8).float()
#         loss = F.nll_loss(focal_weight * log_prob, target, weight=self.weight)
#
#         #############################################################################
#         #                              END OF YOUR CODE                             #
#         #############################################################################
#         return loss
#
# def loss_fn_selector(loss_func_name, train_num, class_freq):
#     loss_func = None
#
#     if loss_func_name == 'CE':
#         loss_func = nn.CrossEntropyLoss()
#
#     if loss_func_name == 'BCE':
#         loss_func = ResampleLoss(reweight_func=None, loss_weight=1.0,
#                                  focal=dict(focal=False, alpha=0.5, gamma=2),
#                                  logit_reg=dict(),
#                                  class_freq=class_freq, train_num=train_num)
#
#     if loss_func_name == 'FL':
#         # loss_func = ResampleLoss(reweight_func=None, loss_weight=1.0,
#         #                          focal=dict(focal=True, alpha=0.5, gamma=2),
#         #                          logit_reg=dict(),
#         #                          class_freq=class_freq, train_num=train_num)
#         loss_func = FocalLoss(class_freq, beta=0.99, gamma=2)
#
#     if loss_func_name == 'CBloss':  # CB
#         loss_func = ResampleLoss(reweight_func='CB', loss_weight=5.0,
#                                  focal=dict(focal=True, alpha=0.5, gamma=2),
#                                  logit_reg=dict(),
#                                  CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
#                                  class_freq=class_freq, train_num=train_num)
#
#     if loss_func_name == 'R-BCE-Focal':  # R-FL
#         loss_func = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
#                                  focal=dict(focal=True, alpha=0.5, gamma=2),
#                                  logit_reg=dict(),
#                                  map_param=dict(alpha=0.1, beta=10.0, gamma=0.05),
#                                  class_freq=class_freq, train_num=train_num)
#
#     if loss_func_name == 'NTR-Focal':  # NTR-FL
#         loss_func = ResampleLoss(reweight_func=None, loss_weight=0.5,
#                                  focal=dict(focal=True, alpha=0.5, gamma=2),
#                                  logit_reg=dict(init_bias=0.05, neg_scale=2.0),
#                                  class_freq=class_freq, train_num=train_num)
#
#     if loss_func_name == 'DBloss-noFocal':  # DB-0FL
#         loss_func = ResampleLoss(reweight_func='rebalance', loss_weight=0.5,
#                                  focal=dict(focal=False, alpha=0.5, gamma=2),
#                                  logit_reg=dict(init_bias=0.05, neg_scale=2.0),
#                                  map_param=dict(alpha=0.1, beta=10.0, gamma=0.05),
#                                  class_freq=class_freq, train_num=train_num)
#
#     if loss_func_name == 'CBloss-ntr':  # CB-NTR
#         loss_func = ResampleLoss(reweight_func='CB', loss_weight=10.0,
#                                  focal=dict(focal=True, alpha=0.5, gamma=2),
#                                  logit_reg=dict(init_bias=0.05, neg_scale=2.0),
#                                  CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
#                                  class_freq=class_freq, train_num=train_num)
#
#     if loss_func_name == 'DBloss':  # DB
#         loss_func = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
#                                  focal=dict(focal=True, alpha=0.5, gamma=2),
#                                  logit_reg=dict(init_bias=0.05, neg_scale=2.0),
#                                  map_param=dict(alpha=0.1, beta=10.0, gamma=0.05),
#                                  class_freq=class_freq, train_num=train_num)
#     return loss_func