import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pandas as pd
import json
# from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split
from default_sets import *
# if dataset == 'RAA':
#     from default_sets import p_node, ungroup_p_node, group_hier, ungroup_hier,  groupby, EMS_DIR
# elif dataset == 'MIMIC3':
#     from default_sets import ICD9_DIAG, ICD9_DIAG_GROUP, MIMIC_3_DIR
from transformers import BertTokenizer, AutoTokenizer
from utils import sharpen, AttrDict, checkOnehot, removePunctuation, preprocess
import nltk
import pickle
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from logger import is_main_process
from vocab import WordVocab
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler
import math

class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indices sequentially, making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training), which means that the model params won't
    have to be synced (i.e. will not hang for synchronization even if varied number of forward passes), we still add
    extra samples to the sampler to make it evenly divisible (like in `DistributedSampler`) to make it easy to `gather`
    or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert (
            len(indices) == self.total_size
        ), f"Indices length {len(indices)} and total size {self.total_size} mismatched"

        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        assert (
            len(indices) == self.num_samples
        ), f"Indices length {len(indices)} and sample number {self.num_samples} mismatched"

        return iter(indices)

    def __len__(self):
        return self.num_samples



class EMSDataPipeline(nn.Module):
    def __init__(self, config, vocab, rank, world_size):
        super(EMSDataPipeline, self).__init__()
        self.config = config
        self.backbone = config.train.backbone
        self.max_len = config.train.max_len
        self.batch_size = config.train.batch_size
        self.mode = 'ungroup'
        self.vocab = vocab
        self.rank = rank
        self.world_size = world_size

        self.column = 'Ungrouped Protocols'
        self.p_node = label

        self.n_labels = len(self.p_node)
        self.class_freq = [0] * len(self.p_node)
        self.train_label_cnt = {}
        self.val_label_cnt = {}
        self.test_label_cnt = {}

    def __statistics_dataset(self, train, val, test):

        self.train_label_cnt = self.cnt_instance_per_label(train)
        self.val_label_cnt = self.cnt_instance_per_label(val)
        self.test_label_cnt = self.cnt_instance_per_label(test)

        train_p_list = self.cnt_protocol_per_dataset(train, self.column, self.mode)
        val_p_list = self.cnt_protocol_per_dataset(val, self.column, self.mode)
        test_p_list = self.cnt_protocol_per_dataset(test, self.column, self.mode)
        p_dict = {}
        df_protocol = train[self.column]

        for ps in df_protocol:
            for p in ps.split(';'):
                p_dict[p] = p_dict.get(p, 0) + 1
        for i, p in enumerate(self.p_node):
            if p in p_dict:
                self.class_freq[i] = p_dict[p]
            else:
                self.class_freq[i] = 0

        if is_main_process():
            print('the number of protocols in train dataset are {}'.format(len(train_p_list)))
            print('the number of protocols in validation dataset are {}'.format(len(val_p_list)))
            print('the number of protocols in test dataset are {}'.format(len(test_p_list)))
            print('the total number of labels is {}'.format(len(self.p_node)))

    def encode_label(self, df):
        labels = []
        narratives = []
        for idx in range(len(df)):
            real_protocol = df['Ungrouped Protocols'][idx]
            protocol = df[self.column][idx]
            if type(real_protocol) != float:
                label = self.one_hot_encoder(protocol, self.p_node)
            else:
                # change here
                protocol = protocol.strip()
                ps = [float(p) for p in protocol.split(';')]
                label = np.clip(np.array(ps), 0, 1)

            labels.append(label)
            narratives.append(df['Narrative'][idx])

        labels = pd.DataFrame(labels)
        data = pd.concat([pd.DataFrame(narratives, columns=['Narrative']), labels], axis=1)
        return data

    def build_single_dataloader(self, df):
        if df['Protocols'].isnull().all():
            labels = None
        else:
            labels = []
            for idx in range(len(df)):
                if self.mode == 'group':
                    protocol = df[self.column][idx]
                elif self.mode == 'ungroup':
                    protocol = df[self.column][idx]
                else:
                    raise Exception('mode can only be [group, ungroup]')
                label = self.one_hot_encoder(protocol, self.p_node)
                labels.append(label)

        data_set = EMSDataset(df.Narrative.tolist(), labels, self.config, df, self.vocab)
        data_loader = DataLoader(data_set, batch_size=self.batch_size, shuffle=False)
        return data_loader

    def build_single_dataset(self, df):
        if df['Protocols'].isnull().all():
            labels = None
        else:
            labels = []
            for idx in range(len(df)):
                if self.mode == 'group':
                    protocol = df[self.column][idx]
                elif self.mode == 'ungroup':
                    protocol = df[self.column][idx]
                else:
                    raise Exception('mode can only be [group, ungroup]')
                label = self.one_hot_encoder(protocol, self.p_node)
                labels.append(label)
        data_set = EMSDataset(df.Narrative.tolist(), labels, self.config, df, self.vocab)
        return data_set

    def build_dataset(self, train_df, val_df, test_df, train, val, test):
        train_dataset = EMSDataset(train.Narrative.tolist(), train[range(self.n_labels)].values.tolist(),
                                  self.config, train_df, self.vocab)
        valid_dataset = EMSDataset(val.Narrative.tolist(), val[range(self.n_labels)].values.tolist(),
                                  self.config, val_df, self.vocab)
        test_dataset = EMSDataset(test.Narrative.tolist(), test[range(self.n_labels)].values.tolist(),
                                  self.config, test_df, self.vocab)
        return train_dataset, valid_dataset, test_dataset

    def build_dataloader(self, train_dataset, val_dataset, test_dataset):
        if self.config.DDP.enable:
            train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False,
                                           sampler=DistributedSampler(train_dataset))
            val_sampler = SequentialDistributedSampler(val_dataset, batch_size=self.batch_size)
            test_sampler = SequentialDistributedSampler(test_dataset, batch_size=self.batch_size)
            val_data_loader = DataLoader(val_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False, sampler=val_sampler)
            test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False, sampler=test_sampler)
        elif self.config.FSDP.enable:
            train_data_loader = DataLoader(train_dataset,
                                           batch_size=self.batch_size,
                                           pin_memory=True,
                                           sampler=DistributedSampler(train_dataset, rank=self.rank, num_replicas=self.world_size, shuffle=True))
            val_data_loader = DataLoader(val_dataset,
                                         batch_size=self.batch_size,
                                         pin_memory=True,
                                         sampler=DistributedSampler(val_dataset, rank=self.rank, num_replicas=self.world_size))
            test_data_loader = DataLoader(test_dataset,
                                          batch_size=self.batch_size,
                                          pin_memory=True,
                                          sampler=DistributedSampler(test_dataset, rank=self.rank, num_replicas=self.world_size))
        else:
            train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
            val_data_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_data_loader, val_data_loader, test_data_loader

    def forward(self, train_df, val_df, test_df):
        # train_indices, val_indices, test_indices = self.split_dataset(train, val, test)
        train = self.encode_label(train_df)
        val = self.encode_label(val_df)
        test = self.encode_label(test_df)
        self.__statistics_dataset(train_df, val_df, test_df)
        train_dataset, valid_dataset, test_dataset = self.build_dataset(train_df, val_df, test_df, train, val, test)
        train_data_loader, val_data_loader, test_data_loader = self.build_dataloader(train_dataset, valid_dataset, test_dataset)
        return train_data_loader, val_data_loader, test_data_loader

    @staticmethod
    def cnt_instance_per_label(df):
        label_cnt = {}
        # if groupby == 'hierarchy':
        #     column = 'Ungrouped Hierarchy'
        # elif groupby == 'age':
        #     column = 'Ungrouped Protocols'
        # else:
        #     column = 'Ungrouped Protocols'
        column = 'Ungrouped Protocols'
        for i in range(len(df)):
            ### if group by age, column name is 'Ungrouped Protocols'
            ### ig group by hierarchy, column name is 'Hierarchy'
            if type(df[column][i]) == float:
                continue
            ps = df[column][i].strip()
            for p in ps.split(';'):
                p = p.strip()
                label_cnt[p] = label_cnt.get(p, 0) + 1
        return label_cnt

    @staticmethod
    def cnt_protocol_per_dataset(df, column, mode):
        p_list = []
        for i in range(len(df)):
            if type(df['Ungrouped Protocols'][i]) == float:
                continue
            if mode == 'group':
                protocol = df[column][i]
            elif mode =='ungroup':
                protocol = df[column][i]
            else:
                raise Exception('mode can only be [group, ungroup]')
            for p in protocol.split(';'):
                p = p.strip().lower()

                if p not in p_list:
                    p_list.append(p)
        return p_list

    @staticmethod
    def one_hot_encoder(ps, p_list):
        one_hot_encoding = [0] * len(p_list)
        for p in ps.split(';'):
            p = p.strip().lower()
            if p in p_list:
                one_hot_encoding[p_list.index(p)] = 1

        # if label not in the group, mark label as others
        if not np.array(one_hot_encoding).any():
            raise Exception('labels are all zeros')

        return one_hot_encoding

class EMSDataset():
    def __init__(self, texts, labels, config, df, vocab):
        self.config = config
        self.texts = texts
        self.labels = labels
        self.backbone = config.train.backbone
        self.max_len = config.train.max_len
        self.window_size = config.train.window_size
        if self.backbone != 'CNN':
            self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, do_lower_Case=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.vocab = vocab

        self.cahcedStopwords = stopwords.words('english')

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text_ = self.texts[index]
        text = preprocess(text_)
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.float)


        if isinstance(self.window_size, int):
            if self.window_size > 1:
                tokens = text.split()
                length = len(tokens)
                tokens += [self.tokenizer.pad_token] * (self.window_size * self.max_len - length)
                ids_list = []
                mask_list = []
                for i in range(0, self.window_size):
                    chunk = ' '.join(tokens[i * self.max_len:(i + 1) * self.max_len])
                    inputs = self.tokenizer.__call__(chunk,
                                                     None,
                                                     add_special_tokens=True,
                                                     max_length=self.max_len,
                                                     padding="max_length",
                                                     truncation=True,
                                                     )
                    ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
                    mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)
                    ids_list.append(ids)
                    mask_list.append(mask)
                ids_tensor = torch.stack(ids_list, dim=0)
                mask_tensor = torch.stack(mask_list, dim=0)
                return {
                    "ids": ids_tensor,
                    "mask": mask_tensor,
                    "labels": label,
                    'id': self.df['ID'][index]
                }
            else:
                inputs = self.tokenizer.__call__(text,
                                                 None,
                                                 add_special_tokens=True,
                                                 max_length=self.max_len,
                                                 padding="max_length",
                                                 truncation=True,
                                                 )
                ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
                mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)
                return {
                    "ids": ids,
                    "mask": mask,
                    "labels": label,
                    'id': self.df['ID'][index]
                }
        else:
            ids, mask, _ = self.text2inp(text, self.max_len)
            ids = torch.tensor(ids, dtype=torch.long)
            mask = torch.tensor(mask, dtype=torch.long)

            return {
                "ids": ids,
                "mask": mask,
                "labels": label,
                'id': self.df['ID'][index]
            }
    def text2inp(self, text, max_length, flag_special_token=True):
        text = preprocess(text)
        tokens = self.raw_text2tokens(text)
        tokens = [self.vocab.sos_index] + tokens + [self.vocab.eos_index] if flag_special_token else tokens
        tokens = tokens[:max_length]


        length = len(tokens)
        tokens += [self.vocab.pad_index] * (max_length - length)
        mask = [0] * length + [1] * (max_length - length)

        return tokens, mask, length

    def raw_text2tokens(self, text):
        list_word = text.split()
        tokens = list_word.copy()

        for i, token in enumerate(tokens):
            tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

        return tokens


class MIMIC3DataPipeline(nn.Module):
    def __init__(self, config, vocab, rank, world_size):
        super(MIMIC3DataPipeline, self).__init__()
        self.config = config
        self.backbone = config.train.backbone
        self.max_len = config.train.max_len
        self.batch_size = config.train.batch_size
        self.vocab = vocab
        self.rank = rank
        self.world_size = world_size

        self.labels = label
        self.column = 'LABELS'
        self.n_labels = len(self.labels)
        self.class_freq = [0] * len(self.labels)
        self.train_label_cnt = {}
        self.val_label_cnt = {}
        self.test_label_cnt = {}

    def encode_label(self, df):
        labels = []
        texts = []
        for i in range(len(df)):
            diags = df[self.column][i]
            text = df['TEXT'][i]
            label = self.one_hot_encoder(diags, self.labels)
            labels.append(label)
            texts.append(text)
        labels = pd.DataFrame(labels)
        data = pd.concat([pd.DataFrame(texts, columns=['TEXT']), labels], axis=1)
        return data

    def cnt_protocol_per_dataset(self, df):
        p_list = []
        for i in range(len(df)):
            protocol = df[self.column][i]
            for p in protocol.split(';'):
                p = p.strip()
                if p not in p_list:
                    p_list.append(p)
        return p_list

    def __statistics_dataset(self, train, val, test):

        self.train_label_cnt = self.cnt_instance_per_label(train)
        self.val_label_cnt = self.cnt_instance_per_label(val)
        self.test_label_cnt = self.cnt_instance_per_label(test)

        train_p_list = self.cnt_protocol_per_dataset(train)
        val_p_list = self.cnt_protocol_per_dataset(val)
        test_p_list = self.cnt_protocol_per_dataset(test)

        p_dict = {}
        df_protocol = train[self.column]
        for ps in df_protocol:
            for p in ps.split(';'):
                p_dict[p] = p_dict.get(p, 0) + 1
        for i, p in enumerate(self.labels):
            if p in p_dict:
                self.class_freq[i] = p_dict[p]
            else:
                self.class_freq[i] = 0
        if is_main_process():
            print('the number of labels in train dataset are {}'.format(len(train_p_list)))
            print('the number of labels in validation dataset are {}'.format(len(val_p_list)))
            print('the number of labels in test dataset are {}'.format(len(test_p_list)))
            print('the total number of labels is {}'.format(len(self.labels)))
            print(f'train length {len(train)}')
            print(f'val length {len(val)}')
            print(f'test length {len(test)}')

    def build_dataset(self, df, raw_df):
        dataset = MIMIC3Dataset(df.TEXT.tolist(), df[range(self.n_labels)].values.tolist(), self.config, self.vocab)
        return dataset

    def build_dataloader(self, dataset, is_train=False):
        if self.config.DDP.enable:
            data_loader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False,
                                           sampler=DistributedSampler(dataset))
        elif self.config.FSDP.enable:
            shuffle = True if is_train else False
            data_loader = DataLoader(dataset,
                                     batch_size=self.batch_size,
                                     pin_memory=True,
                                     sampler=DistributedSampler(dataset,
                                                                rank=self.rank,
                                                                num_replicas=self.world_size,
                                                                shuffle=shuffle))
        else:
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return data_loader

    def forward(self, train_df, val_df, test_df):
        train = self.encode_label(train_df)
        val = self.encode_label(val_df)
        test = self.encode_label(test_df)
        self.__statistics_dataset(train_df, val_df, test_df)
        train_dataset = self.build_dataset(train, train_df)
        val_dataset = self.build_dataset(val, val_df)
        test_dataset = self.build_dataset(test, test_df)
        train_loader = self.build_dataloader(train_dataset, is_train=True)
        val_loader = self.build_dataloader(val_dataset)
        test_loader = self.build_dataloader(test_dataset)
        return train_loader, val_loader, test_loader


    @staticmethod
    def one_hot_encoder(ps, p_list):
        one_hot_encoding = [0] * len(p_list)
        for p in ps.split(';'):
            p = p.strip()
            if p in p_list:
                one_hot_encoding[p_list.index(p)] = 1

        # if label not in the group, mark label as others
        if not np.array(one_hot_encoding).any():
            print(ps)
            raise Exception('labels are all zeros')
        return one_hot_encoding

    @staticmethod
    def cnt_instance_per_label(df):
        label_cnt = {}
        column = 'LABELS'
        for i in range(len(df)):
            if type(df[column][i]) == float:
                continue
            ps = df[column][i].strip()
            for p in ps.split(';'):
                p = p.strip()
                label_cnt[p] = label_cnt.get(p, 0) + 1
        return label_cnt

class MIMIC3Dataset():
    def __init__(self, texts, labels, config, vocab):
        self.config = config
        self.texts = texts
        self.labels = labels
        self.backbone = config.train.backbone
        self.max_len = config.train.max_len
        # self.window_size = 3 if config.model == 'DKEC' else 1
        self.window_size = config.train.window_size

        if self.backbone != 'CNN' and self.backbone != 'RNN':
            self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, do_lower_Case=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.vocab = vocab
        self.cahcedStopwords = stopwords.words('english')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text_ = self.texts[index]
        text = preprocess(text_)
        label = torch.tensor(self.labels[index], dtype=torch.float)

        if isinstance(self.window_size, int):
            if self.window_size > 1:
                tokens = text.split()
                length = len(tokens)
                tokens += [self.tokenizer.pad_token] * (self.window_size * self.max_len - length)
                ids_list = []
                mask_list = []
                for i in range(0, self.window_size):
                    chunk = ' '.join(tokens[i * self.max_len:(i + 1) * self.max_len])
                    inputs = self.tokenizer.__call__(chunk,
                                                     None,
                                                     add_special_tokens=True,
                                                     max_length=self.max_len,
                                                     padding="max_length",
                                                     truncation=True,
                                                     )
                    ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
                    mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)
                    ids_list.append(ids)
                    mask_list.append(mask)

                ids_tensor = torch.stack(ids_list, dim=0)
                mask_tensor = torch.stack(mask_list, dim=0)

                return {
                    "ids": ids_tensor,
                    "mask": mask_tensor,
                    "labels": label
                }
            else:
                inputs = self.tokenizer.__call__(text,
                                                 None,
                                                 add_special_tokens=True,
                                                 max_length=self.max_len,
                                                 padding="max_length",
                                                 truncation=True,
                                                 )
                ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
                mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)
                return {
                    "ids": ids,
                    "mask": mask,
                    "labels": label
                }
        else:
            ids, mask, _ = self.text2inp(text, self.max_len)
            ids = torch.tensor(ids, dtype=torch.long)
            mask = torch.tensor(mask, dtype=torch.long)
            return {
                "ids": ids,
                "mask": mask,
                "labels": label
            }


    def text2inp(self, text, max_length, flag_special_token=True):
        text = preprocess(text)
        tokens = self.raw_text2tokens(text)
        tokens = [self.vocab.sos_index] + tokens + [self.vocab.eos_index] if flag_special_token else tokens
        tokens = tokens[:max_length]


        length = len(tokens)
        tokens += [self.vocab.pad_index] * (max_length - length)
        mask = [0] * length + [1] * (max_length - length)

        return tokens, mask, length


    def raw_text2tokens(self, text):
        list_word = text.split()
        tokens = list_word.copy()

        for i, token in enumerate(tokens):
            tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

        return tokens


if __name__ == '__main__':
    from trainer import ddp_setup
    ddp_setup()
    vocab = WordVocab.load_vocab(os.path.join(DIR, 'vocab_{}.pkl'.format(dataset)))

    if dataset == 'RAA':
        train = pd.read_excel('./data/multi_label/RAA_GT_synthesis_fold{}.xlsx'.format(0), sheet_name='GT_train', index_col=0)
        val = pd.read_excel('./data/multi_label/RAA_GT_synthesis_fold{}.xlsx'.format(0), sheet_name='GT_val', index_col=0)
        test = pd.read_excel('./data/multi_label/RAA_GT_synthesis_fold{}.xlsx'.format(0), sheet_name='GT_test', index_col=0)
        data_pipeline = EMSDataPipeline(config, vocab)
    elif dataset == 'MIMIC3':
        from default_sets import DIR
        train = pd.read_csv(os.path.join(DIR, 'clean_train.csv'))
        val = pd.read_csv(os.path.join(DIR, 'clean_val.csv'))
        test = pd.read_csv(os.path.join(DIR, 'clean_test.csv'))
        data_pipeline = MIMIC3DataPipeline(config, vocab)
    else:
        raise Exception('check dataset in default_sets.py')

    train_data_loader, val_data_loader, test_data_loader = data_pipeline(train, val, test)
    for i, batch in enumerate(train_data_loader):
        print(batch['ids'])
        print(batch['mask'])
        print(batch['labels'].shape)
        break

