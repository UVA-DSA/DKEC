import pickle
import tqdm
from collections import Counter


class TorchVocab(object):
    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<oov>'],
                 vectors=None, unk_init=None, vectors_cache=None):

        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)

        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        super().__init__(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"],
                         max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentece, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    def from_seq(self, seq, join=False, with_pad=False):
        pass

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


# Building Vocab with text files
class WordVocab(Vocab):
    def __init__(self, src, tgt, max_size=None, min_freq=1):
        print("Building Vocab")
        counter = Counter()
        for line in tqdm.tqdm(src):
            text = line.replace('\n', '').replace('\t', ' ').replace('##', ' ')
            words = text.split()
            for word in words:
                counter[word] += 1

        for line in tqdm.tqdm(tgt):
            text = line.replace('\n', '').replace('\t', ' ').replace('##', ' ')
            words = text.split()
            for word in words:
                counter[word] += 1

        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            sentence = sentence.split()

        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        if with_eos:
            seq += [self.eos_index]  # this would be index 1
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.itos[idx]
                 if idx < len(self.itos)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)



def read_mimic_train_test():
    import pandas as pd
    import os
    from tqdm import tqdm
    DIR = '/home/xueren/Downloads/physionet.org/files/mimiciii/1.4'
    train = pd.read_csv(os.path.join(DIR, 'disch_train_split.csv'))
    val = pd.read_csv(os.path.join(DIR, 'disch_dev_split.csv'))
    test = pd.read_csv(os.path.join(DIR, 'disch_test_split.csv'))

    train_l = []
    val_l = []
    test_l = []
    for i in tqdm(range(len(train)), desc='train'):
        t = train['TEXT'][i]
        ls = [x.strip() for x in train['LABELS'][i].split(';')]
        tag_ls = '##'.join(ls)
        s = t + '\t' + tag_ls
        train_l.append(s)

    for i in tqdm(range(len(val)), desc='val'):
        t = val['TEXT'][i]
        ls = [x.strip() for x in val['LABELS'][i].split(';')]
        tag_ls = '##'.join(ls)
        s = t + '\t' + tag_ls
        val_l.append(s)

    for i in tqdm(range(len(test)), desc='test'):
        t = test['TEXT'][i]
        ls = [x.strip() for x in test['LABELS'][i].split(';')]
        tag_ls = '##'.join(ls)
        s = t + '\t' + tag_ls
        test_l.append(s)

    with open(os.path.join('../data', 'train_mimic_iii.txt'), 'w') as fp:
        for item in train_l:
            # write each item on a new line
            fp.write("%s\n" % item)

    with open(os.path.join('../data', 'eval_mimic_iii.txt'), 'w') as fp:
        for item in val_l:
            # write each item on a new line
            fp.write("%s\n" % item)

    with open(os.path.join('../data', 'test_mimic_iii.txt'), 'w') as fp:
        for item in test_l:
            # write each item on a new line
            fp.write("%s\n" % item)


def read_raa_train_test():
    import pandas as pd
    import os
    from tqdm import tqdm
    DIR = '/home/xueren/Desktop/NACCL/dataset/RAA'
    train = pd.read_excel(os.path.join(DIR, 'train.xlsx'), index_col=0)
    val = pd.read_excel(os.path.join(DIR, 'val.xlsx'), index_col=0)
    test = pd.read_excel(os.path.join(DIR, 'test.xlsx'), index_col=0)

    train_l = []
    val_l = []
    test_l = []
    for i in tqdm(range(len(train)), desc='train'):
        t = train['Narrative'][i]
        ls = [x.strip() for x in train['Ungrouped Protocols'][i].split(';')]
        tag_ls = '##'.join(ls)
        s = t + '\t' + tag_ls
        train_l.append(s)

    for i in tqdm(range(len(val)), desc='val'):
        t = val['Narrative'][i]
        ls = [x.strip() for x in val['Ungrouped Protocols'][i].split(';')]
        tag_ls = '##'.join(ls)
        s = t + '\t' + tag_ls
        val_l.append(s)

    for i in tqdm(range(len(test)), desc='test'):
        t = test['Narrative'][i]
        ls = [x.strip() for x in test['Ungrouped Protocols'][i].split(';')]
        tag_ls = '##'.join(ls)
        s = t + '\t' + tag_ls
        test_l.append(s)

    with open(os.path.join('../data', 'train_raa.txt'), 'w') as fp:
        for item in train_l:
            # write each item on a new line
            fp.write("%s\n" % item)

    with open(os.path.join('../data', 'eval_raa.txt'), 'w') as fp:
        for item in val_l:
            # write each item on a new line
            fp.write("%s\n" % item)

    with open(os.path.join('../data', 'test_raa.txt'), 'w') as fp:
        for item in test_l:
            # write each item on a new line
            fp.write("%s\n" % item)

def build():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-src", "--corpus_path_src", type=str,
                        default='../data/train_raa.txt')

    parser.add_argument("-tgt", "--corpus_path_tgt", type=str,
                        default='../data/test_raa.txt')

    parser.add_argument("-o", "--output_path", type=str,
                        default='../data/vocab_raa.pkl')

    parser.add_argument("-s", "--vocab_size", type=int, default=80000)

    parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    parser.add_argument("-m", "--min_freq", type=int, default=2)

    args = parser.parse_args()
    f_src = open(args.corpus_path_src, 'r')
    f_tgt = open(args.corpus_path_tgt, 'r')
    vocab = WordVocab(f_src, f_tgt, max_size=args.vocab_size, min_freq=args.min_freq)
    f_src.close()
    f_tgt.close()

    print("VOCAB SIZE:", len(vocab))
    vocab.save_vocab(args.output_path)

#
# build()
# read_raa_train_test()


