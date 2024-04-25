import copy
import os
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import torch
from torch.utils.data import Dataset
import itertools
import pickle

import random

seed = 42
negatives_per_pos_pair = 5

random.seed(seed)


def encodeSequence(sequence, seqA, aa_keys, max_sequence_length):
    mat = np.array([aa_keys.loc[aa] for aa in sequence])
    padding = np.zeros((max_sequence_length - mat.shape[0], mat.shape[1]))
    mat = np.append(mat, padding, axis=0)
    mat = np.transpose(mat)

    mat2 = np.array([aa_keys.loc[aa] for aa in seqA])
    padding = np.zeros((max_sequence_length - mat2.shape[0], mat2.shape[1]))
    mat2 = np.append(mat2, padding, axis=0)
    mat2 = np.transpose(mat2)

    matstack = np.stack([mat, mat2])
    matstack = torch.from_numpy(matstack)
    return torch.reshape(matstack, (matstack.size(dim=1), 2, matstack.size(dim=2)))


class Refset(Dataset):
    def __init__(self, list):
        self.data = list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TestDataset(Dataset):
    def __init__(self, sequences, AA_keys_path, max_sequence_length):
        self.aa_keys = pd.read_csv(AA_keys_path, index_col='One Letter')
        self.pairs = []
        self.max_sequence_length = max_sequence_length
        for seq, seqA in sequences.to_records(index=False):
            self.pairs.append(encodeSequence(seq, seqA, self.aa_keys, self.max_sequence_length))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class ValDataset(Dataset):
    def __init__(self, encoding_dict, pairs):
        self.encoding_dict = encoding_dict
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s, epitope, label = self.pairs[idx]

        return self.encoding_dict[s], epitope, label

    def save(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as handle:
            return pickle.load(handle)


class TCRDataset(Dataset):
    def __init__(self, data_path, AA_keys_path, max_sequence_length, holdout_percentage=None):
        self.aa_keys = pd.read_csv(AA_keys_path, index_col='One Letter')
        self.max_sequence_length = max_sequence_length
        self.pairs = []

        self.val_balance = holdout_percentage
        self.val_dataset = None

        self.encoding_dict = {}
        self.epitopes = []

        self.generatePairs(data_path)
        self.tensor_size = (self.aa_keys.shape[1], 1, max_sequence_length)

    def generatePairs(self, data_path):
        all_data = pd.read_csv(data_path)
        grouped_data = dict(tuple(all_data.groupby("Peptide")))

        val_sequences = {}
        rem = []

        self.epitopes = list(grouped_data.keys())

        for i, epitope in enumerate(self.epitopes):
            data = grouped_data[epitope][['CDR3b_extended', 'CDR3a_extended']]
            data['CDR3a_extended'].fillna(data['CDR3b_extended'], inplace=True)
            if len(data) < 2:
                continue

            if self.val_balance:
                train = data.sample(frac=1 - self.val_balance, random_state=seed)
            else:
                train = data

            tmp = train['CDR3b_extended'].to_frame()
            tmp['epitope'] = epitope
            rem.append(tmp)

            valid = data.drop(train.index)
            if not valid.empty:
                val_sequences[epitope] = valid

            for seq, seqA in train.to_records(index=False):
                self.encoding_dict[seq] = [encodeSequence(seq, seqA, self.aa_keys, self.max_sequence_length), epitope]

            for a, b in itertools.combinations(train['CDR3b_extended'], 2):
                self.pairs.append((a, b, 1))  # positive pair

        print(len(self.pairs))

        #full = pd.concat(rem, ignore_index=True, copy=True).drop_duplicates().reset_index(drop=True)
        del rem

        neg_pairs = []

        for j, (s1, s2, _) in enumerate(self.pairs):
            if j % 1000 == 0:
                print(f'\r{j}/{len(self.pairs)}', end='')
            # epitope = self.encoding_dict[s1][1]
            #
            # l1 = int(negatives_per_pos_pair // 2)
            # l2 = negatives_per_pos_pair - l1
            #
            # s = full[full.epitope != epitope].sample(n=negatives_per_pos_pair, random_state=seed)['CDR3b_extended'].to_frame()
            # l = [s1]*l1 + [s2]*l2
            # s['seq2'] = l

            neg_pairs.extend([(0,0,0)]*5)

        print('\n', end='')
        self.pairs.extend(neg_pairs)
        print(f'Negative pairs generated {len(neg_pairs)}')

        if self.val_balance:
            val_dict = {}
            val_pairs = []

            val_epitopes = list(val_sequences.keys())
            for epitope in val_sequences:
                remainder = copy.deepcopy(val_epitopes)
                remainder.remove(epitope)
                for seq, seqA in val_sequences[epitope].to_records(index=False):
                    val_dict[seq] = encodeSequence(seq, seqA, self.aa_keys, self.max_sequence_length)
                    val_pairs.append((seq, epitope, 1))  # positive pair

                    for neg_epitope in random.sample(remainder, negatives_per_pos_pair):
                        val_pairs.append((seq, neg_epitope, 0))  # negative pair

            self.val_dataset = ValDataset(val_dict, val_pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s1, s2, label = self.pairs[idx]

        return self.encoding_dict[s1][0], self.encoding_dict[s2][0], label

    def save(self, train_path, val_path):
        if self.val_dataset:
            self.val_dataset.save(val_path)
        self.val_dataset = None
        with open(train_path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as handle:
            return pickle.load(handle)


def parse_vdjdb(filename, q=0):
    """
    Parse files in the VDJdb format.
    q-score defines the quality of the database entry (3 > 2 > 1 > 0).
    """
    vdjdb = pd.read_csv(filename, sep='\t',low_memory=False)
    vdjdb = vdjdb[vdjdb['species']=='HomoSapiens']
    vdjdb = vdjdb[['cdr3.alpha', 'v.alpha',
                   'cdr3.beta', 'v.beta',
                   'vdjdb.score', 'meta.subject.id', 'antigen.epitope']]
    vdjdb = vdjdb[vdjdb['vdjdb.score'] >= q]  # Quality score cut-off
    vdjdb = vdjdb[~vdjdb['v.alpha'].str.contains("/", na=False)]  # Remove ambiguous entries
    vdjdb = vdjdb[~vdjdb['v.beta'].str.contains("/", na=False)]  # Remove ambiguous entries
    vdjdb.drop(columns=['vdjdb.score'], inplace=True)
    vdjdb.rename(columns={'meta.subject.id':'subject'}, inplace=True)
    vdjdb = vdjdb[vdjdb['subject'].astype(bool)]
    vdjdb = vdjdb[~vdjdb.subject.str.contains('mouse', na=False)]
    vdjdb['count'] = [1] * len(vdjdb)
    vdjdb.drop_duplicates(inplace=True)

    paired = vdjdb[['antigen.epitope', 'cdr3.beta', 'cdr3.alpha']].dropna().drop_duplicates()
    paired.rename(columns={'cdr3.alpha': 'CDR3a_extended',
                           'cdr3.beta': 'CDR3b_extended',
                           'antigen.epitope': 'Peptide'},
                  inplace=True)

    return paired


def main():
    data = parse_vdjdb('../data/vdjdb_full.txt')
    data.drop_duplicates(subset='CDR3b_extended', inplace=True)
    print(f'Usable samples: {len(data.index)}')

    train_data = data.sample(frac=0.5, random_state=seed)
    test_data = data.drop(train_data.index)

    train_data.to_csv('../data/training_data/concatenated.csv', index=False)
    test_data.to_csv('../data/test_data/test.csv', index=False)

    training_data = TCRDataset("../data/training_data/concatenated.csv", "../data/AA_keys.csv", 26, 0.10)
    training_data.save("../output/trainContrastive.pickle", "../output/valContrastive.pickle")


if __name__ == "__main__":
    main()