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
    def __init__(self, data_path, val_peptides, AA_keys_path, max_sequence_length, holdout_percentage=None):
        self.aa_keys = pd.read_csv(AA_keys_path, index_col='One Letter')
        self.max_sequence_length = max_sequence_length
        self.pairs = []

        self.val_balance = holdout_percentage
        self.val_dataset = None

        self.encoding_dict = {}
        self.epitopes = []

        self.generatePairs(data_path, val_peptides)
        self.tensor_size = (self.aa_keys.shape[1], 1, max_sequence_length)

    def generatePairs(self, data_path, val_peptides):
        all_data = pd.read_csv(data_path)
        grouped_data = dict(tuple(all_data.groupby("Peptide")))

        val_sequences = {}
        train_sequences = {}

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

            train_sequences[epitope] = train

            valid = data.drop(train.index)
            if not valid.empty:
                val_sequences[epitope] = valid

            for seq, seqA in train.to_records(index=False):
                self.encoding_dict[seq] = [encodeSequence(seq, seqA, self.aa_keys, self.max_sequence_length), epitope]

            for a, b in itertools.combinations(train['CDR3b_extended'], 2):
                self.pairs.append((a, b, 1))  # positive pair

        print(len(self.pairs))

        rem = []
        for epitope in list(train_sequences.keys()):
            idk = train_sequences[epitope]['CDR3b_extended'].to_frame()
            idk['epitope'] = epitope
            rem.append(idk)

        full = pd.concat(rem, ignore_index=True, copy=True).drop_duplicates().reset_index(drop=True)
        fullmix = full.merge(full, how='cross')
        n = (len(self.pairs) * negatives_per_pos_pair)
        q = fullmix.query("(epitope_x != epitope_y) and (CDR3b_extended_x > CDR3b_extended_y)").sample(n=n,
                                                                                                       random_state=seed)
        d = q[['CDR3b_extended_x', 'CDR3b_extended_y']]
        d['label'] = 0
        self.pairs.extend(d.to_records(index=False))

        # for epitope in list(train_sequences.keys()):
        #     rem = []
        #     for epitope2 in list(train_sequences.keys()):
        #         if epitope2 is epitope:
        #             continue
        #         rem.append(train_sequences[epitope2]['CDR3b_extended'].copy())

        #     a = train_sequences[epitope]['CDR3b_extended'].to_frame()
        #     b = pd.concat(rem, ignore_index=True, copy=True).drop_duplicates().reset_index(drop=True).to_frame()

        #     c = a.merge(b, how='cross')
        #     c['label'] = 0
        #     d = c.sample(n=25000, random_state=seed, replace=True).drop_duplicates().to_records(index=False)

        #     self.pairs.extend(d)

        print(len(self.pairs))

        if self.val_balance:
            val_dict = {}
            val_pairs = []

            val_epitopes = [e for e in self.epitopes if e in val_peptides]
            for epitope in val_epitopes:
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


def main():
    test_peptides = list(pd.read_csv('../data/test_data/test.csv')['Peptide'].unique())

    t1 = pd.read_csv('../data/training_data/VDJdb_paired_chain.csv')[['Peptide', 'CDR3b_extended', 'CDR3a_extended']]

    t2 = pd.read_csv('../data/training_data/McPAS-TCR_search.csv')[['Epitope.peptide', 'CDR3.beta.aa', 'CDR3.alpha.aa']]
    t2.columns = ['Peptide', 'CDR3b_extended', 'CDR3a_extended']

    frames = [t1, t2]
    d_path = '../data/training_data/immrep22/'
    for f in os.listdir(d_path):
        data = pd.read_csv(d_path + f, sep='\t')
        data = data.loc[data['Label'] == 1]
        epitope = f[:-4]
        data['Peptide'] = epitope
        data = data[['Peptide', 'TRB_CDR3', 'TRA_CDR3']]
        data.columns = ['Peptide', 'CDR3b_extended', 'CDR3a_extended']
        frames.append(data)

    training_data = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    training_data.to_csv('../data/training_data/concatenated.csv', index=False)

    training_data = TCRDataset("../data/training_data/concatenated.csv", test_peptides, "../data/AA_keys.csv", 25, 0.20)
    training_data.save("../output/train.pickle", "../output/val.pickle")


if __name__ == "__main__":
    main()