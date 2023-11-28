import copy

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import itertools
import pickle

import random

seed = 42
negatives_per_val_pair = 5

random.seed(seed)

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

    def encodeSequence(self, sequence):
        mat = np.array([self.aa_keys.loc[aa] for aa in sequence])
        padding = np.zeros((self.max_sequence_length - mat.shape[0], mat.shape[1]))
        mat = np.append(mat, padding, axis=0)
        mat = np.transpose(mat)
        mat = torch.from_numpy(mat)
        return torch.reshape(mat, (mat.size(dim=0), 1,  mat.size(dim=1)))

    def generatePairs(self, data_path, val_peptides):
        all_data = pd.read_csv(data_path)
        grouped_data = dict(tuple(all_data.groupby("Peptide")))

        val_sequences = {}

        self.epitopes = list(grouped_data.keys())

        for i, epitope in enumerate(self.epitopes):
            data = grouped_data[epitope]['CDR3b_extended']
            if len(data) < 2:
                continue

            train = data.sample(frac=1-self.val_balance, random_state=seed)

            valid = data.drop(train.index)
            if not valid.empty:
                val_sequences[epitope] = valid

            for seq in train:
                self.encoding_dict[seq] = [self.encodeSequence(seq), epitope]

            for a, b in itertools.combinations(train, 2):
                self.pairs.append((a, b))

        val_dict = {}
        val_pairs = []

        val_epitopes = [e for e in self.epitopes if e in val_peptides]
        for epitope in val_epitopes:
            remainder = copy.deepcopy(val_epitopes)
            remainder.remove(epitope)
            for seq in val_sequences[epitope]:
                val_dict[seq] = self.encodeSequence(seq)
                val_pairs.append((seq, epitope, 1))  # positive pair

                for neg_epitope in random.sample(remainder, negatives_per_val_pair):
                    val_pairs.append((seq, neg_epitope, 0))  # negative pair

        self.val_dataset = ValDataset(val_dict, val_pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s1, s2 = self.pairs[idx]

        return self.encoding_dict[s1][0], self.encoding_dict[s2][0]

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
    # b = list(pd.read_csv('../data/training_data/VDJdb_paired_chain.csv')['Peptide'].unique())
    #
    # print(test_peptides, len(test_peptides))
    #
    # print()
    #
    # inter = [x for x in b if x in test_peptides]
    # print(inter, len(inter))
    #
    # print()
    #
    # notin = [x for x in test_peptides if x not in b]
    # print(notin, len(notin))
    #
    # print()

    training_data = TCRDataset("../data/training_data/VDJdb_paired_chain.csv", test_peptides, "../data/AA_keys.csv", 25, 0.20)
    training_data.save("../output/train.pickle", "../output/val.pickle")


if __name__ == "__main__":
    main()