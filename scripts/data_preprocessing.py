import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import itertools
import pickle


class TCRContrastiveDataset(Dataset):
    def __init__(self, data_path, AA_keys_path, max_sequence_length):
        self.aa_keys = pd.read_csv(AA_keys_path, index_col='One Letter')
        self.max_sequence_length = max_sequence_length
        self.pairs = []
        self.encoding_dict = {}
        self.generatePairs(data_path)
        self.tensor_size = (self.aa_keys.shape[1], 1, max_sequence_length)

    def encodeSequence(self, sequence):
        mat = np.array([self.aa_keys.loc[aa] for aa in sequence])
        padding = np.zeros((self.max_sequence_length - mat.shape[0], mat.shape[1]))
        mat = np.append(mat, padding, axis=0)
        mat = np.transpose(mat)
        mat = torch.from_numpy(mat)
        return torch.reshape(mat, (mat.size(dim=0), 1,  mat.size(dim=1)))

    def generatePairs(self, data_path):
        for f in os.listdir(data_path):
            data = pd.read_csv(data_path + f, sep='\t')
            #data = data.filter(['Label', 'TRA_CDR3', 'TRB_CDR3'])

            positives_seqs = data[data['Label'] == 1]
            positives_seqs = positives_seqs['TRB_CDR3'].tolist()

            negative_seqs = data[data['Label'] == -1]
            negative_seqs = negative_seqs['TRB_CDR3'].tolist()

            for s in positives_seqs:
                self.encoding_dict[s] = self.encodeSequence(s)

            for s in negative_seqs:
                self.encoding_dict[s] = self.encodeSequence(s)

            for a, b in itertools.combinations(positives_seqs, 2):
                self.pairs.append((a, b, 1))

            for a, b in itertools.product(positives_seqs, negative_seqs):
                self.pairs.append((a, b, 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s1, s2, label = self.pairs[idx]

        return self.encoding_dict[s1], self.encoding_dict[s2], label

    def save(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as handle:
            return pickle.load(handle)


def main():
    training_data = TCRContrastiveDataset("../data/training_data/", "../data/AA_keys.csv", 23)
    training_data.save("../output/training_dataset_contrastive.pickle")

    test_data = TCRContrastiveDataset("../data/true_set/", "../data/AA_keys.csv", 23)
    test_data.save("../output/test_dataset_contrastive.pickle")


if __name__ == "__main__":
    main()