import pandas as pd
import os


def main():
    training_data = pd.DataFrame(columns=['Epitope', 'Label', 'TRA', 'TRB'])
    test_data = pd.DataFrame(columns=['Epitope', 'Label', 'TRA', 'TRB'])

    for f in os.listdir("../data/training_data/"):
        tmp = pd.DataFrame(columns=['Epitope', 'Label', 'TRA', 'TRB'])
        epitope = f[:-4]
        data = pd.read_csv("../data/training_data/"+f, sep='\t')
        tmp['Label'] = data['Label']
        tmp['TRA'] = data['TRA_CDR3']
        tmp['TRB'] = data['TRB_CDR3']
        tmp['Epitope'] = epitope
        training_data = pd.concat([training_data, tmp], ignore_index=True)

    for f in os.listdir("../data/true_set/"):
        tmp = pd.DataFrame(columns=['Epitope', 'Label', 'TRA', 'TRB'])
        epitope = f[:-4]
        data = pd.read_csv("../data/true_set/"+f, sep='\t')
        tmp['Label'] = data['Label']
        tmp['TRA'] = data['TRA_CDR3']
        tmp['TRB'] = data['TRB_CDR3']
        tmp['Epitope'] = epitope
        test_data = pd.concat([test_data, tmp], ignore_index=True)

    training_data.to_csv('../output/train.csv', index=False)
    test_data.to_csv('../output/test.csv', index=False)


if __name__ == "__main__":
    main()