import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pickle

from utils import setupLogger
from data_preprocessing import TCRDataset, ValDataset

from modelBYOL import SiameseNetworkBYOL as SiameseNetwork, evaluate_model, encode_data
from trainingBYOL import Refset, classify, roc_auc_score


def main():
    model_name = "T0/model_13.pt"
    model_path = "../output/byolModel/"+model_name
    output_dir = f"../output/byolModel/{model_name[:-3]}/"

    logger = setupLogger(output_dir+"output.txt")

    train_data = TCRDataset.load('../output/train.pickle')
    validation_data = ValDataset.load('../output/val.pickle')

    input_size = train_data.tensor_size

    val_loader = DataLoader(validation_data, batch_size=10000, num_workers=6, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SiameseNetwork(input_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    reference_data = pd.DataFrame(train_data.encoding_dict.values(), columns=['sequence', 'epitope'])
    print(f'\rEncoding validation data', end='')
    val_encodings, val_epitopes, val_labels = evaluate_model(val_loader, model, device)

    print(f'\rEncoding reference data', end='')
    ref_encodings = Refset(list(reference_data['sequence']))
    ref_loader = DataLoader(ref_encodings, batch_size=10000, num_workers=6, shuffle=False)
    ref_encodings = encode_data(ref_loader, model, device)
    ref_epitopes = reference_data['epitope']

    best_k = 1
    best_score = 0
    for k in range(1, 20):
        val_pred = classify(val_encodings, val_epitopes, ref_encodings, ref_epitopes, K=k)
        print('\r', end='')

        results = pd.DataFrame.from_dict({'epitope':val_epitopes, 'y': val_labels, 'y_pred': val_pred})
        scores = []
        grouped_data = dict(tuple(results.groupby("epitope")))
        for epitope in list(grouped_data.keys()):
            data = grouped_data[epitope]
            y, y_pred = data['y'], data['y_pred']

            score = roc_auc_score(y, y_pred, max_fpr=0.1)
            scores.append(score)

        macro_auc = np.average(scores)
        if macro_auc > best_score:
            best_score = macro_auc
            best_k = k

    print(f'best K: {best_k}')

    val_pred = classify(val_encodings, val_epitopes, ref_encodings, ref_epitopes, K=best_k)
    print('\r', end='')

    results = pd.DataFrame.from_dict({'epitope': val_epitopes, 'y': val_labels, 'y_pred': val_pred})

    scores = []
    logger.info('\n== ROC AUC@0.1 Scores ==')

    grouped_data = dict(tuple(results.groupby("epitope")))

    for epitope in list(grouped_data.keys()):
        data = grouped_data[epitope]
        y, y_pred = data['y'], data['y_pred']

        score = roc_auc_score(y, y_pred, max_fpr=0.1)
        scores.append(score)
        logger.info(f'{epitope:10} : {score}')

    logger.info('')

    macro_auc = np.average(scores)
    logger.info(f'{"Macro Auc@0.1":10} : {macro_auc}')

    scores += [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    est_score = np.average(scores)
    logger.info(f'{"Estimated Macro Auc@0.1":10} : {est_score}')




if __name__ == "__main__":
    main()
