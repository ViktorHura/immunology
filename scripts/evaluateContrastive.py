import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

from utils import setupLogger
from data_preprocessingContrastive import TCRDataset, ValDataset, TestDataset, Refset

from modelContrastive import SiameseNetworkContrastive as SiameseNetwork, evaluate_model, encode_data
from trainingContrastive import classify, roc_auc_score


def main():
    model_name = "ct2/model_ct2.pt"
    model_path = "../output/contrastiveModel/"+model_name
    output_dir = f"../output/contrastiveModel/{model_name[:-3]}/"

    logger = setupLogger(output_dir+"output.txt")

    train_data = TCRDataset.load('../output/trainContrastive.pickle')
    validation_data = ValDataset.load('../output/valContrastive.pickle')

    input_size = train_data.tensor_size

    val_loader = DataLoader(validation_data, batch_size=2000, num_workers=4, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SiameseNetwork(input_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    reference_data = pd.DataFrame(train_data.encoding_dict.values(), columns=['sequence', 'epitope'])
    print(f'\rEncoding validation data', end='')
    val_encodings, val_epitopes, val_labels = evaluate_model(val_loader, model, device)

    print(f'\rEncoding reference data', end='')
    ref_encodings = Refset(list(reference_data['sequence']))
    ref_loader = DataLoader(ref_encodings, batch_size=2000, num_workers=4, shuffle=False)
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

    scores += [0.5, 0.5, 0.5, 0.5]
    est_score = np.average(scores)
    logger.info(f'{"Estimated Macro Auc@0.1":10} : {est_score}')

    RocCurveDisplay.from_predictions(val_labels, val_pred, plot_chance_level=True).plot()
    plt.show()

    test_data = pd.read_csv('../data/test_data/test.csv')
    test_peptides = list(test_data['Peptide'].unique())
    full_reference_data = TCRDataset("../data/training_data/concatenated.csv", test_peptides, "../data/AA_keys.csv", 25, None)

    reference_data = pd.DataFrame(full_reference_data.encoding_dict.values(), columns=['sequence', 'epitope'])
    ref_encodings = Refset(list(reference_data['sequence']))
    ref_loader = DataLoader(ref_encodings, batch_size=2000, num_workers=4, shuffle=False)
    ref_encodings = encode_data(ref_loader, model, device)
    ref_epitopes = reference_data['epitope']

    test_set = TestDataset(test_data[['CDR3b_extended', 'CDR3a_extended']], "../data/AA_keys.csv", 25)
    test_loader = DataLoader(test_set, batch_size=2000, num_workers=4, shuffle=False)
    test_encodings = encode_data(test_loader, model, device)
    test_epitopes = test_data['Peptide']

    predictions = classify(test_encodings, test_epitopes, ref_encodings, ref_epitopes, K=best_k)

    test_data['Prediction'] = predictions

    test_data.to_csv("../output/submissionContrastive.csv", index=False)


if __name__ == "__main__":
    main()
