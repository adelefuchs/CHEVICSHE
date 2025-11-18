import argparse
import csv
import os
import pickle
import random
import sys
from collections import defaultdict
from math import sqrt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparameter import HyperParameter
from MyDataset import CustomDataSet, batch2tensor, my_collate_fn
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from LLMDTA import LLMDTA as LLMDTA


def cindex_score(y, p):
    sum_m = 0
    pair = 0
    for i in range(1, len(y)):
        for j in range(0, i):
            if i is not j:
                if y[i] > y[j]:
                    pair += 1
                    sum_m += 1 * (p[i] > p[j]) + 0.5 * (p[i] == p[j])
    if pair != 0:
        return sum_m / pair
    else:
        return 0


def regression_scores(label, pred):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    mse = ((label - pred) ** 2).mean(axis=0)
    rmse = sqrt(mse)
    ci = cindex_score(label, pred)
    r2 = r2_score(label, pred)
    pearson = np.corrcoef(label, pred)[0, 1]
    spearman = stats.spearmanr(label, pred)[0]
    return (
        round(mse, 6),
        round(rmse, 6),
        round(ci, 6),
        round(r2, 6),
        round(pearson, 6),
        round(spearman, 6),
    )


def load_pickle(dir):
    with open(dir, "rb+") as f:
        return pickle.load(f)


def test(model, dataloader, dataset, return_preds=False):
    model.eval()
    preds = []
    labels = []
    drug_ids = []
    protein_ids = []

    for batch_i, batch_data in enumerate(dataloader):
        (
            mol_vec,
            prot_vec,
            mol_mat,
            mol_mat_mask,
            prot_mat,
            prot_mat_mask,
            affinity,
            drug_id,
            protein_id,
        ) = batch_data
        with torch.no_grad():
            pred = model(
                mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask
            )
            preds += pred.cpu().detach().numpy().reshape(-1).tolist()
            labels += affinity.cpu().numpy().astype(np.float64).reshape(-1).tolist()
            #labels += affinity.cpu().numpy().reshape(-1).tolist()
            drug_ids += drug_id
            protein_ids += protein_id

    preds = np.array(preds)
    labels = np.array(labels)
    drug_ids = np.array(drug_ids)
    protein_ids = np.array(protein_ids)

    mse_value, rmse_value, ci, r2, pearson_value, spearman_value = regression_scores(
        labels, preds
    )

    if return_preds:
        return (
            mse_value,
            rmse_value,
            ci,
            r2,
            pearson_value,
            spearman_value,
            drug_ids,
            protein_ids,
            labels,
            preds,
        )
    return mse_value, rmse_value, ci, r2, pearson_value, spearman_value


def load_and_cast_csv(path):
    df = pd.read_csv(path, sep=",")
    print(df.columns)
    if "drug_id" in df.columns:
        df["drug_id"] = df["drug_id"].astype(str)
    if "protein_id" in df.columns:
        df["protein_id"] = df["protein_id"].astype(str)
    if "prot_id" in df.columns:
        df["prot_id"] = df["prot_id"].astype(str)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train model with K-fold cross-validation."
    )
    parser.add_argument(
        "--k_repeats",
        type=int,
        default=1,
        help="Number of times to repeat the K-fold cross-validation.",
    )
    args = parser.parse_args()

    SEED = 0
    K_REPEATS = args.k_repeats  # Use the command-line argument
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)

    hp = HyperParameter()
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(f"Dataset-{hp.dataset}")
    print(f"Pretrain-{hp.mol2vec_dir}-{hp.protvec_dir}")

    # Initialize the arrays that will hold the final blinded test predictions and the pharos test predictions
    prediction_dict = defaultdict(list)
    pharos_pred_dict = defaultdict(list)
    pharos_label_dict = {}

    fold_metrics = {
        "mse": [],
        "rmse": [],
        "ci": [],
        "r2": [],
        "pearson": [],
        "spearman": [],
    }
    dataset_root = os.path.join(hp.data_root, hp.dataset)

    drug_df = load_and_cast_csv(hp.drugs_dir)
    prot_df = load_and_cast_csv(hp.prots_dir)
    mol2vec_dict = load_pickle(hp.mol2vec_dir)
    protvec_dict = load_pickle(hp.protvec_dir)
    
    train_dir = os.path.join(dataset_root, f"davis_b3_train_{K_REPEATS}.csv")
    test_dir = os.path.join(dataset_root, f"davis_b3_test_{K_REPEATS}.csv")

    train_set_full = CustomDataSet(load_and_cast_csv(train_dir), hp)
    test_set = CustomDataSet(load_and_cast_csv(test_dir), hp)
    
    print("\nTraining on full training set for final evaluation...")
    model_fromTrain = f"./savemodel/{hp.dataset}--{hp.current_time}--repeat{K_REPEATS}.pth"

    full_train_loader = DataLoader(
        train_set_full,
        batch_size=hp.Batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=lambda x: my_collate_fn(
            x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict
        ),
    )

    final_model = nn.DataParallel(LLMDTA(hp, device)).to(device)
    optimizer = torch.optim.Adam(
        final_model.parameters(), lr=hp.Learning_rate, betas=(0.9, 0.999)
    )
    criterion = F.mse_loss

    best_full_train_mse = float("inf")
    patience = 0

    for epoch in range(1, hp.Epoch + 1):
        final_model.train()
        pred, label = [], []

        for batch_data in full_train_loader:
            (
                mol_vec,
                prot_vec,
                mol_mat,
                mol_mat_mask,
                prot_mat,
                prot_mat_mask,
                affinity,
                drug_id,
                prot_id,
            ) = batch_data
            predictions = final_model(
                mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask
            )
            loss = criterion(predictions.squeeze(), affinity)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pred.extend(predictions.cpu().detach().numpy().reshape(-1).tolist())
            label.extend(affinity.cpu().detach().numpy().reshape(-1).tolist())

        mse_value = mean_squared_error(label, pred)
        if mse_value < best_full_train_mse:
            best_full_train_mse = mse_value
            patience = 0
            torch.save(final_model.state_dict(), model_fromTrain)  # reuse same path
            print(f"Final trained model saved to {model_fromTrain}")
            print(f"[Full Training Epoch {epoch}] Improved MSE: {mse_value}")
        else:
            patience += 1
            if patience > hp.max_patience:
                print(f"Early stopping at epoch {epoch} for full training.")
                break

    # Test
    predModel = nn.DataParallel(LLMDTA(hp, device))
    predModel.load_state_dict(torch.load(model_fromTrain))
    predModel = predModel.to(device)
    test_dataset_load = DataLoader(
        test_set,
        batch_size=hp.Batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=lambda x: my_collate_fn(
            x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict
        ),
    )
    (
        mse,
        rmse,
        ci,
        r2,
        pearson,
        spearman,
        test_drug_id,
        test_prot_id,
        test_labels,
        test_preds,
    ) = test(predModel, test_dataset_load, hp.dataset, return_preds=True)
    print(
        f"Test: mse: {mse}, rmse: {rmse}, ci: {ci}, r2: {r2}, pearson: {pearson}, spearman: {spearman}\n"
    )

    # Load original test CSV and extract true affinities in order
    original_test_df = load_and_cast_csv(test_dir)
    true_affinity_precise = original_test_df["affinity"].tolist()

    # Create DataFrame for output with full-precision true affinities
    test_df = pd.DataFrame(
        {
            "Drug_ID": test_drug_id,
            "Protein_ID": test_prot_id,
            "true_affinity": true_affinity_precise,
            "predicted_affinity": test_preds,
        }
    )
    
    test_df.to_csv(
        f"./{hp.dataset}_repeat{K_REPEATS}_final_test_predictions_INDIV.csv",
        index=False,
        float_format="%.25g"
    )
    
    print(
        f"Final test predictions saved to ./{hp.dataset}_repeat{K_REPEATS}_final_test_predictions_INDIV.csv"
    )

    # PHAROS TEST SET EVALUATION
    print("Predicting on Pharos test set...")

    # Load the Pharos dataset
    pharos_dir = os.path.join(dataset_root, f"pharos.csv")
    pharos_data = CustomDataSet(load_and_cast_csv(pharos_dir), hp)

    # Create DataLoader for Pharos test set
    pharos_loader = DataLoader(
        pharos_data,
        batch_size=hp.Batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=lambda x: my_collate_fn(
            x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict
        ),
    )

    # Predict on Pharos set using the test function
    (
        mse,
        rmse,
        ci,
        r2,
        pearson,
        spearman,
        pharos_drug_ids,
        pharos_protein_ids,
        pharos_labels,
        pharos_preds,
    ) = test(predModel, pharos_loader, hp.dataset, return_preds=True)

    # Collect predictions and true labels for the Pharos set
    pharos_avg_drug_ids = []
    pharos_avg_protein_ids = []
    pharos_avg_preds = []
    pharos_avg_labels = []

    for d_id, p_id, pred, true in zip(
        pharos_drug_ids, pharos_protein_ids, pharos_preds, pharos_labels
    ):
        key = (d_id, p_id, true)  # Include affinity (true label) in the key
        pharos_pred_dict[key].append(pred)
        if key not in pharos_label_dict:
            pharos_label_dict[key] = true

    for (d_id, p_id, true), preds in pharos_pred_dict.items():
        pharos_avg_drug_ids.append(d_id)
        pharos_avg_protein_ids.append(p_id)
        pharos_avg_preds.append(np.mean(preds))
        pharos_avg_labels.append(true)

    print(f"Total predictions written: {len(pharos_avg_drug_ids)}")

    pharos_df = pd.DataFrame(
        {
            "Drug_ID": pharos_avg_drug_ids,
            "Protein_ID": pharos_avg_protein_ids,
            "True_Label": pharos_avg_labels,
            "Predicted_Value": pharos_avg_preds,
        }
    )

    pharos_output_path = f"pharos_test_predictions_repeat{K_REPEATS}_INDIV.csv"
    pharos_df.to_csv(pharos_output_path, index=False, float_format="%.25g")
    print(f"Pharos test predictions saved to {pharos_output_path}")
