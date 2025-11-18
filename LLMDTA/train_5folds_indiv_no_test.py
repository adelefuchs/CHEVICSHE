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

    for repeat in range(K_REPEATS):
        print(repeat)
        if repeat != (K_REPEATS - 1):
            continue  # skip all other folds
        print(f"\n================ Repeat {repeat+1}/{K_REPEATS} ================\n")
        train_dir = os.path.join(dataset_root, f"davis_b3_train_{repeat+1}.csv")
        test_dir = os.path.join(dataset_root, f"davis_b3_test_{repeat+1}.csv")

        train_set_full = CustomDataSet(load_and_cast_csv(train_dir), hp)
        test_set = CustomDataSet(load_and_cast_csv(test_dir), hp)

        # do 5 fold split of the training data
        # for each k fold
        n_folds = 5
        kf = KFold(
            n_splits=n_folds, shuffle=True, random_state=12345
        )  # for reproducibility
        fold_val_preds = []  # To store validation set predictions for each fold
        fold_val_labels = []  # To store corresponding true labels
        fold_results = []
        fold_drug_ids = []
        fold_protein_ids = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_set_full)):
            print(f"\n--- Fold {fold+1}/{n_folds} ---")
            # Create train and valid datasets for this fold
            train_subset = torch.utils.data.Subset(train_set_full, train_idx)
            valid_subset = torch.utils.data.Subset(train_set_full, val_idx)

            train_dataset_load = DataLoader(
                train_subset,
                batch_size=hp.Batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0,
                collate_fn=lambda x: my_collate_fn(
                    x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict
                ),
            )
            valid_dataset_load = DataLoader(
                valid_subset,
                batch_size=hp.Batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                collate_fn=lambda x: my_collate_fn(
                    x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict
                ),
            )
            print("load dataset finished")

            # Initialize model, optimizer, and criterion
            model = nn.DataParallel(LLMDTA(hp, device))
            model = model.to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=hp.Learning_rate, betas=(0.9, 0.999)
            )
            criterion = F.mse_loss

            # Initialize training loop variables
            best_valid_mse = 10
            patience = 0
            os.makedirs("./savemodel", exist_ok=True)
            model_fromTrain = f"./savemodel/{hp.dataset}--{hp.current_time}--repeat{repeat+1}--fold{fold+1}.pth"

            # Training loop
            for epoch in range(1, hp.Epoch + 1):
                model.train()
                pred = []
                label = []
                for batch_data in train_dataset_load:
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
                    predictions = model(
                        mol_vec,
                        mol_mat,
                        mol_mat_mask,
                        prot_vec,
                        prot_mat,
                        prot_mat_mask,
                    )
                    pred = (
                        pred + predictions.cpu().detach().numpy().reshape(-1).tolist()
                    )
                    label = label + affinity.cpu().detach().numpy().reshape(-1).tolist()

                    loss = criterion(predictions.squeeze(), affinity)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                pred = np.array(pred)
                label = np.array(label)
                mse_value, rmse_value, ci, r2, pearson_value, spearman_value = (
                    regression_scores(pred, label)
                )
                # print(
                #     f"Training Log epoch-{epoch}: mse-{mse_value}, rmse-{rmse_value}, r2-{r2}"
                # )

                # Validation
                val_metrics = test(
                    model, valid_dataset_load, hp.dataset, return_preds=False
                )
                mse, rmse, ci, r2, pearson, spearman = val_metrics

                # Early stopping
                if mse < best_valid_mse:
                    patience = 0
                    best_valid_mse = mse
                    torch.save(model.state_dict(), model_fromTrain)
                    # print(
                    #     f"Update best_mse: mse-{mse}, rmse-{rmse}, ci-{ci}, r2-{r2}, pearson-{pearson}, spearman-{spearman}"
                    # )
                else:
                    patience += 1
                    if patience > hp.max_patience:
                        print(
                            f"Training stopped at epoch-{epoch}, model saved at-{model_fromTrain}"
                        )
                        break

            # After training...Save fold predictions from best model?
            model.load_state_dict(torch.load(model_fromTrain))
            final_val = test(model, valid_dataset_load, hp.dataset, return_preds=True)
            (
                mse,
                rmse,
                ci,
                r2,
                pearson,
                spearman,
                val_drug_ids,
                val_prot_ids,
                val_labels,
                val_preds,
            ) = final_val
            fold_val_preds.append(val_preds)
            fold_val_labels.append(val_labels)
            fold_drug_ids.append(val_drug_ids)
            fold_protein_ids.append(val_prot_ids)

            fold_result = [rmse, mse, pearson, spearman, ci, r2]
            fold_results.append(fold_result)

        # After all folds: Aggregate validation results
        fold_val_preds = np.concatenate(fold_val_preds)
        fold_val_labels = np.concatenate(fold_val_labels)
        fold_drug_ids = np.concatenate(fold_drug_ids)
        fold_protein_ids = np.concatenate(fold_protein_ids)

        for d_id, p_id, pred, true in zip(
            fold_drug_ids, fold_protein_ids, fold_val_preds, fold_val_labels
        ):
            key = (d_id, p_id, true)
            prediction_dict[key].append(pred)

        # Save fold metrics
        fold_metrics_df = pd.DataFrame(
            fold_results, columns=["RMSE", "MSE", "Pearson", "Spearman", "CI", "R2"]
        )
        fold_metrics_df.to_csv(
            f"crossval_fold_metrics_{hp.current_time}-{hp.dataset}_{repeat+1}_INDIV.csv",
            index=False,
        )

# After K_REPEATS
final_drug_ids = []
final_protein_ids = []
final_preds = []
final_labels = []

for (d_id, p_id, true), preds in prediction_dict.items():
    final_drug_ids.append(d_id)
    final_protein_ids.append(p_id)
    final_preds.append(np.mean(preds))  # average prediction
    final_labels.append(true)

validation_df = pd.DataFrame(
    {
        "Drug_ID": final_drug_ids,
        "Protein_ID": final_protein_ids,
        "True_Label": final_labels,
        "Predicted_Value": final_preds,
    }
)

validation_df.to_csv(f"crossval_predictions_repeat{K_REPEATS}_INDIV.csv", index=False)
