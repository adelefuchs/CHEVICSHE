# coldDTA_double_5fold

import os
import os.path as osp
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from metrics import get_cindex, get_rm2
from model import ColdDTA

# ========== CONFIG ==========
K_REPEATS = 5
N_FOLDS = 5
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LR = 3e-4
NUM_EPOCHS = 300

DATASET = ""
PHAROS_DATASET = "pharos"
ROOT_DIR = f"data/"

class GNNDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        index,
        types="train",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        if types == "pharos":
            self.data, self.slices = torch.load(
                os.path.join(root, "processed", "pharos.pt")
            )
        elif types == "train":
            self.data, self.slices = torch.load(self.processed_paths[index])
        elif types == "test":
            self.data, self.slices = torch.load(self.processed_paths[index + K_REPEATS])

    @property
    def raw_file_names(self):
        return [f"raw/davis_b3_train_{i}.csv" for i in range(1, K_REPEATS+1)] + [
            f"raw/davis_b3_test_{i}.csv" for i in range(1, K_REPEATS+1)
        ]

    @property
    def processed_file_names(self):
        return [f"davis_b3_train_{i}.pt" for i in range(1, K_REPEATS+1)] + [
            f"davis_b3_test_{i}.pt" for i in range(1, K_REPEATS+1)
        ]

    def download(self):
        pass

def train_epoch(model, device, loader, optimizer, loss_fn):
    model.train()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output.view(-1), data.y.view(-1))
        loss.backward()
        optimizer.step()

def predict(model, device, loader):
    model.eval()
    true, pred, d_ids, p_ids = [], [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            pred.append(output.view(-1).cpu().numpy())
            true.append(data.y.view(-1).cpu().numpy())
            d_ids.extend([d for d in data.drug_id])
            p_ids.extend([p for p in data.protein_id])
    return np.concatenate(true), np.concatenate(pred), d_ids, p_ids

def evaluate(model, loader, device, loss_fn):
    y_true, y_pred, _, _ = predict(model, device, loader)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ci = get_cindex(y_true, y_pred)
    r2 = get_rm2(y_true, y_pred)
    return mse, rmse, ci, r2

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    blinded_test_dict = defaultdict(list)
    crossval_dict = defaultdict(list)
    pharos_pred_dict = defaultdict(list)

    for repeat in range(K_REPEATS):
        print(f"\n================ Repeat {repeat+1}/{K_REPEATS} ================\n")

        train_data = GNNDataset(ROOT_DIR, index=repeat, types="train")
        test_data = GNNDataset(ROOT_DIR, index=repeat, types="test")

        inner_kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=repeat)
        best_model_state = None
        best_val_loss = float("inf")

        for fold, (tr_idx, val_idx) in enumerate(inner_kf.split(train_data)):
            print(f"\n  - Inner Fold {fold+1}/{N_FOLDS}")

            train_subset = Subset(train_data, tr_idx)
            val_subset = Subset(train_data, val_idx)
            train_loader = DataLoader(train_subset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=TEST_BATCH_SIZE, shuffle=False)

            model = ColdDTA().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            loss_fn = nn.MSELoss()

            for epoch in range(NUM_EPOCHS):
                train_epoch(model, device, train_loader, optimizer, loss_fn)

            val_true, val_pred, val_dids, val_pids = predict(model, device, val_loader)
            for d_id, p_id, pred, true in zip(val_dids, val_pids, val_pred, val_true):
                crossval_dict[(d_id, p_id, true)].append(pred)

            val_mse, _, _, _ = evaluate(model, val_loader, device, loss_fn)
            print(f"    Validation MSE: {val_mse:.4f}")
            if val_mse < best_val_loss:
                best_val_loss = val_mse
                best_model_state = model.state_dict()

        model = ColdDTA().to(device)
        model.load_state_dict(best_model_state)
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range(NUM_EPOCHS):
            train_epoch(model, device, train_loader, optimizer, loss_fn)


        model_save_path = os.path.join("saved_models", f"ColdDTA_repeat{repeat+1}.pt")
        os.makedirs("saved_models", exist_ok=True)

        torch.save({
            "model_state_dict": model.state_dict(),
            "hyperparameters": {
                "repeat": repeat,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "test_batch_size": TEST_BATCH_SIZE,
                "learning_rate": LR,
                "num_epochs": NUM_EPOCHS,
                "n_folds": N_FOLDS,
                "k_repeats": K_REPEATS,
                "model_name": "ColdDTA",
                "dataset": DATASET,
                "pharos_dataset": PHAROS_DATASET,
            }
        }, model_save_path)

        print(f"    🔒 Saved model + hyperparameters to {model_save_path}")

        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        G_test, P_test, d_ids, p_ids = predict(model, device, test_loader)
        for d_id, p_id, pred, true in zip(d_ids, p_ids, P_test, G_test):
            blinded_test_dict[(d_id, p_id, true)].append(pred)

        df = pd.DataFrame({"Drug_ID": d_ids, "Protein_ID": p_ids, "True_Label": G_test, "Predicted_Value": P_test})
        df.to_csv(f"final_test_predictions_ColdDTA_{DATASET}_repeat{repeat+1}.csv", index=False)

        pharos_data = GNNDataset("data", index=0, types="pharos")
        pharos_loader = DataLoader(pharos_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        G_pharos, P_pharos, d_ids_p, p_ids_p = predict(model, device, pharos_loader)
        for d_id, p_id, pred, true in zip(d_ids_p, p_ids_p, P_pharos, G_pharos):
            pharos_pred_dict[(d_id, p_id, true)].append(pred)

    rows = []
    for (d_id, p_id, true), preds in crossval_dict.items():
        rows.append([d_id, p_id, true, np.mean(preds)])
    pd.DataFrame(rows, columns=["Drug_ID", "Protein_ID", "True_Label", "Predicted_Value"]).to_csv(f"crossval_predictions_ColdDTA_{DATASET}.csv", index=False)

    rows = []
    for (d_id, p_id, true), preds in blinded_test_dict.items():
        rows.append([d_id, p_id, true, np.mean(preds)])
    pd.DataFrame(rows, columns=["Drug_ID", "Protein_ID", "True_Label", "Predicted_Value"]).to_csv(f"blinded_test_predictions_ColdDTA_{DATASET}.csv", index=False)

    rows = []
    for (d_id, p_id, true), preds in pharos_pred_dict.items():
        rows.append([d_id, p_id, true, np.mean(preds)])
    pd.DataFrame(rows, columns=["Drug_ID", "Protein_ID", "True_Label", "Predicted_Value"]).to_csv(f"pharos_test_predictions_ColdDTA.csv", index=False)

    print(" All Done.")
