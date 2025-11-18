import os
import sys
from random import shuffle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *

from sklearn.model_selection import KFold
from collections import defaultdict

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    #print("Training on {} samples...".format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    #print("Make prediction for {} samples...".format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


datasets = [["davis_b3", "kiba", "pharos"][int(sys.argv[1])]]
modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv) > 3:
    cuda_name = ["cuda:0", "cuda:1"][int(sys.argv[3])]
print("cuda_name:", cuda_name)

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print("Learning rate: ", LR)
print("Epochs: ", NUM_EPOCHS)


K_REPEATS = 5  # or whatever k you want
prediction_dict = defaultdict(list)
pharos_pred_dict = defaultdict(list)

for repeat in range(K_REPEATS):
    print(f"\n================ Repeat {repeat+1}/{K_REPEATS} ================\n")
    # Main program: iterate over different datasets
    for dataset in datasets:
        print("\nrunning on ", model_st + "_" + dataset)
        processed_data_file_train = f"data/processed/{dataset}_train_{repeat+1}.pt"
        processed_data_file_test = f"data/processed/{dataset}_test_{repeat+1}.pt"
        if (not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test)):
            print("please run create_data.py to prepare data in pytorch format!")
            continue

        train_data = TestbedDataset(root="data", dataset=f"{dataset}_train_{repeat+1}")
        test_data = TestbedDataset(root="data", dataset=f"{dataset}_test_{repeat+1}")
        
        n_folds = 5
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=12345)  # for reproducibility
        fold_val_preds = []  # To store validation set predictions for each fold
        fold_val_labels = []  # To store corresponding true labels
        fold_results = []
        fold_drug_ids = []
        fold_protein_ids = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
            print(f"\n--- Fold {fold+1}/{n_folds} ---")

            # Create train and valid datasets for this fold
            train_subset = torch.utils.data.Subset(train_data, train_idx)
            valid_subset = torch.utils.data.Subset(train_data, val_idx)

            # make data PyTorch mini-batch processing ready
            train_loader = DataLoader(train_subset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            valid_loader = DataLoader(valid_subset, batch_size=TEST_BATCH_SIZE, shuffle=False)
            
            # training the model
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            model = modeling().to(device)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            best_mse = 1000
            best_test_mse = 1000
            best_test_ci = 0
            best_epoch = -1
            for epoch in range(NUM_EPOCHS):
                train(model, device, train_loader, optimizer, epoch + 1)
                G, P = predicting(model, device, valid_loader)
                val = mse(G, P)
                if val < best_mse:
                    best_mse = val
                    best_epoch = epoch + 1
                    best_model_state = model.state_dict()
                    
            print(f"Best validation MSE for fold {fold+1}: {best_mse} at epoch {best_epoch}")
            # After training, predict with best model on validation set
            model.load_state_dict(best_model_state)
            G_val, P_val = predicting(model, device, valid_loader)
            drug_ids = []
            protein_ids = []
            for data in valid_loader.dataset:
                drug_ids.append(data.drug_id)
                protein_ids.append(data.protein_id)
            fold_val_preds.append(P_val)
            fold_val_labels.append(G_val)
            fold_drug_ids.append(drug_ids)
            fold_protein_ids.append(protein_ids)

            # Optionally save fold-by-fold results
            fold_result = [rmse(G_val, P_val), mse(G_val, P_val), pearson(G_val, P_val), spearman(G_val, P_val), ci(G_val, P_val)]
            fold_results.append(fold_result)

        # After all folds: Aggregate validation results
        fold_val_preds = np.concatenate(fold_val_preds)
        fold_val_labels = np.concatenate(fold_val_labels)
        fold_drug_ids = np.concatenate(fold_drug_ids)
        fold_protein_ids = np.concatenate(fold_protein_ids)

        for d_id, p_id, pred, true in zip(fold_drug_ids, fold_protein_ids, fold_val_preds, fold_val_labels):
            key = (d_id, p_id, true)
            prediction_dict[key].append(pred)

        # Save fold metrics
        fold_metrics_df = pd.DataFrame(fold_results, columns=["RMSE", "MSE", "Pearson", "Spearman", "CI"])
        fold_metrics_df.to_csv(f"crossval_fold_metrics_{model_st}_{dataset}_{repeat+1}.csv", index=False)

        # FINAL TEST SET EVALUATION   
        print("Training final model on full training data...")
        # Retrain model on full training set
        full_train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range(NUM_EPOCHS):
            train(model, device, full_train_loader, optimizer, epoch + 1)
            
        # Save final model
        final_model_path = f"saved_models/final_model_{model_st}_{dataset}_{repeat+1}.pt"
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)  # Create directory if missing
        torch.save(model.state_dict(), final_model_path)
        print(f"Final trained model saved to {final_model_path}")

        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        G_test, P_test = predicting(model, device, test_loader)
        # Collecting the drug and protein IDs for the test set
        test_drug_ids = []
        test_protein_ids = []
        for data in test_loader.dataset:
            test_drug_ids.append(data.drug_id)
            test_protein_ids.append(data.protein_id)

        test_result = [rmse(G_test, P_test), mse(G_test, P_test), pearson(G_test, P_test), spearman(G_test, P_test), ci(G_test, P_test)]
        print("Final test set results:", test_result)

        # Save final test predictions
        test_df = pd.DataFrame({
            "Drug_ID": test_drug_ids,
            "Protein_ID": test_protein_ids,
            "True_Label": G_test,
            "Predicted_Value": P_test
        })
        test_df.to_csv(f"final_test_predictions_{model_st}_{dataset}_{repeat+1}.csv", index=False)
        print(f"Final test predictions saved to final_test_predictions_{model_st}_{dataset}_{repeat+1}.csv")
        
        # PHAROS TEST SET EVALUATION
        print("Predicting on Pharos test set...")
        
        pharos_data = TestbedDataset(root='data', dataset='pharos_test')

        # Create DataLoader for Pharos test set
        pharos_loader = DataLoader(pharos_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # Predict on Pharos set
        G_pharos, P_pharos = predicting(model, device, pharos_loader)

        # Collect drug and protein IDs for the Pharos set
        for d_id, p_id, pred, true in zip(
            [d.drug_id for d in pharos_loader.dataset],
            [d.protein_id for d in pharos_loader.dataset],
            P_pharos, G_pharos
        ):
            pharos_pred_dict[(d_id, p_id, true)].append(pred)

pharos_avg_drug_ids = []
pharos_avg_protein_ids = []
pharos_avg_preds = []
pharos_avg_labels = []

for (d_id, p_id, true), preds in pharos_pred_dict.items():
    pharos_avg_drug_ids.append(d_id)
    pharos_avg_protein_ids.append(p_id)
    pharos_avg_preds.append(np.mean(preds))
    pharos_avg_labels.append(true)
    
pharos_df = pd.DataFrame({
    "Drug_ID": pharos_avg_drug_ids,
    "Protein_ID": pharos_avg_protein_ids,
    "True_Label": pharos_avg_labels,
    "Predicted_Value": pharos_avg_preds
})

pharos_output_path = f"pharos_test_predictions_{model_st}.csv"
pharos_df.to_csv(pharos_output_path, index=False)
print(f"Pharos test predictions saved to {pharos_output_path}")
        
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

validation_df = pd.DataFrame({
    "Drug_ID": final_drug_ids,
    "Protein_ID": final_protein_ids,
    "True_Label": final_labels,
    "Predicted_Value": final_preds
})

validation_df.to_csv(f"crossval_predictions_{model_st}_{dataset}.csv", index=False)
