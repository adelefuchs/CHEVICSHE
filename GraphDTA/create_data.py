import json
import os
import pickle
from collections import OrderedDict

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolFromSmiles

from utils import *


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x  


# from DeepDTA data
all_prots = []
datasets = ['davis_b3']
    
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

compound_iso_smiles = []
folds = 5 
for dt_name in ['davis_b3']:
    opts = ['train','test']
    for opt in opts:
        for fold in range(folds):
            file_path = f'data/{dt_name}_{opt}_{fold+1}.csv'
            if os.path.isfile(file_path):
                df = pd.read_csv(file_path)
                compound_iso_smiles += list(df['compound_iso_smiles'])
            else:
                print(f'Warning: {file_path} does not exist and will be skipped.')

compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g

datasets = ['davis_b3']
# convert to PyTorch data format
for dataset in datasets:
    for fold in range(folds):
        processed_data_file_train = f"data/processed/{dataset}_train_{fold+1}.pt"
        processed_data_file_test = f"data/processed/{dataset}_test_{fold+1}.pt"

        if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
            df = pd.read_csv(f"data/{dataset}_train_{fold+1}.csv")
            train_drugs, train_prots,  train_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
            train_drug_ids = list(df['drug_id'])  # Ensure these columns are in the CSV
            train_protein_ids = list(df['protein_id']) 
            XT = [seq_cat(t) for t in train_prots]
            train_drugs, train_prots,  train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)
            train_drug_ids, train_protein_ids = np.asarray(train_drug_ids), np.asarray(train_protein_ids)
            df = pd.read_csv(f"data/{dataset}_test_{fold+1}.csv")
            test_drugs, test_prots,  test_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
            test_drug_ids = list(df['drug_id'])  # Ensure these columns are in the CSV
            test_protein_ids = list(df['protein_id'])  # Ensure these columns are in the CSV
            XT = [seq_cat(t) for t in test_prots]
            test_drugs, test_prots,  test_Y = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y)
            test_drug_ids, test_protein_ids = np.asarray(test_drug_ids), np.asarray(test_protein_ids)

            # make data PyTorch Geometric ready
            train_data = TestbedDataset(root='data', dataset=f"{dataset}_train_{fold+1}", xd=train_drugs, xt=train_prots, y=train_Y, drug_ids=train_drug_ids, protein_ids=train_protein_ids, smile_graph=smile_graph)
            test_data = TestbedDataset(root='data', dataset=f"{dataset}_test_{fold+1}", xd=test_drugs, xt=test_prots, y=test_Y, drug_ids=test_drug_ids, protein_ids=test_protein_ids, smile_graph=smile_graph)
            print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')        
        else:
            print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')        
