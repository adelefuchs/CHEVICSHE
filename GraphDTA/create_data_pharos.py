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
    if mol is None:
        print(f"Invalid SMILES: {smile}")
        
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
datasets = ['pharos']
    
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

compound_iso_smiles = []
for dt_name in ['pharos']:
    df = pd.read_csv('data/' + dt_name + '.csv')
    compound_iso_smiles += list( df['compound_iso_smiles'] )
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g
    
# Iterate over the dictionary and remove invalid entries
keys_to_remove = []

for smile, (c_size, features, edge_index) in smile_graph.items():
    edge_index = np.array(edge_index)

    # Check if the shape is incorrect
    if edge_index.shape != (edge_index.shape[0], 2):
        print(f"Warning: Element with SMILES {smile} has incorrect edge_index shape: {edge_index.shape}")
        keys_to_remove.append(smile)

# Remove the invalid entries from the original dictionary
print(len(keys_to_remove))
for key in keys_to_remove:
    del smile_graph[key]

# Now smile_graph contains only valid entries
print(f"Number of valid entries in smile_graph: {len(smile_graph)}")


datasets = ['pharos']
# convert to PyTorch data format
for dataset in datasets:
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if (not os.path.isfile(processed_data_file_test)):
        df = pd.read_csv('data/' + dataset + '.csv')
        test_drugs, test_prots,  test_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
        test_drug_ids = list(df['drug_id'])  # Ensure these columns are in the CSV
        test_protein_ids = list(df['protein_id'])  # Ensure these columns are in the CSV
        XT = [seq_cat(t) for t in test_prots]
        test_drugs, test_prots,  test_Y = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y)
        test_drug_ids, test_protein_ids = np.asarray(test_drug_ids), np.asarray(test_protein_ids)

        print('preparing ', dataset + '_test.pt in pytorch format!')
        test_data = TestbedDataset(root='data', dataset=dataset+'_test', xd=test_drugs, xt=test_prots, y=test_Y,drug_ids=test_drug_ids,protein_ids=test_protein_ids, smile_graph=smile_graph)
        print(processed_data_file_test, ' have been created')        
    else:
        print(processed_data_file_test, ' are already created')        
