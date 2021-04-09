from rdkit import Chem
import pandas as pd
from random import shuffle
import torch
from torch_geometric.data import InMemoryDataset, Data
import pickle
import os

class UnscreenedCompounds(InMemoryDataset):
    def __init__(self, root,raw_file_name, transform=None, pre_transform=None):
        self.root = root
        self.raw_file_name = raw_file_name
        super(UnscreenedCompounds, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return [self.raw_file_name]
    @property
    def processed_file_names(self):
        return [self.raw_file_name.replace('csv','pt')]
    def download(self):
        pass
    def process(self):
        data_list = []
        bond_orders = {Chem.BondType.SINGLE:1.0,Chem.BondType.DOUBLE:2.0,Chem.BondType.TRIPLE:3.0,Chem.BondType.AROMATIC:1.5}
        print('hi')
        with open(os.path.join(self.root,'vocab.txt'), 'rb') as file:
            vocab = pickle.load(file)
            print('hi')
            for i, row in pd.read_csv(self.raw_paths[0]).iterrows():
                mol = Chem.MolFromSmiles(row["SMILES"])
                feature_matrix = torch.zeros(mol.GetNumAtoms(),len(vocab))
                for i, atom in enumerate(mol.GetAtoms()):
                    feature_matrix[i][vocab.index(atom.GetSymbol())]=1.0
                edge_index = torch.tensor([[index for bond in mol.GetBonds() for index in (bond.GetBeginAtomIdx(),bond.GetEndAtomIdx())],
                    [index for bond in mol.GetBonds() for index in (bond.GetEndAtomIdx(),bond.GetBeginAtomIdx())]])
                edge_weight = torch.tensor([bond_orders[bond.GetBondType()] for i in range(2) for bond in mol.GetBonds()])
                data = Data(x=feature_matrix, edge_index=edge_index, edge_weight=edge_weight)
                data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])