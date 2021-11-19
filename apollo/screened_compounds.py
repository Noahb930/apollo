from rdkit import Chem
import pandas as pd
from random import shuffle
import torch
from torch_geometric.data import InMemoryDataset, Data
import pickle
import os

class ScreenedCompounds(InMemoryDataset):
    def __init__(self, root,validation_split,testing_split, mode, transform=None, pre_transform=None):
        self.root =root
        self.validation_split =validation_split
        self.testing_split = testing_split
        super(ScreenedCompounds, self).__init__(root, transform, pre_transform)
        path_indexes = {'training':0,'validation':1,'testing':2}
        self.data, self.slices = torch.load(self.processed_paths[path_indexes[mode]])
    @property
    def raw_file_names(self):
        return ["screened_compounds.csv"]
    @property
    def processed_file_names(self):
        return ['training_compounds.pt','validation_compounds.pt','testing_compounds.pt']
    def download(self):
        pass
    def process(self):
        data_list = []
        df = pd.read_csv(self.raw_paths[0])
        bond_orders = {Chem.BondType.SINGLE:1.0,Chem.BondType.DOUBLE:2.0,Chem.BondType.TRIPLE:3.0,Chem.BondType.AROMATIC:1.5}
        vocab = list(set([atom.GetSymbol() for smiles in df["SMILES"] for atom in Chem.MolFromSmiles(smiles).GetAtoms()]))
        with open(os.path.join(self.root,'vocab.txt'), 'wb') as file:
            pickle.dump(vocab, file)
        for i, row in df.iterrows():
            mol = Chem.MolFromSmiles(row["SMILES"])
            feature_matrix = torch.zeros(mol.GetNumAtoms(),len(vocab))
            for i, atom in enumerate(mol.GetAtoms()):
                feature_matrix[i][vocab.index(atom.GetSymbol())]=1.0
            edge_index = torch.tensor([[index for bond in mol.GetBonds() for index in (bond.GetBeginAtomIdx(),bond.GetEndAtomIdx())],
                [index for bond in mol.GetBonds() for index in (bond.GetEndAtomIdx(),bond.GetBeginAtomIdx())]])
            edge_weight = torch.tensor([bond_orders[bond.GetBondType()] for i in range(2) for bond in mol.GetBonds()])
            score = row['Score']
            data = Data(x=feature_matrix, edge_index=edge_index, edge_weight=edge_weight, y=torch.tensor([score]), id=i)
            data_list.append(data)
        shuffle(data_list)
        training_data_list = data_list[int(len(data_list)*self.validation_split):-1*int(len(data_list)*self.testing_split)]
        validation_data_list = data_list[:int(len(data_list)*self.validation_split)]
        testing_data_list = data_list[-1*int(len(data_list)*self.testing_split):]
        torch.save(self.collate(training_data_list), self.processed_paths[0])
        torch.save(self.collate(validation_data_list), self.processed_paths[1])
        torch.save(self.collate(testing_data_list), self.processed_paths[2])