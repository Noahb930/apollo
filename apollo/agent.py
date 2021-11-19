import sys
import os
from random import shuffle
import pickle
from torch_geometric.data import InMemoryDataset, Data

import click
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import DataLoader, Batch
import matplotlib.pyplot as plt
import pandas as pd
import xlsxwriter
from rdkit import Chem
from rdkit.Chem import Draw
from json import load, dump
from io import BytesIO, StringIO
from PIL import Image
import base64
from IPython.display import display, HTML

from hierarchical_pooling_network import HierarchicalPoolingNetwork
from lambada_rank_loss import LambadaRankLoss
from screened_compounds import ScreenedCompounds
from unscreened_compounds import UnscreenedCompounds

class Agent():
    def __init__(self,name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.path = name
        self.config = load(open(os.path.join(self.path,'config.json'),'r'))
        self.training_data = ScreenedCompounds(os.path.join(self.path,'data'),self.config["validation_split"],self.config["testing_split"],'training')
        self.validation_data = ScreenedCompounds(os.path.join(self.path,'data'),self.config["validation_split"],self.config["testing_split"],'validation')
        self.testing_data = ScreenedCompounds(os.path.join(self.path,'data'),self.config["validation_split"],self.config["testing_split"],'testing')
        self.vocab = pickle.load(open(os.path.join(self.path,'data/vocab.txt'),'rb'))
        self.model = HierarchicalPoolingNetwork(self.config['num_blocks'],len(self.vocab),self.config['num_channels'],'SAG',self.config['pooling_ratio'],self.config['dropout']).to(self.device)
        self.loss_func = LambadaRankLoss(self.config['ndcg_cutoff']).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'],weight_decay=self.config['weight_decay'])
        self.ndcg_cutoff = self.config['ndcg_cutoff']
        self.minibatch_size = self.config['minibatch_size']
        self.history = pd.DataFrame({'training_loss':[],'training_auc':[],'training_ndcg':[],'validation_loss':[],'validation_auc':[],'validation_ndcg':[]})
        if os.path.exists(os.path.join(self.path,'checkpoint.pt')):
            self.load()
    def train(self,patience):
        training_data_loader = DataLoader(self.training_data,batch_size = self.minibatch_size,shuffle=True)
        validation_batch = Batch.from_data_list(self.validation_data).to(self.device)
        record = float('inf')
        epochs_without_record = 0
        print("Starting Training")
        while True:
            validation_ranking, validation_loss, validation_auc, validation_ndcg = self.evaluate_one_epoch(validation_batch)
            if validation_loss < record:
                record=validation_loss
                epochs_without_record = 0
                self.save()
            else:
                epochs_without_record +=1
            if epochs_without_record > patience:
                break
            running_training_loss = torch.tensor(0.0).to(self.device)
            running_training_auc = torch.tensor(0.0).to(self.device)
            running_training_ndcg = torch.tensor(0.0).to(self.device)
            for training_minibatch in training_data_loader:
                loss, auc, ndcg = self.train_one_epoch(training_minibatch.to(self.device))
                running_training_loss += loss
                running_training_auc += auc
                running_training_ndcg += ndcg
            training_loss = running_training_loss / len(training_data_loader)
            training_auc = running_training_auc / len(training_data_loader)
            training_ndcg = running_training_ndcg / len(training_data_loader)
            entry = pd.DataFrame({'training_loss':[training_loss.item()],'training_auc':[training_auc.item()],'training_ndcg':[training_ndcg.item()],'validation_loss':[validation_loss.item()],'validation_auc':[validation_auc.item()],'validation_ndcg':[validation_ndcg.item()]})
            self.history = self.history.append(entry)
        print("Finished")
        self.load(self.experiment)
        self.evaluate()
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            data_lists = {'training':self.training_data,'validation':self.validation_data,'testing':self.testing_data}
            batches = {dataset: Batch.from_data_list(data_list).to(self.device) for dataset, data_list in data_lists.items()}
            outputs = {dataset:self.model.forward(batch) for dataset, batch in batches.items()}
            ranks = {dataset: torch.argsort(torch.argsort(output,descending=True,dim=0),dim=0) for dataset, output in outputs.items()}
            confusion_matrices = {'training':torch.zeros(2,2),'validation':torch.zeros(2,2),'testing':torch.zeros(2,2)}
            summary = {}
            points = []
            df = pd.read_csv(os.path.join(self.path,'data/raw/screened_compounds.csv'))
            for dataset in ['training','validation','testing']:
                for predicted, actual, id in zip(ranks[dataset], batches[dataset].y, batches[dataset].id):
                    confusion_matrices[dataset][int(predicted < self.ndcg_cutoff)][int(actual > 4.276)]+=1
                    mol = Chem.MolFromSmiles(df.iloc[int(id)]['SMILES'])
                    # Draw.MolToImage(mol,size=(200,200)).save(bytes, format="JPEG")
                    points.append({"x":predicted.item(),"y":actual.item(),"dataset":dataset,"name":df.iloc[int(id)]['Name']})
                precision = confusion_matrices[dataset][1][1]/(confusion_matrices[dataset][1][1] + confusion_matrices[dataset][1][0])
                specificity = confusion_matrices[dataset][0][0]/(confusion_matrices[dataset][0][0] + confusion_matrices[dataset][0][1])
                sensitivity = confusion_matrices[dataset][1][1]/(confusion_matrices[dataset][1][1] + confusion_matrices[dataset][0][1])
                ndcg = self.calculate_ndcg(outputs[dataset].view(-1),batches[dataset].y.view(-1))
                auc = self.calculate_auc(outputs[dataset].view(-1),batches[dataset].y.view(-1))
                summary[dataset] = {'precision':precision.item(),'sensitivity':sensitivity.item(),'specificity':specificity.item(),'pairwise-accuracy':auc.item(),'ndcg':ndcg.item()}
            pd.DataFrame(points).to_csv(self.path+"/points.csv")
            return pd.DataFrame(summary), pd.DataFrame(points)
    def train_one_epoch(self,minibatch):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model.forward(minibatch)
        loss = self.loss_func.forward(outputs,minibatch.y)
        auc = self.calculate_auc(outputs,minibatch.y)
        ndcg = self.calculate_ndcg(outputs.view(-1),minibatch.y.view(-1))
        loss.backward()
        self.optimizer.step()
        return loss, auc, ndcg
    def evaluate_one_epoch(self,batch):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(batch)
            loss = self.loss_func.forward(outputs,batch.y)
            auc = self.calculate_auc(outputs,batch.y)
            ndcg = self.calculate_ndcg(outputs.view(-1),batch.y.view(-1))
            idxs =torch.argsort(outputs.view(-1),descending=True)
            ranking = np.array([(idxs==i).nonzero() for i in range(list(outputs.size())[0])])
        return ranking, loss, auc, ndcg
    def screen(self,file):
        if file != {}:
            df = pd.read_csv(BytesIO(list(file.values())[0]['content']))
            df.to_csv(os.path.join(self.path,'data/raw/unscreened_compounds.csv'))
            unlabeled_data = UnscreenedCompounds(os.path.join(self.path,'data'),'unscreened_compounds.csv')
            batch = Batch.from_data_list(unlabeled_data).to(self.device)
            self.model.eval()
            outputs = self.model.forward(batch)
            indexes = torch.argsort(outputs,dim=0,descending=True).view(-1).cpu().numpy()
            df = df.reindex(indexes).reset_index(drop=True)
            df.index.name = "Rank"
            for i,row in df.iterrows():
                mol = Chem.MolFromSmiles(row['SMILES'])
                bytes = BytesIO()
                Draw.MolToImage(mol,size=(200,200)).save(bytes, format="JPEG")
                df.loc[i,'Image'] = f"<img src='data:image/png;base64, {base64.b64encode(bytes.getvalue()).decode('utf-8')}' width='200px' height='200px'></img>"
            return df
    def calculate_ndcg(self, outputs, scores):
        dcg = torch.sum((torch.pow(2.0,scores[torch.argsort(outputs,descending=True)[:self.ndcg_cutoff]]) - 1.0) / torch.log2(torch.arange(2,self.ndcg_cutoff+2,dtype=torch.float)).to(scores.device))
        idcg = torch.sum((torch.pow(2.0,torch.sort(scores,descending=True)[0][:self.ndcg_cutoff]) - 1.0) / torch.log2(torch.arange(2,self.ndcg_cutoff+2,dtype=torch.float)).to(scores.device))
        return dcg / idcg
    def calculate_auc(self,outputs,scores):
        correct = []
        for outputi, scorei in zip(outputs,scores):
            for outputj, scorej in zip(outputs, scores):
                true_order = int(scorei>scorej)
                outputed_prob = torch.sigmoid(outputi-outputj)
                if scorei!=scorej:
                    correct.append(int(torch.abs(true_order - outputed_prob) < 0.5))
        return torch.tensor(sum(correct)/len(correct)).to(self.device)
    def save(self):
        torch.save({'model_state_dict':self.model.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict()},
            os.path.join(self.path,'checkpoint.pt'))
    def load(self):
        checkpoint = torch.load(os.path.join(self.path,'checkpoint.pt'),map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    def graph(self,window):
        fig, ax = plt.subplots(constrained_layout=True)
        training_batch = Batch.from_data_list(self.training_data).to(self.device)
        validation_batch = Batch.from_data_list(self.validation_data).to(self.device)
        testing_batch = Batch.from_data_list(self.testing_data).to(self.device)
        training_ranking, training_loss, training_auc, training_ndcg = self.evaluate_one_epoch(training_batch)
        validation_ranking, validation_loss, validation_auc, validation_ndcg = self.evaluate_one_epoch(validation_batch)
        testing_ranking, testing_loss, testing_auc, testing_ndcg = self.evaluate_one_epoch(testing_batch)
        ax.set_xlabel('Predicted Ranking')
        ax.set_ylabel('Actual Score')
        training_scatter = ax.scatter(training_ranking,training_batch.y.cpu())
        validation_scatter = ax.scatter(validation_ranking,validation_batch.y.cpu())
        testing_scatter = ax.scatter(testing_ranking,testing_batch.y.cpu())
        dmso_score = ax.axhline(y=4.276,c='black',ls='dashed')
        ndcg_cutoff = ax.axvline(x=9.5,c='black',ls='dotted')
        ax.legend([training_scatter,validation_scatter,testing_scatter,dmso_score,ndcg_cutoff],['Training Data','Validation Data','Testing Data','DMSO Score','NDCG@k Cutoff'])
        plt.show()
    @classmethod
    def create(cls,name,model_params,data):
        if not os.path.exists(name):
            os.mkdir(name)
            with open(os.path.join(name,'config.json'),'w') as write_file:
                dump(model_params,write_file)
            print(os.path.join(name,'data/raw'))
            os.makedirs(os.path.join(name,'data/raw'))
            with open(os.path.join(name,'data/raw/screened_compounds.csv'),'wb') as write_file:
                write_file.write(data)
            return cls(name)
