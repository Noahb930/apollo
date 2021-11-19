import torch
import torch.nn as nn

class LambadaRankLoss(nn.Module):
    def __init__(self,ndcg_cutoff):
        super().__init__()
        self.bce = nn.BCELoss(reduction='none')
        self.ndcg_cutoff = ndcg_cutoff
    def calculate_ndcg(self, outputs, scores):
        dcg = torch.sum((torch.pow(2.0,scores[torch.argsort(outputs,descending=True)[:self.ndcg_cutoff]]) - 1.0) / torch.log2(torch.arange(2,self.ndcg_cutoff+2,dtype=torch.float)).to(scores.device))
        idcg = torch.sum((torch.pow(2.0,torch.sort(scores,descending=True)[0][:self.ndcg_cutoff]) - 1.0) / torch.log2(torch.arange(2,self.ndcg_cutoff+2,dtype=torch.float)).to(scores.device))
        ndcg = dcg/idcg
        return ndcg
    def forward(self, outputs, scores):
        logits = torch.sigmoid(outputs - outputs.T)
        ones = torch.ones(scores.shape[0], scores.shape[0]).to(scores.device)
        zeros = torch.zeros(scores.shape[0], scores.shape[0]).to(scores.device)
        labels = torch.where(scores.view(-1,1) > scores.view(1,-1), ones,zeros)
        delta_ndcgs = torch.zeros(scores.shape[0],scores.shape[0]).to(scores.device)
        for i, outputi in enumerate(outputs):
            for j, outputj in enumerate(outputs):
                if outputi != outputj:
                    swapped = outputs.clone()
                    swapped[i] = outputj
                    swapped[j] = outputi
                    delta_ndcgs[i][j] = torch.abs(self.calculate_ndcg(outputs.view(-1),scores.view(-1)) - self.calculate_ndcg(swapped.view(-1),scores.view(-1)))
        loss = torch.sum(self.bce(logits, labels)*delta_ndcgs)/scores.shape[0]
        return loss