import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
import torch.nn as nn

class GraphConvolutionBlock(torch.nn.Module):
    def __init__(self,input_channels,output_channels,pooling_type,pooling_ratio,dropout):
        super(GraphConvolutionBlock, self).__init__()
        self.drop = nn.Dropout(p=dropout)
        self.conv = gnn.GCNConv(input_channels, output_channels)
        self.pooling_type = pooling_type
        if pooling_type == "ASA":
            self.pool = gnn.ASAPooling(output_channels,ratio=pooling_ratio)
        elif pooling_type == "SAG":
            self.pool = gnn.SAGPooling(output_channels,ratio=pooling_ratio)
        elif pooling_type == "TopK":
            self.pool = gnn.TopKPooling(output_channels,ratio=pooling_ratio)
        elif self.pool == "None":
            pass
        else:
            raise ValueError(f'{pooling_type} Pooling is not a suppourted pooling method.')
    def forward(self, x, edge_index, edge_weight, batch):
        x = self.drop(x)
        x = F.leaky_relu(self.conv(x, edge_index, edge_weight))
        if self.pooling_type == "ASA":
            x, edge_index, edge_weight, batch, perm = self.pool(x, edge_index, edge_weight, batch)
        elif self.pooling_type == "SAG" or pooling_type == "TopK":
            x, edge_index, edge_weight, batch, perm, score = self.pool(x, edge_index, edge_weight, batch)
        return x, edge_index, edge_weight, batch
