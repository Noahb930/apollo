import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from graph_convolution_block import GraphConvolutionBlock

class HierarchicalPoolingNetwork(torch.nn.Module):
    def __init__(self,num_conv_blocks,vocab_length,num_feature_channels,pooling_type,pooling_ratio,dropout):
        super(HierarchicalPoolingNetwork, self).__init__()
        self.conv_blocks = nn.ModuleList([GraphConvolutionBlock(vocab_length,num_feature_channels,pooling_type,pooling_ratio,dropout)])
        self.conv_blocks.extend([GraphConvolutionBlock(num_feature_channels,num_feature_channels,pooling_type,pooling_ratio,dropout) for i in range(num_conv_blocks-1)])
        self.mlp = nn.Sequential(nn.Linear(num_feature_channels*2,num_feature_channels),
            nn.ReLU(),
            nn.Linear(num_feature_channels,int(num_feature_channels/2)),
            nn.ReLU(),
            nn.Linear(int(num_feature_channels/2),1))
    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        rs = []
        for conv_block in self.conv_blocks:
            x, edge_index, edge_weight, batch = conv_block(x, edge_index, edge_weight, batch)
            r = torch.cat((gnn.global_mean_pool(x, batch),gnn.global_max_pool(x, batch)),axis=1)
            rs.append(r)
        x = sum(rs)
        x = self.mlp(x)
        return x
