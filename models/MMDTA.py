import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_add_pool as gap
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

device = torch.device('cuda')

class Sequence_Model(nn.Module):
    def __init__(self, in_channel, embedding_channel, med_channel, out_channel, kernel_size=3, stride=1, padding=1, relative_position=False, Heads=None, use_residue=False):
        super(Sequence_Model, self).__init__()
        self.in_channel = in_channel
        self.med_channel = med_channel
        self.out_channel = out_channel
        self.residue_in_channel = 64
        self.dim = '1d'
        self.dropout = 0.1
        self.relative_position = relative_position
        self.use_residue = use_residue
        
        self.emb = nn.Linear(in_channel, embedding_channel)
        self.dropout = nn.Dropout(self.dropout)
        
        self.layers = nn.Sequential(
            nn.Conv1d(embedding_channel, med_channel[1], kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(med_channel[1]),
            nn.LeakyReLU(),
            nn.Conv1d(med_channel[1], med_channel[2], kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(med_channel[2]),
            nn.LeakyReLU(),
            nn.Conv1d(med_channel[2], out_channel, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.AdaptiveMaxPool1d(1)
        )

    def forward(self, x):
        x = self.dropout(self.emb(x))
        x = self.layers(x.permute(0, 2, 1)).view(-1, 256)
        return x
        
class Flat_Model(nn.Module):
    def __init__(self, in_channel, med_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(Flat_Model, self).__init__()
        self.in_channel = in_channel
        self.med_channel = med_channel
        self.out_channel = out_channel
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, med_channel[1], kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(med_channel[1]),
            nn.LeakyReLU(),
            nn.Conv2d(med_channel[1], med_channel[2], kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(med_channel[2]),
            nn.LeakyReLU(),
            nn.Conv2d(med_channel[2], out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.AdaptiveMaxPool2d(1)
        )
        
    def forward(self, x):
        x = self.layers(x).view(-1, 256)
        return x

class GraphConv(nn.Module):
    def __init__(self, feature_dim, emb_dim, hidden_dim=32, output_dim=256, dropout=0.1):
        super(GraphConv, self).__init__()
        self.dropout = dropout
        self.hidden = hidden_dim
        self.emb = nn.Linear(feature_dim, emb_dim)
        self.cconv1 = SAGEConv(emb_dim, hidden_dim, aggr='sum')
        self.cconv2 = SAGEConv(hidden_dim, hidden_dim * 2, aggr='sum')
        self.cconv3 = SAGEConv(hidden_dim * 2, hidden_dim * 4, aggr='sum')
        
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.flat = nn.Linear(hidden_dim * 4, output_dim)

    
    def forward(self, data):
        # 获取小分子和蛋白质输入的结构信息
        compound_feature, compound_index, compound_batch = data.x, data.edge_index, data.batch
        # 对小分子进行卷积操作
        compound_feature = self.dropout(self.emb(compound_feature))

        compound_feature = self.cconv1(compound_feature, compound_index)
        compound_feature = self.relu(compound_feature)

        compound_feature = self.cconv2(compound_feature, compound_index)
        compound_feature = self.relu(compound_feature)

        compound_feature = self.cconv3(compound_feature, compound_index)

        # 对卷积后的小分子进行图的池化操作
        compound_feature = gap(compound_feature, compound_batch)

        compound_feature = self.flat(compound_feature)

        return compound_feature

# Our proposed model, fusion of the above models
class Multimodal_Affinity(nn.Module):
    def __init__(self, compound_sequence_channel, protein_sequence_channel, med_channel, out_channel, n_output=1):
        super(Multimodal_Affinity, self).__init__()
        self.embedding_dim = 128
        
        self.compound_sequence = Sequence_Model(compound_sequence_channel, self.embedding_dim, med_channel, out_channel, kernel_size=3, padding=1)
        self.protein_sequence = Sequence_Model(protein_sequence_channel, self.embedding_dim, med_channel, out_channel, kernel_size=3, padding=1)
    
        self.compound_stru = GraphConv(27, self.embedding_dim)
        self.protein_stru = Flat_Model(1, med_channel, out_channel)
        
        self.fc_input = 256 * 2

        self.sequence_sequence_fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, n_output)
        )
        self.sequence_graph_fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, n_output)
        )
        self.graph_sequence_fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, n_output)
        )
        self.graph_graph_fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, n_output)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, n_output)
        )
        self.reg = nn.Linear(5, 1)

    def forward(self, compound_sequence, compound_graph, protein_sequence, protein_graph):
        batch = compound_graph.batch
        c_sequence_feature = self.compound_sequence(compound_sequence)
        c_graph_feature = self.compound_stru(compound_graph)

        p_sequence_feature = self.protein_sequence(protein_sequence)
        p_graph_feature = self.protein_stru(protein_graph)
        
        # feature fusion
        c_x = c_sequence_feature + c_graph_feature
        p_x = p_sequence_feature + p_graph_feature
            
        early_x = torch.cat((c_x, p_x), 1)
        early_x = self.fc_layers(early_x)
        
        sequence_sequence_x = self.sequence_sequence_fc_layers(torch.cat((c_sequence_feature, p_sequence_feature), dim=1))
        stru_stru_x = self.graph_graph_fc_layers(torch.cat((c_graph_feature, p_graph_feature), dim=1))
        sequence_stru_x = self.sequence_graph_fc_layers(torch.cat((c_sequence_feature, p_graph_feature), dim=1))
        stru_sequence_x = self.graph_sequence_fc_layers(torch.cat((c_graph_feature, p_sequence_feature), dim=1))
        
        x = torch.cat((early_x, sequence_sequence_x, stru_stru_x, sequence_stru_x, stru_sequence_x), dim=1)

        x = self.reg(x)

        return x



if __name__ == '__main__':
    model = Sequence_Model(65, [128, 256], 256, use_residue=True)
    # print(model)
