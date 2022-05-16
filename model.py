
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random

RANDOM_SEED = 123
def seed_torch():
    random.seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(in_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, input, adj):
        h = torch.mm(input, self.W)

        f_1 = torch.matmul(h, self.a1)
        f_2 = torch.matmul(h, self.a2)
        e = self.leakyrelu(f_1 + f_2.transpose(0,1))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        return F.elu(h_prime), attention

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    
class MedML(nn.Module):
    def __init__(self, node_dim, hidden_dim, mlp_dim):
        super(MedML, self).__init__()
        self.gat_1 = GATLayer(node_dim+1, hidden_dim)
        self.gat_2 = GATLayer(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, mlp_dim)
        self.fc_2 = nn.Linear(mlp_dim, 1)
        self.node_embd = nn.Parameter(torch.randn(len(feat_list), node_dim))
        self.reset_parameters()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)

    def forward(self, g, adj, bm):
        g, att = self.gat_1(g, adj)
        g_vec = []
        att_vec = []
        for i in range(len(bm)):
            g_vec.append(torch.mean(g[bm[i, 0]:bm[i, 0]+bm[i, 1], :], dim=0))
            att_vec.append(att[bm[i, 0]:bm[i, 0]+bm[i, 1], bm[i, 0]:bm[i, 0]+bm[i, 1]].cpu().detach().numpy())
        g_vec = torch.stack(g_vec)
        pred = self.fc(g_vec)
        pred = torch.relu(pred)
        pred = self.fc_2(pred)
        pred = torch.sigmoid(pred)
        return pred, g_vec, att_vec
