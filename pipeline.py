from pyspark.sql.functions import lower, col,lit
import pickle
import sklearn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

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

setK1 = 250
setK2 = 250
setK3 = 250
setK4 = 250

def print_metrics_binary(y_true, predictions, verbose=1):
    predictions = np.array(predictions)
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))
    if verbose:
        print("confusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])

    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, predictions[:, 1])
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

    if verbose:
        print("accuracy = {}".format(acc))
        print("precision class 0 = {}".format(prec0))
        print("precision class 1 = {}".format(prec1))
        print("recall class 0 = {}".format(rec0))
        print("recall class 1 = {}".format(rec1))
        print("AUC of ROC = {}".format(auroc))
        print("AUC of PRC = {}".format(auprc))
        print("min(+P, Se) = {}".format(minpse))

    return {"acc": acc,
            "prec0": prec0,
            "prec1": prec1,
            "rec0": rec0,
            "rec1": rec1,
            "auroc": auroc,
            "auprc": auprc,
            "minpse": minpse}

def balanced_sample_maker(X, y, ratio=1.15, random_seed=None):
    """
    #pos: #neg = ratio: 1
    """
    uniq_levels = np.unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if not random_seed is None:
        random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx

    # oversampling on observations of positive label
    sample_size = uniq_counts[0]
    over_sample_idx = np.random.choice(groupby_levels[1], size=round(sample_size * ratio), replace=True).tolist()
    balanced_copy_idx = groupby_levels[0] + over_sample_idx
    random.shuffle(balanced_copy_idx)

    return X[balanced_copy_idx, :], y[balanced_copy_idx]

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

feat_list = ['asthma', 'post-traumatic stress disorder', 'diabetes', 'anxiety', 'depression', 'malnutrition', 'malignancy', 'metabolic derangement', 'atrial septal defect', 'ventricular septal defect', 'gestation period, 21 weeks', 'apnea', 'dehydration', 'respiratory distress', 'obesity', 'chronic kidney disease', 'prematurity', 'failure to thrive', 'gestation period, 19 weeks', 'cerebral palsy', 'immunodeficiency', 'congenital pulmonary anomaly', 'gestation period, 28 weeks', 'hypoxemia', 'renal injury', 'gestation period, 10 weeks', 'congenital cardiac anomaly', 'gestation period, 25 weeks', 'gestation period, 27 weeks', 'tetralogy of fallot', 'gestation period, 16 weeks', 'gestation period, 30 weeks', 'gestation period, 31 weeks', 'weight loss', 'panic disorder', 'gestation period, 13 weeks', 'pulmonary hypertension', 'heart failure', 'chronic respiratory failure', 'gestation period, 22 weeks', 'gestation period, 20 weeks', 'gestation period, 29 weeks', 'gestation period, 18 weeks', 'gestation period, 8 weeks', 'acute heart failure', 'gestation period, 23 weeks', 'marasmus', 'gestation period, 12 weeks', 'gestation period, 11 weeks', 'gestation period, 14 weeks', 'gestation period, 15 weeks', 'gestation period, 26 weeks', 'on ventilator', 'gestation period, 24 weeks', 'secondary infection', 'kwashiorkor', 'gestation period, 17 weeks', 'high altitude', 'inborn error metabolism', 'congenital cystic adenomatoid malformation', 'gaucher disease', 'spo2', 'bmi', 'creatinine', 'gestational age', 'glucose', 'weight', 'a1c', 'pao2', 'gfr', 'myeloblast count', 'tumor resection', 'bone marrow biopsy', 'on ventilator', 'gastrostomy tube placement', 'central venous catheter', 'cardiac catheterization', 'dexamethasone', 'albuterol', 'levocarnitine', 'methotrexate', 'cyclophosphamide', 'tacrolimus', 'clonidine', 'insulin', 'vincristine', 'filgrastim', 'bleomycin', 'sirolimus', 'doxorubicin', 'daunorubicin', 'vinblastine']



def get_graph_batch(raw_x, node_embd, idx2name, edge_pair, M):
    drug_idx2name, meas_idx2name, cond_idx2name, proc_idx2name = idx2name
    M1, M2, M3, M4, M5, M6, M7 = M
    cond_vec = raw_x[:, :M1]
    cond_val = raw_x[:, M1:2*M1]
    cond_idx = np.argwhere(cond_vec == 1)
    cond_vec = np.array([[cond_idx[idx, 0], feat_list.index(cond_idx2name[cond_idx[idx, 1]])] for idx in range(len(cond_idx))])
    if cond_vec.shape[0] > 0:
        cond_vec = np.concatenate([cond_vec, np.expand_dims(cond_val[cond_idx[:, 0], cond_idx[:, 1]], axis=-1), ], axis=1)
    else:
        cond_vec = np.array([[-1, -1, -1]])
    
    meas_vec = raw_x[:, 2*M1:2*M1+M2]
    meas_val = raw_x[:, 2*M1+M2:2*M1+2*M2]
    meas_idx = np.argwhere(meas_vec == 1)
    meas_vec = np.array([[meas_idx[idx, 0], feat_list.index(meas_idx2name[meas_idx[idx, 1]])] for idx in range(len(meas_idx))])
    if meas_vec.shape[0] > 0:
        meas_vec = np.concatenate([meas_vec, np.expand_dims(meas_val[meas_idx[:, 0], meas_idx[:, 1]], axis=-1), ], axis=1)
    else:
        meas_vec = np.array([[-1, -1, -1]])
    
    proc_vec = raw_x[:, 2*M1+2*M2:2*M1+2*M2+M3]
    proc_val = raw_x[:, 2*M1+2*M2+M3:2*M1+2*M2+2*M3]
    proc_idx = np.argwhere(proc_vec == 1)
    proc_vec = np.array([[proc_idx[idx, 0], feat_list.index(proc_idx2name[proc_idx[idx, 1]])] for idx in range(len(proc_idx))])
    if proc_vec.shape[0] > 0:
        proc_vec = np.concatenate([proc_vec, np.expand_dims(proc_val[proc_idx[:, 0], proc_idx[:, 1]], axis=-1), ], axis=1)
    else:
        proc_vec = np.array([[-1, -1, -1]])
    
    drug_vec = raw_x[:, 2*M1+2*M2+2*M3:2*M1+2*M2+2*M3+M4]
    drug_val = raw_x[:, 2*M1+2*M2+2*M3+M4:2*M1+2*M2+2*M3+2*M4]
    drug_idx = np.argwhere(drug_vec == 1)
    drug_vec = np.array([[drug_idx[idx, 0], feat_list.index(drug_idx2name[drug_idx[idx, 1]])] for idx in range(len(drug_idx))])
    if drug_vec.shape[0] > 0:
        drug_vec = np.concatenate([drug_vec, np.expand_dims(drug_val[drug_idx[:, 0], drug_idx[:, 1]], axis=-1), ], axis=1)
    else:
        drug_vec = np.array([[-1, -1, -1]])

    graph_vec = np.concatenate([cond_vec, meas_vec, proc_vec, drug_vec], axis=0)
    # number of patient, id of feature, value of feature
    batch_g = []
    adj_list= []
    batch_size = 0
    batch_flag = []
    batch_node = []
    batch_info = torch.tensor(raw_x[:, 2*M1+2*M2+2*M3+2*M4:], dtype=torch.float32).to(device)
    for i in range(len(raw_x)):
        cur_p = graph_vec[graph_vec[:, 0] == i, 1:]
        if cur_p.shape[0] > 0:
            cur_nodes = np.array(cur_p[:, 0], dtype=np.int32)
            batch_node.append(cur_nodes)
            cur_val = node_embd[cur_nodes]
            cur_val = torch.cat([torch.tensor(cur_p[:, 1], dtype=torch.float32).unsqueeze(-1).to(device), cur_val], dim=-1)
            batch_g.append(cur_val)
            cur_adj = np.eye(cur_nodes.shape[0])
            for i in range(cur_nodes.shape[0]):
                for j in range(i+1, cur_nodes.shape[0]):
                    if (cur_nodes[i], cur_nodes[j]) in edge_pair:
                        cur_adj[i, j] = 1
                        cur_adj[j, i] = 1
            adj_list.append(cur_adj)
            batch_size += len(cur_nodes)
            batch_flag.append(1)
        else:
            batch_flag.append(0)
    batch_adj = torch.zeros((batch_size, batch_size), dtype=torch.int).to(device)
    cum_size = 0
    batch_mask = []

    for i in range(len(adj_list)):
        cur_size = len(adj_list[i])
        batch_adj[cum_size:cum_size+cur_size, cum_size:cum_size+cur_size] = torch.tensor(adj_list[i], dtype=torch.int).to(device)
        batch_mask.append([cum_size, cur_size])
        cum_size += cur_size
    batch_mask = torch.tensor(batch_mask, dtype=torch.int).to(device)
    batch_g = torch.cat(batch_g, dim=0)
    return batch_g, batch_adj, batch_mask, np.array(batch_flag), batch_info, batch_node
        

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.2c104028-530c-4b8a-8b27-89b1c7b23877"),
    cmb_process_full=Input(rid="ri.foundry.main.dataset.b8fddf32-e938-4144-8602-8deb3db06d1a"),
    feature_process=Input(rid="ri.foundry.main.dataset.55041bc9-f5d0-483a-b4a5-c13b0383c13b"),
    state_split=Input(rid="ri.foundry.main.dataset.7605bc24-43b1-4171-b6d8-93f9675b37fc")
)
def DV_split(state_split, feature_process, cmb_process_full):
    with state_split.filesystem().open('state2pid', 'rb') as f:
        state2pid = pickle.load(f)
    with state_split.filesystem().open('state_cnt', 'rb') as f:
        state_cnt = pickle.load(f)
    with cmb_process_full.filesystem().open('final_embd', 'rb') as f:
        embd_list = pickle.load(f)
    with cmb_process_full.filesystem().open('y_list', 'rb') as f:
        y_list = np.array(pickle.load(f))
    with feature_process.filesystem().open('plist', 'rb') as f:
        plist = pickle.load(f)
    
    dv1 = ['Connecticut','Connecticut','Maine','Massachusetts','New Hampshire','Rhode Island','Vermont']
    dv2 = ['New Jersey','New York','Pennsylvania']
    dv3 = ['Indiana','Illinois','Michigan','Ohio','Wisconsin']
    dv4 = ['Iowa','Nebraska', 'Kansas', 'North Dakota','Minnesota', 'South Dakota', 'Missouri']
    dv5 = ['Delaware', 'District of Columbia','Florida','Georgia','Maryland','North Carolina','South Carolina','Virginia','West Virginia']
    dv6 = ['Alabama','Kentucky','Mississippi','Tennessee']
    dv7 = ['Arkansas','Louisiana','Oklahoma','Texas']
    dv8 = ['Arizona','Colorado','Idaho','New Mexico','Montana','Utah','Nevada','Wyoming']
    dv9 = ['Alaska','California','Hawaii','Oregon','Washington']
    dv = [dv1, dv2,dv3,dv4,dv5,dv6,dv7,dv8,dv9]

    abbr2name = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
    }
    name2abbr = {abbr2name[k]:k for k in abbr2name}
    print(state_cnt)
    dv_cnt = {idx:0 for idx in range(9)}
    for each_state in state_cnt:
        for idx, each_dv in enumerate(dv):
            if abbr2name[each_state] in each_dv:
                dv_cnt[idx] += state_cnt[each_state]
    
    train_idx = []
    test_idx = []

    for idx, each_dv in enumerate(dv):
        cur_train_idx = []
        cur_test_idx = []
        for idx_2, each_dv_2 in enumerate(dv):
            if idx_2 == idx:
                continue
            for each_state in each_dv_2:
                if name2abbr[each_state] not in state2pid:
                    continue
                cur_plist = state2pid[name2abbr[each_state]]
                for each_p in cur_plist:
                    cur_idx = plist.index(each_p)
                    cur_train_idx.append(cur_idx)
        for each_state in each_dv:
            if name2abbr[each_state] not in state2pid:
                continue
            cur_plist = state2pid[name2abbr[each_state]]
            for each_p in cur_plist:
                cur_idx = plist.index(each_p)
                cur_test_idx.append(cur_idx)
        train_idx.append(cur_train_idx)
        test_idx.append(cur_test_idx)

        print('Finish %d/%d'%(idx,len(dv)))

    output = Transforms.get_output()
    with output.filesystem().open('train_idx', 'wb') as f:
        pickle.dump(train_idx, f)
    with output.filesystem().open('test_idx', 'wb') as f:
        pickle.dump(test_idx, f)
    with output.filesystem().open('dv', 'wb') as f:
        pickle.dump(dv, f)
    return output

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.53a790ac-191d-4614-9d8c-bcbe04da35d2"),
    cmb_process_full=Input(rid="ri.foundry.main.dataset.b8fddf32-e938-4144-8602-8deb3db06d1a"),
    feature_process=Input(rid="ri.foundry.main.dataset.55041bc9-f5d0-483a-b4a5-c13b0383c13b"),
    state_split=Input(rid="ri.foundry.main.dataset.7605bc24-43b1-4171-b6d8-93f9675b37fc")
)
def DV_split_plist(state_split, feature_process, cmb_process_full):
    with state_split.filesystem().open('state2pid', 'rb') as f:
        state2pid = pickle.load(f)
    with state_split.filesystem().open('state_cnt', 'rb') as f:
        state_cnt = pickle.load(f)
    with cmb_process_full.filesystem().open('final_embd', 'rb') as f:
        embd_list = pickle.load(f)
    with cmb_process_full.filesystem().open('y_list', 'rb') as f:
        y_list = np.array(pickle.load(f))
    with feature_process.filesystem().open('plist', 'rb') as f:
        plist = pickle.load(f)
    
    dv1 = ['Connecticut','Connecticut','Maine','Massachusetts','New Hampshire','Rhode Island','Vermont']
    dv2 = ['New Jersey','New York','Pennsylvania']
    dv3 = ['Indiana','Illinois','Michigan','Ohio','Wisconsin']
    dv4 = ['Iowa','Nebraska', 'Kansas', 'North Dakota','Minnesota', 'South Dakota', 'Missouri']
    dv5 = ['Delaware', 'District of Columbia','Florida','Georgia','Maryland','North Carolina','South Carolina','Virginia','West Virginia']
    dv6 = ['Alabama','Kentucky','Mississippi','Tennessee']
    dv7 = ['Arkansas','Louisiana','Oklahoma','Texas']
    dv8 = ['Arizona','Colorado','Idaho','New Mexico','Montana','Utah','Nevada','Wyoming']
    dv9 = ['Alaska','California','Hawaii','Oregon','Washington']
    dv = [dv1, dv2,dv3,dv4,dv5,dv6,dv7,dv8,dv9]

    abbr2name = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
    }
    name2abbr = {abbr2name[k]:k for k in abbr2name}
    print(state_cnt)
    dv_cnt = {idx:0 for idx in range(9)}
    for each_state in state_cnt:
        for idx, each_dv in enumerate(dv):
            if abbr2name[each_state] in each_dv:
                dv_cnt[idx] += state_cnt[each_state]
    
    train_idx = []
    test_idx = []

    for idx, each_dv in enumerate(dv):
        cur_train_idx = []
        cur_test_idx = []
        for idx_2, each_dv_2 in enumerate(dv):
            if idx_2 == idx:
                continue
            for each_state in each_dv_2:
                if name2abbr[each_state] not in state2pid:
                    continue
                cur_plist = state2pid[name2abbr[each_state]]
                cur_train_idx += cur_plist
        for each_state in each_dv:
            if name2abbr[each_state] not in state2pid:
                continue
            cur_plist = state2pid[name2abbr[each_state]]
            cur_test_idx += cur_plist
        train_idx.append(cur_train_idx)
        test_idx.append(cur_test_idx)

        print('Finish %d/%d'%(idx,len(dv)))

    output = Transforms.get_output()
    with output.filesystem().open('train_idx', 'wb') as f:
        pickle.dump(train_idx, f)
    with output.filesystem().open('test_idx', 'wb') as f:
        pickle.dump(test_idx, f)
    return output

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c68f658f-c149-499c-9b7e-c40f6c57ddf9"),
    cond_output=Input(rid="ri.foundry.main.dataset.ed7e653b-75ff-488e-bd8e-18b79cac26ed"),
    drug_output=Input(rid="ri.foundry.main.dataset.8d0c0780-88a7-4100-8851-e8240db21d3a"),
    feature_process=Input(rid="ri.foundry.main.dataset.55041bc9-f5d0-483a-b4a5-c13b0383c13b"),
    meas_output=Input(rid="ri.foundry.main.dataset.b293ed4e-716c-4539-b9ff-addcfc3ccb67"),
    proc_output=Input(rid="ri.foundry.main.dataset.aa5b29a2-62d8-4d47-a0ba-a5b1cac111ce")
)
def build_graph(cond_output, drug_output, proc_output, meas_output, feature_process):
    with feature_process.filesystem().open('M', 'rb') as f:
        M1, M2, M3, M4, M5, M6, M7 = pickle.load(f)
    drug_name2idx = {}
    for idx, row in drug_output.iterrows():
        drug_name2idx[row[0]] = row[1]+M1+M2+M3

    meas_name2idx = {}
    for idx, row in meas_output.iterrows():
        meas_name2idx[row[0]] = row[1]+M1
        
    cond_name2idx = {}
    for idx, row in cond_output.iterrows():
        cond_name2idx[row[0]] = row[1]

    proc_name2idx = {}
    for idx, row in proc_output.iterrows():
        proc_name2idx[row[0]] = row[1]+M1+M2

    drug_idx2name = {}
    for idx, row in drug_output.iterrows():
        drug_idx2name[row[1]] = row[0]

    meas_idx2name = {}
    for idx, row in meas_output.iterrows():
        meas_idx2name[row[1]] = row[0]
        
    cond_idx2name = {}
    for idx, row in cond_output.iterrows():
        cond_idx2name[row[1]] = row[0]

    proc_idx2name = {}
    for idx, row in proc_output.iterrows():
        proc_idx2name[row[1]] = row[0]

    sort_proc = {k: v for k, v in sorted(proc_name2idx.items(), key=lambda item: item[1])}
    sort_cond = {k: v for k, v in sorted(cond_name2idx.items(), key=lambda item: item[1])}
    sort_drug = {k: v for k, v in sorted(drug_name2idx.items(), key=lambda item: item[1])}
    sort_meas = {k: v for k, v in sorted(meas_name2idx.items(), key=lambda item: item[1])}
    feat_list = list(sort_cond.keys())+list(sort_meas.keys())+list(sort_proc.keys())+list(sort_drug.keys())
    print(feat_list)

    map_dict = {}

    #1st to 2nd
    map_dict[feat_list.index('hypoxemia')] = []
    map_dict[feat_list.index('hypoxemia')].append(feat_list.index('chronic respiratory failure'))
    map_dict[feat_list.index('hypoxemia')].append(feat_list.index('obesity'))
    map_dict[feat_list.index('hypoxemia')].append(feat_list.index('congenital cardiac anomaly'))
    map_dict[feat_list.index('hypoxemia')].append(feat_list.index('asthma'))
    map_dict[feat_list.index('hypoxemia')].append(feat_list.index('apnea'))
    map_dict[feat_list.index('hypoxemia')].append(feat_list.index('congenital pulmonary anomaly'))
    map_dict[feat_list.index('hypoxemia')].append(feat_list.index('high altitude'))

    map_dict[feat_list.index('respiratory distress')] = []
    map_dict[feat_list.index('respiratory distress')].append(feat_list.index('chronic respiratory failure'))
    map_dict[feat_list.index('respiratory distress')].append(feat_list.index('obesity'))
    map_dict[feat_list.index('respiratory distress')].append(feat_list.index('congenital cardiac anomaly'))
    map_dict[feat_list.index('respiratory distress')].append(feat_list.index('asthma'))
    map_dict[feat_list.index('respiratory distress')].append(feat_list.index('congenital pulmonary anomaly'))
    map_dict[feat_list.index('respiratory distress')].append(feat_list.index('high altitude'))
    map_dict[feat_list.index('respiratory distress')].append(feat_list.index('anxiety'))

    map_dict[feat_list.index('dehydration')] = []
    map_dict[feat_list.index('dehydration')].append(feat_list.index('chronic respiratory failure'))
    map_dict[feat_list.index('dehydration')].append(feat_list.index('malnutrition'))
    map_dict[feat_list.index('dehydration')].append(feat_list.index('diabetes'))
    map_dict[feat_list.index('dehydration')].append(feat_list.index('congenital cardiac anomaly'))
    map_dict[feat_list.index('dehydration')].append(feat_list.index('congenital pulmonary anomaly'))

    map_dict[feat_list.index('secondary infection')] = []
    map_dict[feat_list.index('secondary infection')].append(feat_list.index('chronic respiratory failure'))
    map_dict[feat_list.index('secondary infection')].append(feat_list.index('malnutrition'))
    map_dict[feat_list.index('secondary infection')].append(feat_list.index('central venous catheter'))
    map_dict[feat_list.index('secondary infection')].append(feat_list.index('immunodeficiency'))
    map_dict[feat_list.index('secondary infection')].append(feat_list.index('congenital cardiac anomaly'))
    map_dict[feat_list.index('secondary infection')].append(feat_list.index('congenital pulmonary anomaly'))

    map_dict[feat_list.index('acute heart failure')] = []
    map_dict[feat_list.index('acute heart failure')].append(feat_list.index('congenital cardiac anomaly'))
    map_dict[feat_list.index('acute heart failure')].append(feat_list.index('malnutrition'))
    map_dict[feat_list.index('acute heart failure')].append(feat_list.index('apnea'))
    map_dict[feat_list.index('acute heart failure')].append(feat_list.index('chronic respiratory failure'))

    map_dict[feat_list.index('metabolic derangement')] = []
    map_dict[feat_list.index('metabolic derangement')].append(feat_list.index('inborn error metabolism'))
    map_dict[feat_list.index('metabolic derangement')].append(feat_list.index('diabetes'))
    map_dict[feat_list.index('metabolic derangement')].append(feat_list.index('renal injury'))

    map_dict[feat_list.index('renal injury')] = []
    map_dict[feat_list.index('renal injury')].append(feat_list.index('gfr'))
    map_dict[feat_list.index('renal injury')].append(feat_list.index('creatinine'))

    map_dict[feat_list.index('chronic respiratory failure')] = []
    map_dict[feat_list.index('chronic respiratory failure')].append(feat_list.index('cerebral palsy'))
    map_dict[feat_list.index('chronic respiratory failure')].append(feat_list.index('on ventilator'))

    map_dict[feat_list.index('congenital cardiac anomaly')] = []
    map_dict[feat_list.index('congenital cardiac anomaly')].append(feat_list.index('atrial septal defect'))
    map_dict[feat_list.index('congenital cardiac anomaly')].append(feat_list.index('ventricular septal defect'))
    map_dict[feat_list.index('congenital cardiac anomaly')].append(feat_list.index('pulmonary hypertension'))
    map_dict[feat_list.index('congenital cardiac anomaly')].append(feat_list.index('tetralogy of fallot'))
    map_dict[feat_list.index('congenital cardiac anomaly')].append(feat_list.index('cardiac catheterization'))

    map_dict[feat_list.index('apnea')] = []
    map_dict[feat_list.index('apnea')].append(feat_list.index('prematurity'))
    map_dict[feat_list.index('apnea')].append(feat_list.index('on ventilator'))
    map_dict[feat_list.index('apnea')].append(feat_list.index('gestational age'))

    map_dict[feat_list.index('congenital pulmonary anomaly')] = []
    map_dict[feat_list.index('congenital pulmonary anomaly')].append(feat_list.index('pulmonary hypertension'))
    #map_dict[feat_list.index('congenital pulmonary anomaly')].append(feat_list.index('azygos lobe lung'))
    map_dict[feat_list.index('congenital pulmonary anomaly')].append(feat_list.index('congenital cystic adenomatoid malformation'))
    #map_dict[feat_list.index('congenital pulmonary anomaly')].append(feat_list.index('congenital pulmonary airway malformation'))

    map_dict[feat_list.index('anxiety')] = []
    map_dict[feat_list.index('anxiety')].append(feat_list.index('panic disorder'))
    map_dict[feat_list.index('anxiety')].append(feat_list.index('depression'))
    map_dict[feat_list.index('anxiety')].append(feat_list.index('post-traumatic stress disorder'))
    #map_dict[feat_list.index('anxiety')].append(feat_list.index('ssri'))
    #map_dict[feat_list.index('anxiety')].append(feat_list.index('snri'))
    #map_dict[feat_list.index('anxiety')].append(feat_list.index('atypical antipsychotic'))

    map_dict[feat_list.index('malnutrition')] = []
    #map_dict[feat_list.index('malnutrition')].append(feat_list.index('nutritional supplement'))
    map_dict[feat_list.index('malnutrition')].append(feat_list.index('gastrostomy tube placement'))
    map_dict[feat_list.index('malnutrition')].append(feat_list.index('failure to thrive'))
    map_dict[feat_list.index('malnutrition')].append(feat_list.index('weight loss'))
    map_dict[feat_list.index('malnutrition')].append(feat_list.index('weight'))
    map_dict[feat_list.index('malnutrition')].append(feat_list.index('kwashiorkor'))
    map_dict[feat_list.index('malnutrition')].append(feat_list.index('cyclophosphamide'))
    map_dict[feat_list.index('malnutrition')].append(feat_list.index('marasmus'))

    map_dict[feat_list.index('immunodeficiency')] = []
    map_dict[feat_list.index('immunodeficiency')].append(feat_list.index('filgrastim'))
    #map_dict[feat_list.index('immunodeficiency')].append(feat_list.index('granulocyte colony stimulating factor'))
    map_dict[feat_list.index('immunodeficiency')].append(feat_list.index('sirolimus'))
    map_dict[feat_list.index('immunodeficiency')].append(feat_list.index('tacrolimus'))
    #map_dict[feat_list.index('immunodeficiency')].append(feat_list.index('chemotherapeutic'))
    map_dict[feat_list.index('immunodeficiency')].append(feat_list.index('vincristine'))
    map_dict[feat_list.index('immunodeficiency')].append(feat_list.index('methotrexate'))
    map_dict[feat_list.index('immunodeficiency')].append(feat_list.index('daunorubicin'))
    map_dict[feat_list.index('immunodeficiency')].append(feat_list.index('bleomycin'))
    map_dict[feat_list.index('immunodeficiency')].append(feat_list.index('dexamethasone'))
    map_dict[feat_list.index('immunodeficiency')].append(feat_list.index('vinblastine'))
    map_dict[feat_list.index('immunodeficiency')].append(feat_list.index('doxorubicin'))
    #map_dict[feat_list.index('immunodeficiency')].append(feat_list.index('steroid'))
    #map_dict[feat_list.index('immunodeficiency')].append(feat_list.index('cortisol'))
    map_dict[feat_list.index('immunodeficiency')].append(feat_list.index('chronic kidney disease'))
    map_dict[feat_list.index('immunodeficiency')].append(feat_list.index('malignancy'))

    map_dict[feat_list.index('obesity')] = []
    map_dict[feat_list.index('obesity')].append(feat_list.index('bmi'))

    map_dict[feat_list.index('inborn error metabolism')] = []
    #map_dict[feat_list.index('inborn error metabolism')].append(feat_list.index('free fatty acid defect'))
    map_dict[feat_list.index('inborn error metabolism')].append(feat_list.index('gaucher disease'))
    #map_dict[feat_list.index('inborn error metabolism')].append(feat_list.index('fabre disease'))

    map_dict[feat_list.index('asthma')] = []
    #map_dict[feat_list.index('asthma')].append(feat_list.index('inhaled beta agonist'))
    map_dict[feat_list.index('asthma')].append(feat_list.index('albuterol'))

    map_dict[feat_list.index('diabetes')] = []
    map_dict[feat_list.index('diabetes')].append(feat_list.index('insulin'))
    map_dict[feat_list.index('diabetes')].append(feat_list.index('glucose'))
    map_dict[feat_list.index('diabetes')].append(feat_list.index('a1c'))

    #cond to med
    map_dict[feat_list.index('cerebral palsy')] = []
    map_dict[feat_list.index('cerebral palsy')].append(feat_list.index('clonidine'))

    map_dict[feat_list.index('malignancy')] = []
    map_dict[feat_list.index('malignancy')].append(feat_list.index('bone marrow biopsy'))
    map_dict[feat_list.index('malignancy')].append(feat_list.index('tumor resection'))

    #map_dict[feat_list.index('free fatty acid defect')] = []
    #map_dict[feat_list.index('free fatty acid defect')].append(feat_list.index('levocarnitine'))

    md = map_dict
    edge_pair = []
    src = []
    dst = []
    for each_src in md:
        for each_dst in md[each_src]:
            if (each_src,each_dst) not in edge_pair:
                edge_pair.append((each_src,each_dst))
                src.append(each_src)
                dst.append(each_dst)
                edge_pair.append((each_dst,each_src))
                src.append(each_dst)
                dst.append(each_src)        

    output = Transforms.get_output()
    with output.filesystem().open('edge_pair', 'wb') as f:
        pickle.dump(edge_pair, f)
    with output.filesystem().open('name2idx', 'wb') as f:
        pickle.dump([drug_name2idx, meas_name2idx, cond_name2idx, proc_name2idx], f)
    with output.filesystem().open('idx2name', 'wb') as f:
        pickle.dump([drug_idx2name, meas_idx2name, cond_idx2name, proc_idx2name], f)
    return output

@transform_pandas(
    Output(rid="ri.vector.main.execute.115cc428-bf4c-4920-ae66-7fd286e39209"),
    cond_f_step1=Input(rid="ri.foundry.main.dataset.12f46eea-b019-4abf-80ce-2fd722642f6d"),
    drug_f_step1=Input(rid="ri.foundry.main.dataset.f88a25cb-9da5-4a5d-9368-d663e830a274"),
    meas_f_step1=Input(rid="ri.foundry.main.dataset.10cd40f3-70c0-43e2-a92a-561f0d074e18"),
    person_level_f=Input(rid="ri.foundry.main.dataset.16401b3e-063d-4f8a-8493-36c631fecb27"),
    proc_f_step1=Input(rid="ri.foundry.main.dataset.eb5b5c83-8a32-4a25-be3e-15628097ef47"),
    visit_seq_vec=Input(rid="ri.foundry.main.dataset.ec0df1ca-df03-431d-ac19-3aed08ef0bcc"),
    visit_type_vec=Input(rid="ri.foundry.main.dataset.f2d70200-9d12-4cc6-a1c5-7fd33a4c5f57")
)
def case_info(visit_type_vec, visit_seq_vec, person_level_f, drug_f_step1, proc_f_step1, cond_f_step1, meas_f_step1):
    with person_level_f.filesystem().open('person_info_dict', 'rb') as f:
        person_info_dict = pickle.load(f)
    with visit_type_vec.filesystem().open('patient_visit_cnt_vec', 'rb') as f:
        person_visit_cnt_vec = pickle.load(f)
    with visit_seq_vec.filesystem().open('person_visit_seq_vec', 'rb') as f:
        person_visit_seq_vec = pickle.load(f)
    idx = '3906519179678687930'
    print(drug_f_step1.where("person_id = %s"%idx).toPandas())
    print(proc_f_step1.where("person_id = %s"%idx).toPandas())
    print(cond_f_step1.where("person_id = %s"%idx).toPandas())
    print(meas_f_step1.where("person_id = %s"%idx).toPandas())
    print(person_info_dict[idx])
    print(person_visit_seq_vec[idx])
    print(person_visit_cnt_vec[idx])

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.a56a7684-c8ae-4c05-bed7-c3bbbcedc309"),
    Cond_cmb_fstep1=Input(rid="ri.foundry.main.dataset.dc1af232-bd74-4302-8018-cda031b12663"),
    cond_f_step1=Input(rid="ri.foundry.main.dataset.12f46eea-b019-4abf-80ce-2fd722642f6d"),
    condition_mapping=Input(rid="ri.foundry.main.dataset.a64152a8-3c4e-48e2-b16f-00230416b495")
)
def cmb_cond(cond_f_step1, condition_mapping, Cond_cmb_fstep1):
    f1 =  Cond_cmb_fstep1.toPandas()
    g1 = cond_f_step1.toPandas()
    dm = condition_mapping.toPandas()
    
    data_feature = set(f1.origin_concept.unique())
    graph_feature = set(dm.concept_name.unique())
    graph_category = set(g1.origin_concept.unique())
    print ('Data driven features:%d'%len(data_feature))
    print ('Graph categories:%d'%len(graph_category))
    print('Overlapping features:%d'%len(data_feature.intersection(graph_feature)))
    final_feature = data_feature.difference(graph_feature).union(graph_category)
    print('Final features:%d'%len(final_feature))
    print(final_feature)
    # collect the dict
    Dict = {}
    for idx, item in enumerate(final_feature):
        Dict[item] = idx

    #final_df = f1
    final_df = f1.append(g1, ignore_index=True)
    final_df = final_df[final_df['origin_concept'].isin(final_feature)]

    # person_dict
    person_cond_dict = {}
    for idx, (person, tmp) in enumerate(final_df.groupby('person_id')):
        binary_vec = [0 for _ in range(len(Dict))]
        count_vec = [0 for _ in range(len(Dict))]
        for _, fname, cnt in tmp.values:
            if cnt>=2:
                binary_vec[Dict[fname]] = 1
                count_vec[Dict[fname]] = cnt
        person_cond_dict[person] = [binary_vec, count_vec]
    
    # output
    output = Transforms.get_output()
    with output.filesystem().open('person_cond_dict_cmb', 'wb') as f:
        pickle.dump(person_cond_dict, f)

    print (len(person_cond_dict))

    return output

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.f29206af-f0b2-4f03-9d05-4400888b6ba4"),
    Drug_cmb_fstep1=Input(rid="ri.foundry.main.dataset.23c8d01e-5e9e-46f2-967f-712fc63456b5"),
    drug_f_step1=Input(rid="ri.foundry.main.dataset.f88a25cb-9da5-4a5d-9368-d663e830a274"),
    drug_mapping=Input(rid="ri.foundry.main.dataset.c9b47a81-3ff8-4c57-b5ad-b994b365ca16")
)
def cmb_drug(drug_f_step1, drug_mapping, Drug_cmb_fstep1):
    f1 =  drug_f_step1.toPandas()
    f2 = Drug_cmb_fstep1.toPandas()
    dm = drug_mapping.toPandas()
    
    data_feature = set(f2.origin_concept.unique())
    graph_feature = set(dm.concept_name.unique())
    graph_category = set(f1.origin_concept.unique())

    print ('Data driven features:%d'%len(data_feature))
    print ('Graph categories:%d'%len(graph_category))
    print('Overlapping features:%d'%len(data_feature.intersection(graph_feature)))
    final_feature = data_feature.difference(graph_feature).union(graph_category)
    print('Final features:%d'%len(final_feature))
    print(final_feature)
    # collect the dict
    Dict = {}
    for idx, item in enumerate(final_feature):
        Dict[item] = idx

    final_df = f1.append(f2, ignore_index=True)
    final_df = final_df[final_df['origin_concept'].isin(final_feature)]

    # person_dict
    person_drug_dict = {}
    for idx, (person, tmp) in enumerate(final_df.groupby('person_id')):
        binary_vec = [0 for _ in range(len(Dict))]
        count_vec = [0 for _ in range(len(Dict))]
        for _, fname, cnt in tmp.values:
            if cnt >= 2:
                binary_vec[Dict[fname]] = 1
                count_vec[Dict[fname]] = cnt
        person_drug_dict[person] = [binary_vec, count_vec]
    
    # output
    output = Transforms.get_output()
    with output.filesystem().open('person_drug_dict_cmb', 'wb') as f:
        pickle.dump(person_drug_dict, f)

    print (len(person_drug_dict))

    return output
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d16df23f-fb7c-4084-8281-afdf30b0b994"),
    Meas_cmb_fstep1=Input(rid="ri.foundry.main.dataset.d89f5662-22d9-4e93-8f7b-b1cba86dd147"),
    meas_f_step1=Input(rid="ri.foundry.main.dataset.10cd40f3-70c0-43e2-a92a-561f0d074e18"),
    meas_mapping=Input(rid="ri.foundry.main.dataset.01377f9c-b83e-4dba-9098-330c87d08a81")
)
def cmb_meas(meas_f_step1, meas_mapping, Meas_cmb_fstep1):
    f1 =  meas_f_step1.toPandas()
    f2 = Meas_cmb_fstep1.toPandas()
    dm = meas_mapping.toPandas()
    
    data_feature = set(f2.origin_concept.unique())
    graph_feature = set(dm.concept_name.unique())
    graph_category = set(f1.origin_concept.unique())
    print ('Data driven features:%d'%len(data_feature))
    print ('Graph categories:%d'%len(graph_category))
    print('Overlapping features:%d'%len(data_feature.intersection(graph_feature)))
    final_feature = data_feature.difference(graph_feature).union(graph_category)
    print('Final features:%d'%len(final_feature))
    print(data_feature)
    print(graph_category)
    print(graph_feature)
    print(final_feature)
    # collect the dict
    Dict = {}
    for idx, item in enumerate(final_feature):
        Dict[item] = idx

    final_df = f1.append(f2, ignore_index=True)
    final_df = final_df[final_df['origin_concept'].isin(final_feature)]

    # person_dict
    person_meas_dict = {}
    for idx, (person, tmp) in enumerate(final_df.groupby('person_id')):
        binary_vec = [0 for _ in range(len(Dict))]
        count_vec = [0 for _ in range(len(Dict))]
        for _, fname, cnt in tmp.values:
            binary_vec[Dict[fname]] = 1
            count_vec[Dict[fname]] = cnt
        person_meas_dict[person] = [binary_vec, count_vec]
    
    # output
    output = Transforms.get_output()
    with output.filesystem().open('person_meas_dict_cmb', 'wb') as f:
        pickle.dump(person_meas_dict, f)

    print (len(person_meas_dict))

    return output

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e2e4fbc2-3faf-4e29-b127-d249fa643b4c"),
    Proc_cmb_fstep1=Input(rid="ri.foundry.main.dataset.9ea25717-3525-4677-8bac-b34e97d31f96"),
    proc_f_step1=Input(rid="ri.foundry.main.dataset.eb5b5c83-8a32-4a25-be3e-15628097ef47"),
    proc_mapping=Input(rid="ri.foundry.main.dataset.8742f851-61f0-4ab4-84e7-22afdf6dfae0")
)
def cmb_proc(proc_f_step1, proc_mapping, Proc_cmb_fstep1):
    f1 =  proc_f_step1.toPandas()
    f2 = Proc_cmb_fstep1.toPandas()
    dm = proc_mapping.toPandas()
    
    data_feature = set(f2.origin_concept.unique())
    graph_feature = set(dm.concept_name.unique())
    graph_category = set(f1.origin_concept.unique())
    print ('Data driven features:%d'%len(data_feature))
    print ('Graph categories:%d'%len(graph_category))
    print('Overlapping features:%d'%len(data_feature.intersection(graph_feature)))
    final_feature = data_feature.difference(graph_feature).union(graph_category)
    print('Final features:%d'%len(final_feature))
    print(final_feature)
    # collect the dict
    Dict = {}
    for idx, item in enumerate(final_feature):
        Dict[item] = idx

    final_df = f1.append(f2, ignore_index=True)
    final_df = final_df[final_df['origin_concept'].isin(final_feature)]

    # person_dict
    person_proc_dict = {}
    for idx, (person, tmp) in enumerate(final_df.groupby('person_id')):
        binary_vec = [0 for _ in range(len(Dict))]
        count_vec = [0 for _ in range(len(Dict))]
        for _, fname, cnt in tmp.values:
            if cnt >= 2:
                binary_vec[Dict[fname]] = 1
                count_vec[Dict[fname]] = cnt
        person_proc_dict[person] = [binary_vec, count_vec]
    
    # output
    output = Transforms.get_output()
    with output.filesystem().open('person_proc_dict_cmb', 'wb') as f:
        pickle.dump(person_proc_dict, f)

    print (len(person_proc_dict))

    return output
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.9468f1a2-ed3e-4ae3-b7a3-12ad3d44b6a5"),
    cmb_cond=Input(rid="ri.foundry.main.dataset.a56a7684-c8ae-4c05-bed7-c3bbbcedc309"),
    cmb_drug=Input(rid="ri.foundry.main.dataset.f29206af-f0b2-4f03-9d05-4400888b6ba4"),
    cmb_meas=Input(rid="ri.foundry.main.dataset.d16df23f-fb7c-4084-8281-afdf30b0b994"),
    cmb_proc=Input(rid="ri.foundry.main.dataset.e2e4fbc2-3faf-4e29-b127-d249fa643b4c"),
    person_level_f=Input(rid="ri.foundry.main.dataset.16401b3e-063d-4f8a-8493-36c631fecb27"),
    split_dataset=Input(rid="ri.foundry.main.dataset.a6c4b4c5-c9a7-46a0-a28f-13ef917c9262"),
    task_1_goldstandard=Input(rid="ri.foundry.main.dataset.78c4cc3c-d0d9-4997-a851-62e1fb5f3e6b"),
    visit_seq_vec=Input(rid="ri.foundry.main.dataset.ec0df1ca-df03-431d-ac19-3aed08ef0bcc"),
    visit_type_vec=Input(rid="ri.foundry.main.dataset.f2d70200-9d12-4cc6-a1c5-7fd33a4c5f57")
)
def cmb_process_2(person_level_f, task_1_goldstandard, visit_type_vec, visit_seq_vec, cmb_drug, cmb_proc, cmb_meas, cmb_cond, split_dataset):
    with cmb_cond.filesystem().open('person_cond_dict_cmb', 'rb') as f:
        person_cond_dict = pickle.load(f)
    with cmb_meas.filesystem().open('person_meas_dict_cmb', 'rb') as f:
        person_measure_dict = pickle.load(f)
    with cmb_proc.filesystem().open('person_proc_dict_cmb', 'rb') as f:
        person_prod_dict = pickle.load(f)
    with cmb_drug.filesystem().open('person_drug_dict_cmb', 'rb') as f:
        person_drug_dict = pickle.load(f)
    with person_level_f.filesystem().open('person_info_dict', 'rb') as f:
        person_info_dict = pickle.load(f)
    with visit_type_vec.filesystem().open('patient_visit_cnt_vec', 'rb') as f:
        person_visit_cnt_vec = pickle.load(f)
    with visit_seq_vec.filesystem().open('person_visit_seq_vec', 'rb') as f:
        person_visit_seq_vec = pickle.load(f)
    with split_dataset.filesystem().open('p_train', 'rb') as f:
        p_train = pickle.load(f)
    with split_dataset.filesystem().open('p_val', 'rb') as f:
        p_val = pickle.load(f)
    with split_dataset.filesystem().open('p_test', 'rb') as f:
        p_test = pickle.load(f)

    task2gt = task_1_goldstandard[['person_id','outcome']].toPandas()
    print ('train size:', len(task2gt))

    print ('files loaded')

    # for the training data
    X, y = [], []
    M1 = len(list(person_cond_dict.values())[0][0])
    M2 = len(list(person_measure_dict.values())[0][0])
    M3 = len(list(person_prod_dict.values())[0][0])
    M4 = len(list(person_drug_dict.values())[0][0])
    M5 = len(list(person_info_dict.values())[0])
    M6 = len(list(person_visit_cnt_vec.values())[0])
    M7 = len(list(person_visit_seq_vec.values())[0])

    #_, task2gt = train_test_split(task2gt, test_size=10000, random_state=RANDOM_SEED, stratify=task2gt['outcome'])
    cmb_list = []

    for i in range(len(p_train)):
        person = p_train[i]
        if person in person_cond_dict:
            f1 = person_cond_dict[person]
        else:
            f1 = [[0 for _ in range(M1)], [0 for _ in range(M1)]]
        if person in person_measure_dict:
            f2 = person_measure_dict[person]
        else:
            f2 = [[0 for _ in range(M2)], [0 for _ in range(M2)]]
        if person in person_prod_dict:
            f3 = person_prod_dict[person]
        else:
            f3 = [[0 for _ in range(M3)], [0 for _ in range(M3)]]
        if person in person_drug_dict:
            f4 = person_drug_dict[person]
        else:
            f4 = [[0 for _ in range(M4)], [0 for _ in range(M4)]]
        tmp = f1[0] + f2[1] + f3[1] + f4[1]
        cmb_list.append(tmp)
    for i in range(len(p_val)):
        person = p_val[i]
        if person in person_cond_dict:
            f1 = person_cond_dict[person]
        else:
            f1 = [[0 for _ in range(M1)], [0 for _ in range(M1)]]
        if person in person_measure_dict:
            f2 = person_measure_dict[person]
        else:
            f2 = [[0 for _ in range(M2)], [0 for _ in range(M2)]]
        if person in person_prod_dict:
            f3 = person_prod_dict[person]
        else:
            f3 = [[0 for _ in range(M3)], [0 for _ in range(M3)]]
        if person in person_drug_dict:
            f4 = person_drug_dict[person]
        else:
            f4 = [[0 for _ in range(M4)], [0 for _ in range(M4)]]
        tmp = f1[0] + f2[1] + f3[1] + f4[1]
        cmb_list.append(tmp)
    for i in range(len(p_test)):
        person = p_test[i]
        if person in person_cond_dict:
            f1 = person_cond_dict[person]
        else:
            f1 = [[0 for _ in range(M1)], [0 for _ in range(M1)]]
        if person in person_measure_dict:
            f2 = person_measure_dict[person]
        else:
            f2 = [[0 for _ in range(M2)], [0 for _ in range(M2)]]
        if person in person_prod_dict:
            f3 = person_prod_dict[person]
        else:
            f3 = [[0 for _ in range(M3)], [0 for _ in range(M3)]]
        if person in person_drug_dict:
            f4 = person_drug_dict[person]
        else:
            f4 = [[0 for _ in range(M4)], [0 for _ in range(M4)]]
        tmp = f1[0] + f2[1] + f3[1] + f4[1]
        cmb_list.append(tmp)
    cmb_list = np.array(cmb_list)
    # for each_feat in range(cmb_list.shape[1]):
    #     mean = np.mean(cmb_list[:, each_feat])
    #     std = np.std(cmb_list[:, each_feat])
    #     if std != 0:
    #         cmb_list[:, each_feat] = (cmb_list[:, each_feat] - mean) / std
    #     else:
    #         cmb_list[:, each_feat] = np.zeros_like(cmb_list[:, each_feat])
            
    cmb_train = np.array(cmb_list[:len(p_train)])
    print(cmb_train.shape)
    cmb_val = np.array(cmb_list[len(p_train):len(p_train)+len(p_val)])
    cmb_test = np.array(cmb_list[len(p_train)+len(p_val):])
    
    output = Transforms.get_output()
    with output.filesystem().open('cmb_train', 'wb') as f:
        pickle.dump(cmb_train, f)
    with output.filesystem().open('cmb_val', 'wb') as f:
        pickle.dump(cmb_val, f)
    with output.filesystem().open('cmb_test', 'wb') as f:
        pickle.dump(cmb_test, f)
    with output.filesystem().open('M', 'wb') as f:
        pickle.dump([M1, M2, M3, M4, M5, M6, M7], f)
    return output

    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.b8fddf32-e938-4144-8602-8deb3db06d1a"),
    cmb_cond=Input(rid="ri.foundry.main.dataset.a56a7684-c8ae-4c05-bed7-c3bbbcedc309"),
    cmb_drug=Input(rid="ri.foundry.main.dataset.f29206af-f0b2-4f03-9d05-4400888b6ba4"),
    cmb_meas=Input(rid="ri.foundry.main.dataset.d16df23f-fb7c-4084-8281-afdf30b0b994"),
    cmb_proc=Input(rid="ri.foundry.main.dataset.e2e4fbc2-3faf-4e29-b127-d249fa643b4c"),
    feature_process=Input(rid="ri.foundry.main.dataset.55041bc9-f5d0-483a-b4a5-c13b0383c13b"),
    get_full_embd=Input(rid="ri.foundry.main.dataset.0aa50f2e-7a23-4176-ab1f-af00630f6435"),
    person_level_f=Input(rid="ri.foundry.main.dataset.16401b3e-063d-4f8a-8493-36c631fecb27"),
    task_1_goldstandard=Input(rid="ri.foundry.main.dataset.78c4cc3c-d0d9-4997-a851-62e1fb5f3e6b"),
    visit_seq_vec=Input(rid="ri.foundry.main.dataset.ec0df1ca-df03-431d-ac19-3aed08ef0bcc"),
    visit_type_vec=Input(rid="ri.foundry.main.dataset.f2d70200-9d12-4cc6-a1c5-7fd33a4c5f57")
)
def cmb_process_full(person_level_f, task_1_goldstandard, visit_type_vec, visit_seq_vec, cmb_drug, cmb_proc, cmb_meas, cmb_cond, feature_process, get_full_embd):
    with cmb_cond.filesystem().open('person_cond_dict_cmb', 'rb') as f:
        person_cond_dict = pickle.load(f)
    with cmb_meas.filesystem().open('person_meas_dict_cmb', 'rb') as f:
        person_measure_dict = pickle.load(f)
    with cmb_proc.filesystem().open('person_proc_dict_cmb', 'rb') as f:
        person_prod_dict = pickle.load(f)
    with cmb_drug.filesystem().open('person_drug_dict_cmb', 'rb') as f:
        person_drug_dict = pickle.load(f)
    with feature_process.filesystem().open('plist', 'rb') as f:
        plist = pickle.load(f)
    with person_level_f.filesystem().open('person_info_dict', 'rb') as f:
        person_info_dict = pickle.load(f)
    with visit_type_vec.filesystem().open('patient_visit_cnt_vec', 'rb') as f:
        person_visit_cnt_vec = pickle.load(f)
    with visit_seq_vec.filesystem().open('person_visit_seq_vec', 'rb') as f:
        person_visit_seq_vec = pickle.load(f)
    with get_full_embd.filesystem().open('embd_list', 'rb') as f:
        embd_list = pickle.load(f)
    with get_full_embd.filesystem().open('y_list', 'rb') as f:
        y_list = pickle.load(f)

    task2gt = task_1_goldstandard[['person_id','outcome']].toPandas()
    print ('train size:', len(task2gt))

    print ('files loaded')

    # for the training data
    X, y = [], []
    M1 = len(list(person_cond_dict.values())[0][0])
    M2 = len(list(person_measure_dict.values())[0][0])
    M3 = len(list(person_prod_dict.values())[0][0])
    M4 = len(list(person_drug_dict.values())[0][0])
    M5 = len(list(person_info_dict.values())[0])
    M6 = len(list(person_visit_cnt_vec.values())[0])
    M7 = len(list(person_visit_seq_vec.values())[0])

    #_, task2gt = train_test_split(task2gt, test_size=10000, random_state=RANDOM_SEED, stratify=task2gt['outcome'])
    cmb_list = []

    for i in range(len(plist)):
        person = plist[i]
        if person in person_cond_dict:
            f1 = person_cond_dict[person]
        else:
            f1 = [[0 for _ in range(M1)], [0 for _ in range(M1)]]
        if person in person_measure_dict:
            f2 = person_measure_dict[person]
        else:
            f2 = [[0 for _ in range(M2)], [0 for _ in range(M2)]]
        if person in person_prod_dict:
            f3 = person_prod_dict[person]
        else:
            f3 = [[0 for _ in range(M3)], [0 for _ in range(M3)]]
        if person in person_drug_dict:
            f4 = person_drug_dict[person]
        else:
            f4 = [[0 for _ in range(M4)], [0 for _ in range(M4)]]
        if person not in person_info_dict:
            f5 = [0 for _ in range(M5)]
        else:
            f5 = person_info_dict[person]
        if person not in person_visit_cnt_vec:
            f6 = [0 for _ in range(M6)]
        else:
            f6 = person_visit_cnt_vec[person]
        if person not in person_visit_seq_vec:
            f7 = [0 for _ in range(M7)]
        else:
            f7 = person_visit_seq_vec[person]
        tmp = f1[0] + f2[1] + f3[1] + f4[1] + f5 + f6 + f7
        cmb_list.append(tmp)
    
    cmb_list = np.array(cmb_list)
    embd_list = np.array(embd_list)[:, :128]
    print(cmb_list.shape)
    print(embd_list.shape)
    final_embd = np.concatenate((embd_list,cmb_list), axis=-1)
    assert(final_embd.shape[1] == 128+110+90)
    
    
    output = Transforms.get_output()
    with output.filesystem().open('final_embd', 'wb') as f:
        pickle.dump(final_embd, f)
    with output.filesystem().open('y_list', 'wb') as f:
        pickle.dump(y_list, f)
    return output

    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.592b526d-a4bf-41e1-8c0c-67eb8e298041"),
    cond_f_step1=Input(rid="ri.foundry.main.dataset.12f46eea-b019-4abf-80ce-2fd722642f6d")
)
def cond_f_step2(cond_f_step1):
    cond_f_step1 = cond_f_step1
    f1 = cond_f_step1.toPandas()
    
    # collect the dict
    Dict = {}
    for idx, item in enumerate(f1.origin_concept.unique()):
        Dict[item] = idx
    print (len(Dict))
    
    # person_dict
    person_cond_dict = {}
    for idx, (person, tmp) in enumerate(f1.groupby('person_id')):
        binary_vec = [0 for _ in range(len(Dict))]
        count_vec = [0 for _ in range(len(Dict))]
        for _, condition, count in tmp.values:
            if count >= 2:
                binary_vec[Dict[condition]] = 1
                count_vec[Dict[condition]] = count
        person_cond_dict[person] = [binary_vec, count_vec]
    
    # output
    output = Transforms.get_output()
    with output.filesystem().open('person_cond_dict', 'wb') as f:
        pickle.dump(person_cond_dict, f)
    with output.filesystem().open('cond_dict', 'wb') as f:
        pickle.dump(Dict, f)

    print (len(person_cond_dict))

    return output

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ed7e653b-75ff-488e-bd8e-18b79cac26ed"),
    cond_f_step2=Input(rid="ri.foundry.main.dataset.592b526d-a4bf-41e1-8c0c-67eb8e298041")
)
def cond_output(cond_f_step2):
    with cond_f_step2.filesystem().open('cond_dict', 'rb') as f:
        Dict = pickle.load(f)
    print(Dict)
    df = pd.DataFrame(Dict.items(), columns=['feat', 'idx'])
    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.a64152a8-3c4e-48e2-b16f-00230416b495"),
    l_concept=Input(rid="ri.foundry.main.dataset.b5d38a62-a97e-4410-ae71-7013d90712bf"),
    map_dict=Input(rid="ri.foundry.main.dataset.628c932c-c6e4-4e70-825b-e609a7efa9e0")
)
def condition_mapping(l_concept, map_dict):
    l_concept = l_concept
    concept_str = "hypoxemia;respiratory distress;dehydration;secondary infection;acute heart failure;metabolic derangement;diabetes;renal injury;immunodeficiency;obesity;asthma;congenital cardiac anomaly;congenital pulmonary anomaly;high altitude;anxiety;malnutrition;inborn error metabolism;chronic respiratory failure;apnea;chronic kidney disease;malignancy;atrial septal defect;ventricular septal defect;tetralogy of fallot;pulmonary hypertension;azygos lobe lung;congenital cystic adenomatoid malformation;congenital pulmonary airway malformation;panic disorder;post-traumatic stress disorder;failure to thrive;kwashiorkor;weight loss;kwashiorkor;marasmus;free fatty acid defect;gaucher disease;fabre disease;cerebral palsy;on ventilator;prematurity;on ventilator;depression;heart failure;gestation period, 1 week"
    for i in range(2,32):
        concept_str += ';gestation period, %d weeks'%i

    concept_list = concept_str.split(';')
    df_concat = None

    with map_dict.filesystem().open('map_dict', 'rb') as f:
        mdict = pickle.load(f)
    mdict['heart failure'] = mdict['acute heart failure']
    mdict['acute heart failure'] = ['4327205', '4023479', '4235646', '442310', '4215446', '4267800', '40481042','764877', '44782733', '40480603']
    
    for each_concept in concept_list:
        if each_concept in mdict:
            df = l_concept.select('concept_id', 'concept_name').filter(l_concept.concept_id.isin(mdict[each_concept]))
            df = df.withColumn("from_set", lit('1'))
        else:
            if 'gestation period' in each_concept:
                df = l_concept.select('concept_id', 'concept_name').where((l_concept.domain_id=='Condition')).filter(col("concept_name")==each_concept)
            else:
                split_feat = each_concept.split(" ")
                regex_str = ''
                for each_feat in split_feat:
                    cur_regex = '(?=.*' + each_feat + ')'
                    regex_str += cur_regex
                df = l_concept.select('concept_id', 'concept_name').where((l_concept.domain_id=='Condition')).filter(col("concept_name").rlike(regex_str))
            df = df.withColumn("from_set", lit('0'))
        df = df.withColumn("origin_concept", lit(each_concept))
        if df_concat is None:
            df_concat = df
        else:
            df_concat = df_concat.union(df)
    return df_concat

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.44a09da3-0b7a-4010-932e-749badc18506"),
    drug_f_step1=Input(rid="ri.foundry.main.dataset.f88a25cb-9da5-4a5d-9368-d663e830a274")
)
def drug_f_step2(drug_f_step1):
    drug_f_step1 = drug_f_step1
    f1 =  drug_f_step1.toPandas()
    
    # collect the dict
    Dict = {}
    for idx, item in enumerate(f1.origin_concept.unique()):
        Dict[item] = idx
    print (len(Dict))
    
    # person_dict
    person_drug_dict = {}
    for idx, (person, tmp) in enumerate(f1.groupby('person_id')):
        binary_vec = [0 for _ in range(len(Dict))]
        count_vec = [0 for _ in range(len(Dict))]
        for _, drug, cnt in tmp.values:
            if cnt >= 2:
                binary_vec[Dict[drug]] = 1
                count_vec[Dict[drug]] = cnt
        person_drug_dict[person] = [binary_vec, count_vec]
    
    # output
    output = Transforms.get_output()
    with output.filesystem().open('person_drug_dict', 'wb') as f:
        pickle.dump(person_drug_dict, f)
    with output.filesystem().open('drug_dict', 'wb') as f:
        pickle.dump(Dict, f)

    print (len(person_drug_dict))

    return output
    
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c9b47a81-3ff8-4c57-b5ad-b994b365ca16"),
    l_concept=Input(rid="ri.foundry.main.dataset.b5d38a62-a97e-4410-ae71-7013d90712bf"),
    map_dict=Input(rid="ri.foundry.main.dataset.628c932c-c6e4-4e70-825b-e609a7efa9e0")
)
def drug_mapping(l_concept, map_dict):
    l_concept = l_concept
    concept_str = "filgrastim;granulocyte colony stimulating factor;sirolimus;tracrolimus;chemotherapeutic;steroid;inhaled beta agonist;snri;ssri;atypical antipsychotic;nutritional supplement;insulin;filgrastim;granulocyte colony stimulating factor;sirolimus;tacrolimus;vincristine;methotrexate;daunorubicin;bleomycin;dexamethasone;cortisol;albuterol;levocarnitine;clonidine;cyclophosphamide;vinblastine;doxorubicin"

    concept_list = concept_str.split(';')
    df_concat = None

    with map_dict.filesystem().open('map_dict', 'rb') as f:
        mdict = pickle.load(f)
    for each_concept in concept_list:
        if each_concept in mdict:
            df = l_concept.select('concept_id', 'concept_name').filter(l_concept.concept_id.isin(mdict[each_concept]))
            df = df.withColumn("from_set", lit('1'))
        else:
            split_feat = each_concept.split(" ")
            regex_str = ''
            for each_feat in split_feat:
                cur_regex = '(?=.*' + each_feat + ')'
                regex_str += cur_regex
            df = l_concept.select('concept_id', 'concept_name').where((l_concept.domain_id=='Drug')).filter(col("concept_name").rlike(regex_str))
            df = df.withColumn("from_set", lit('0'))
        df = df.withColumn("origin_concept", lit(each_concept))
        if df_concat is None:
            df_concat = df
        else:
            df_concat = df_concat.union(df)
    return df_concat

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.8d0c0780-88a7-4100-8851-e8240db21d3a"),
    drug_f_step2=Input(rid="ri.foundry.main.dataset.44a09da3-0b7a-4010-932e-749badc18506")
)
def drug_output(drug_f_step2):
    with drug_f_step2.filesystem().open('drug_dict', 'rb') as f:
        Dict = pickle.load(f)
    df = pd.DataFrame(Dict.items(), columns=['feat', 'idx'])
    return df

@transform_pandas(
    Output(rid="ri.vector.main.execute.ddc8a0c5-dac8-45de-a762-755fd0b06ef9"),
    DV_split=Input(rid="ri.foundry.main.dataset.2c104028-530c-4b8a-8b27-89b1c7b23877"),
    cmb_process_full=Input(rid="ri.foundry.main.dataset.b8fddf32-e938-4144-8602-8deb3db06d1a")
)
def dv_cnt(DV_split, cmb_process_full):
    with DV_split.filesystem().open('train_idx', 'rb') as f:
        train_idx = pickle.load(f)
    with DV_split.filesystem().open('test_idx', 'rb') as f:
        test_idx = np.array(pickle.load(f))
    with DV_split.filesystem().open('dv', 'rb') as f:
        dv = np.array(pickle.load(f))
    with cmb_process_full.filesystem().open('final_embd', 'rb') as f:
        embd_list = pickle.load(f)
    with cmb_process_full.filesystem().open('y_list', 'rb') as f:
        y_list = np.array(pickle.load(f))
        
    train_cnt = []
    test_cnt = []
    train_ratio = []
    test_ratio = []
    for i in range(len(train_idx)):
        train_cnt.append(len(train_idx[i]))
        test_cnt.append(len(test_idx[i]))

        train_y = np.array([y_list[idx] for idx in train_idx[i]])
        test_y = np.array([y_list[idx] for idx in test_idx[i]])
        train_ratio.append(sum(train_y)/len(train_y))
        test_ratio.append(sum(test_y)/len(test_y))
    print(train_cnt)
    print(test_cnt)
    print(train_ratio)
    print(test_ratio)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.526f07c4-ccc8-4ced-9fdc-368b2a6db685"),
    DV_split=Input(rid="ri.foundry.main.dataset.2c104028-530c-4b8a-8b27-89b1c7b23877"),
    cmb_process_full=Input(rid="ri.foundry.main.dataset.b8fddf32-e938-4144-8602-8deb3db06d1a")
)
def dv_train(DV_split, cmb_process_full):
    with DV_split.filesystem().open('train_idx', 'rb') as f:
        train_idx = pickle.load(f)
    with DV_split.filesystem().open('test_idx', 'rb') as f:
        test_idx = np.array(pickle.load(f))
    with DV_split.filesystem().open('dv', 'rb') as f:
        dv = np.array(pickle.load(f))
    with cmb_process_full.filesystem().open('final_embd', 'rb') as f:
        embd_list = pickle.load(f)
    with cmb_process_full.filesystem().open('y_list', 'rb') as f:
        y_list = np.array(pickle.load(f))
        
    auroc = []
    auprc = []
    minpse = []
    for i in range(len(train_idx)):
        print('Start %d/%d'%(i, len(train_idx)))
        print('Train shape: %d'%len(train_idx[i]))
        print('Test shape: %d'%len(test_idx[i]))
        print(dv[i])
        train_embd = np.array([embd_list[idx, :] for idx in train_idx[i]])
        train_y = np.array([y_list[idx] for idx in train_idx[i]])
        test_embd = np.array([embd_list[idx, :] for idx in test_idx[i]])
        test_y = np.array([y_list[idx] for idx in test_idx[i]])

        xgb = GradientBoostingClassifier(random_state=RANDOM_SEED)

        xgb.fit(train_embd, train_y)
        cl_xgb = CalibratedClassifierCV(xgb, cv=5)
        cl_xgb.fit(train_embd, train_y)
        preds = cl_xgb.predict_proba(test_embd)[:, 1]
        ret = print_metrics_binary(test_y, preds, verbose=1)
        auroc.append(ret['auroc'])
        auprc.append(ret['auprc'])
        minpse.append(ret['minpse'])
    print(auroc)
    print(auprc)
    print(minpse)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.55041bc9-f5d0-483a-b4a5-c13b0383c13b"),
    cond_f_step2=Input(rid="ri.foundry.main.dataset.592b526d-a4bf-41e1-8c0c-67eb8e298041"),
    drug_f_step2=Input(rid="ri.foundry.main.dataset.44a09da3-0b7a-4010-932e-749badc18506"),
    meas_f_step2=Input(rid="ri.foundry.main.dataset.dc23cfc9-dcc2-4717-b303-b4b2da50fb03"),
    person_level_f=Input(rid="ri.foundry.main.dataset.16401b3e-063d-4f8a-8493-36c631fecb27"),
    proc_f_step2=Input(rid="ri.foundry.main.dataset.5e1fa143-945c-44b2-bc56-ce4588baf04f"),
    task_1_gt_ten_thousand=Input(rid="ri.foundry.main.dataset.4e4865dc-429f-4e34-8001-2c221a06a786"),
    visit_seq_vec=Input(rid="ri.foundry.main.dataset.ec0df1ca-df03-431d-ac19-3aed08ef0bcc"),
    visit_type_vec=Input(rid="ri.foundry.main.dataset.f2d70200-9d12-4cc6-a1c5-7fd33a4c5f57")
)
def feature_process(person_level_f, visit_type_vec, visit_seq_vec, drug_f_step2, cond_f_step2, proc_f_step2, meas_f_step2, task_1_gt_ten_thousand):
    condition_f_step2 = cond_f_step2
    measurement_f_step2 = meas_f_step2
    procedure_f_step2 = proc_f_step2 
    visit_seq = visit_seq_vec
    task_1_goldstandard = task_1_gt_ten_thousand
    with condition_f_step2.filesystem().open('person_cond_dict', 'rb') as f:
        person_cond_dict = pickle.load(f)
    with measurement_f_step2.filesystem().open('person_measure_dict', 'rb') as f:
        person_measure_dict = pickle.load(f)
    with procedure_f_step2.filesystem().open('person_prod_dict', 'rb') as f:
        person_prod_dict = pickle.load(f)
    with drug_f_step2.filesystem().open('person_drug_dict', 'rb') as f:
        person_drug_dict = pickle.load(f)
    with person_level_f.filesystem().open('person_info_dict', 'rb') as f:
        person_info_dict = pickle.load(f)
    with visit_type_vec.filesystem().open('patient_visit_cnt_vec', 'rb') as f:
        person_visit_cnt_vec = pickle.load(f)
    with visit_seq_vec.filesystem().open('person_visit_seq_vec', 'rb') as f:
        person_visit_seq_vec = pickle.load(f)

    task2gt = task_1_goldstandard[['person_id','outcome']].toPandas()
    print ('train size:', len(task2gt))

    print ('files loaded')

    # for the training data
    X, y = [], []
    plist = []
    M1 = len(list(person_cond_dict.values())[0][0])
    M2 = len(list(person_measure_dict.values())[0][0])
    M3 = len(list(person_prod_dict.values())[0][0])
    M4 = len(list(person_drug_dict.values())[0][0])
    M5 = len(list(person_info_dict.values())[0])
    M6 = len(list(person_visit_cnt_vec.values())[0])
    M7 = len(list(person_visit_seq_vec.values())[0])

    for idx, (person, outcome) in enumerate(task2gt.values):
        if idx % 5000 == 0:
            print (idx, '/', len(task2gt))

        if person not in person_cond_dict:
            f1 = [[0 for _ in range(M1)], [0 for _ in range(M1)]]; flag1 = 0
        else:
            f1 = person_cond_dict[person]; flag1 = 1
        if person not in person_measure_dict:
            f2 = [[0 for _ in range(M2)], [0 for _ in range(M2)]]; flag2 = 0
        else:
            f2 = person_measure_dict[person]; flag2 = 1
        if person not in person_prod_dict:
            f3 = [[0 for _ in range(M3)], [0 for _ in range(M3)]]; flag3 = 0
        else:
            f3 = person_prod_dict[person]; flag3 = 1
        if person not in person_drug_dict:
            f4 = [[0 for _ in range(M4)], [0 for _ in range(M4)]]; flag4 = 0
        else:
            f4 = person_drug_dict[person]; flag4 = 1
        if person not in person_info_dict:
            f5 = [0 for _ in range(M5)]
        else:
            f5 = person_info_dict[person]
        if person not in person_visit_cnt_vec:
            f6 = [0 for _ in range(M6)]
        else:
            f6 = person_visit_cnt_vec[person]
        if person not in person_visit_seq_vec:
            f7 = [0 for _ in range(M7)]
        else:
            f7 = person_visit_seq_vec[person]

        tmp = f1[0] + f1[1] + f2[0] + f2[1] + f3[0] + f3[1] + f4[0] + f4[1] + f5 + f6 + f7

        if flag1 + flag2 + flag3 + flag4 > 0:
            X.append(tmp); y.append(outcome)
            plist.append(person)
    X = np.array(X); y = np.array(y)

    print ('feature size:', X.shape, 'total person:', len(task2gt))

    # output
    output = Transforms.get_output()
    with output.filesystem().open('X', 'wb') as f:
        pickle.dump(X, f)
    with output.filesystem().open('y', 'wb') as f:
        pickle.dump(y, f)
    with output.filesystem().open('plist', 'wb') as f:
        pickle.dump(plist, f)
    with output.filesystem().open('M', 'wb') as f:
        pickle.dump([M1, M2, M3, M4, M5, M6, M7], f)
    return output

    
    
    
        

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.3ff52f12-63c6-4661-b8fe-f3d3f69862a9"),
    split_dataset=Input(rid="ri.foundry.main.dataset.a6c4b4c5-c9a7-46a0-a28f-13ef917c9262")
)
def find_case(split_dataset):
    with split_dataset.filesystem().open('p_train', 'rb') as f:
        p_train = pickle.load(f)
    with split_dataset.filesystem().open('p_val', 'rb') as f:
        p_val = pickle.load(f)
    with split_dataset.filesystem().open('p_test', 'rb') as f:
        p_test = pickle.load(f)

    idx = '3906519179678687930'
    if idx in p_train:
        print('p_train %d'%p_train.index(idx))
    elif idx in p_val:
        print('p_val %d'%p_val.index(idx))
    elif idx in p_test:
        print('p_test %d'%p_test.index(idx))

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.023ec47b-bc40-40aa-8d0f-2031aa8d2db4"),
    cond_f_step1=Input(rid="ri.foundry.main.dataset.12f46eea-b019-4abf-80ce-2fd722642f6d"),
    drug_f_step1=Input(rid="ri.foundry.main.dataset.f88a25cb-9da5-4a5d-9368-d663e830a274"),
    meas_f_step1=Input(rid="ri.foundry.main.dataset.10cd40f3-70c0-43e2-a92a-561f0d074e18"),
    proc_f_step1=Input(rid="ri.foundry.main.dataset.eb5b5c83-8a32-4a25-be3e-15628097ef47"),
    task_1_goldstandard=Input(rid="ri.foundry.main.dataset.78c4cc3c-d0d9-4997-a851-62e1fb5f3e6b")
)
def find_patient(cond_f_step1, task_1_goldstandard, meas_f_step1, proc_f_step1, drug_f_step1):
    t2 = task_1_goldstandard.filter(task_1_goldstandard.outcome == 1)
    cr = cond_f_step1.join(task_1_goldstandard, cond_f_step1.person_id == task_1_goldstandard.person_id, 'inner').select(cond_f_step1.person_id, cond_f_step1.origin_concept).toPandas()
    mr = meas_f_step1.join(task_1_goldstandard, meas_f_step1.person_id == task_1_goldstandard.person_id, 'inner').select(meas_f_step1.person_id, meas_f_step1.origin_concept).toPandas()
    pr = proc_f_step1.join(task_1_goldstandard, proc_f_step1.person_id == task_1_goldstandard.person_id, 'inner').select(proc_f_step1.person_id, proc_f_step1.origin_concept).toPandas()
    dr = drug_f_step1.join(task_1_goldstandard, drug_f_step1.person_id == task_1_goldstandard.person_id, 'inner').select(drug_f_step1.person_id, drug_f_step1.origin_concept).toPandas()
    df = cr.append(mr, ignore_index=True)
    df = df.append(pr, ignore_index=True)
    df = df.append(dr, ignore_index=True)
    df = df.groupby('person_id').agg({'origin_concept':lambda x: list(x)}).reset_index()
    df = df[df['origin_concept'].map(len) >= 10]
    # pd = {}
    # for idx, row in df.iterrows():
    #     pd[row['person_id']] = row['origin_concept']
    # output = Transforms.get_output()
    # with output.filesystem().open('pd', 'wb') as f:
    #     pickle.dump(pd, f)
    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.b664b449-52e1-483c-ac43-8d1a156ba6c0"),
    build_graph=Input(rid="ri.foundry.main.dataset.c68f658f-c149-499c-9b7e-c40f6c57ddf9"),
    cmb_process_2=Input(rid="ri.foundry.main.dataset.9468f1a2-ed3e-4ae3-b7a3-12ad3d44b6a5"),
    feature_process=Input(rid="ri.foundry.main.dataset.55041bc9-f5d0-483a-b4a5-c13b0383c13b"),
    split_dataset=Input(rid="ri.foundry.main.dataset.a6c4b4c5-c9a7-46a0-a28f-13ef917c9262"),
    train_model=Input(rid="ri.foundry.main.dataset.6c407803-217d-40c2-b0d4-3d113d628301")
)
def gen_cmb_embd(split_dataset, build_graph, train_model, cmb_process_2, feature_process):
    with split_dataset.filesystem().open('x_train', 'rb') as f:
        x_train = pickle.load(f)
    with split_dataset.filesystem().open('x_val', 'rb') as f:
        x_val = pickle.load(f)
    with split_dataset.filesystem().open('x_test', 'rb') as f:
        x_test = pickle.load(f)
    with split_dataset.filesystem().open('y_train', 'rb') as f:
        y_train = pickle.load(f)
    with split_dataset.filesystem().open('y_val', 'rb') as f:
        y_val = pickle.load(f)
    with split_dataset.filesystem().open('y_test', 'rb') as f:
        y_test = pickle.load(f)
    with build_graph.filesystem().open('edge_pair', 'rb') as f:
        edge_pair = pickle.load(f)
    with build_graph.filesystem().open('idx2name', 'rb') as f:
        idx2name = pickle.load(f)
    with feature_process.filesystem().open('M', 'rb') as f:
        M = pickle.load(f)
    with train_model.filesystem().open('model', 'rb') as f:
        saved = pickle.load(f)
    with cmb_process_2.filesystem().open('cmb_train', 'rb') as f:
        cmb_train = pickle.load(f)
    with cmb_process_2.filesystem().open('cmb_test', 'rb') as f:
        cmb_test = pickle.load(f)
    with cmb_process_2.filesystem().open('cmb_val', 'rb') as f:
        cmb_val = pickle.load(f)
        
    M1, M2, M3, M4, M5, M6, M7 = M
    node_dim = 128
    hidden_dim = 128
    mlp_dim = 128
    batch_size = 256
    model = MedML(node_dim, hidden_dim, mlp_dim).to(device) 

    model.load_state_dict(saved)
    print('Load weights')
    model.eval()
    train_embd_list = []
    train_y_list = []
    for i in range(0, len(x_train), batch_size):
        batch_g, batch_adj, batch_mask, batch_flag, batch_info = get_graph_batch(x_train[i:i+batch_size], model.node_embd, idx2name, edge_pair, M)
        batch_y = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32).to(device).unsqueeze(-1)
        pred, g_vec = model(batch_g, batch_adj, batch_mask)
        ehr_vec = torch.zeros(batch_info.size(0), g_vec.size(1)).to(device)
        ehr_vec[batch_flag == 1] = g_vec
        value_vec = cmb_train[i:i+batch_size]
        train_embd_list.append(torch.cat((ehr_vec, batch_info, torch.tensor(value_vec).to(device)), dim=1))
        train_y_list += batch_y.squeeze().cpu().detach().tolist()
    train_embd_list = torch.cat(train_embd_list, dim=0).cpu().detach().numpy()
    print(ehr_vec.shape)
    print(value_vec.shape)
    print(batch_info.shape)
    test_embd_list = []
    test_y_list = []
    for i in range(0, len(x_test), batch_size):
        batch_g, batch_adj, batch_mask, batch_flag, batch_info = get_graph_batch(x_test[i:i+batch_size], model.node_embd, idx2name, edge_pair, M)
        batch_y = torch.tensor(y_test[i:i+batch_size], dtype=torch.float32).to(device).unsqueeze(-1)
        pred, g_vec = model(batch_g, batch_adj, batch_mask)
        ehr_vec = torch.zeros(batch_info.size(0), g_vec.size(1)).to(device)
        ehr_vec[batch_flag == 1] = g_vec
        value_vec = cmb_test[i:i+batch_size]
        test_embd_list.append(torch.cat((ehr_vec, batch_info, torch.tensor(value_vec).to(device)), dim=1))
        test_y_list += batch_y.squeeze().cpu().detach().tolist()
    test_embd_list = torch.cat(test_embd_list, dim=0).cpu().detach().numpy()

    val_embd_list = []
    val_y_list = []
    for i in range(0, len(x_val), batch_size):
        batch_g, batch_adj, batch_mask, batch_flag, batch_info = get_graph_batch(x_val[i:i+batch_size], model.node_embd, idx2name, edge_pair, M)
        batch_y = torch.tensor(y_val[i:i+batch_size], dtype=torch.float32).to(device).unsqueeze(-1)
        pred, g_vec = model(batch_g, batch_adj, batch_mask)
        ehr_vec = torch.zeros(batch_info.size(0), g_vec.size(1)).to(device)
        ehr_vec[batch_flag == 1] = g_vec
        value_vec = cmb_val[i:i+batch_size]
        val_embd_list.append(torch.cat((ehr_vec, batch_info, torch.tensor(value_vec).to(device)), dim=1))
        val_y_list += batch_y.squeeze().cpu().detach().tolist()
    val_embd_list = torch.cat(val_embd_list, dim=0).cpu().detach().numpy()

    output = Transforms.get_output()
    with output.filesystem().open('train_embd_list_cmb', 'wb') as f:
        pickle.dump(train_embd_list, f)
    with output.filesystem().open('train_y_list_cmb', 'wb') as f:
        pickle.dump(train_y_list, f)
    with output.filesystem().open('val_y_list_cmb', 'wb') as f:
        pickle.dump(val_y_list, f)
    with output.filesystem().open('val_embd_list_cmb', 'wb') as f:
        pickle.dump(val_embd_list, f)
    with output.filesystem().open('test_embd_list_cmb', 'wb') as f:
        pickle.dump(test_embd_list, f)
    with output.filesystem().open('test_y_list_cmb', 'wb') as f:
        pickle.dump(test_y_list, f)
    return output

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.6bdab2d9-0632-438c-a342-43851da7ee07"),
    build_graph=Input(rid="ri.foundry.main.dataset.c68f658f-c149-499c-9b7e-c40f6c57ddf9"),
    feature_process=Input(rid="ri.foundry.main.dataset.55041bc9-f5d0-483a-b4a5-c13b0383c13b"),
    split_dataset=Input(rid="ri.foundry.main.dataset.a6c4b4c5-c9a7-46a0-a28f-13ef917c9262"),
    train_model=Input(rid="ri.foundry.main.dataset.6c407803-217d-40c2-b0d4-3d113d628301")
)
def gen_embd(split_dataset, build_graph, feature_process, train_model):
    with split_dataset.filesystem().open('x_train', 'rb') as f:
        x_train = pickle.load(f)
    with split_dataset.filesystem().open('x_test', 'rb') as f:
        x_test = pickle.load(f)
    with split_dataset.filesystem().open('x_norm_train', 'rb') as f:
        x_norm_train = pickle.load(f)
    with split_dataset.filesystem().open('x_norm_test', 'rb') as f:
        x_norm_test = pickle.load(f)
    with split_dataset.filesystem().open('y_train', 'rb') as f:
        y_train = pickle.load(f)
    with split_dataset.filesystem().open('y_test', 'rb') as f:
        y_test = pickle.load(f)
    with build_graph.filesystem().open('edge_pair', 'rb') as f:
        edge_pair = pickle.load(f)
    with build_graph.filesystem().open('idx2name', 'rb') as f:
        idx2name = pickle.load(f)
    with feature_process.filesystem().open('M', 'rb') as f:
        M = pickle.load(f)
    with train_model.filesystem().open('model', 'rb') as f:
        saved = pickle.load(f)
        
    M1, M2, M3, M4, M5, M6, M7 = M
    node_dim = 128
    hidden_dim = 128
    mlp_dim = 128
    batch_size = 256
    model = MedML(node_dim, hidden_dim, mlp_dim).to(device) 

    model.load_state_dict(saved)
    print('Load weights')
    model.eval()
    train_embd_list = []
    train_y_list = []
    for i in range(0, len(x_train), batch_size):
        batch_g, batch_adj, batch_mask, batch_flag, batch_info = get_graph_batch(x_norm_train[i:i+batch_size], model.node_embd, idx2name, edge_pair, M)
        batch_y = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32).to(device).unsqueeze(-1)
        pred, g_vec = model(batch_g, batch_adj, batch_mask)
        ehr_vec = torch.zeros(batch_info.size(0), g_vec.size(1)).to(device)
        ehr_vec[batch_flag == 1] = g_vec
        value_vec = []
        for (each_M, interval_start) in zip([M1,M2,M3,M4], [M1, M1*2+M2,M1*2+M2*2+M3,M1*2+M2*2+M3*2+M4]):
            for each_feat in range(interval_start, interval_start+each_M):
                if each_feat < M1*2:
                    cur_val = np.array(x_train[i:i+batch_size, each_feat])
                    cur_val[cur_val > 0] = 1
                    value_vec.append(cur_val)
                else:
                    value_vec.append(x_train[i:i+batch_size, each_feat])
        value_vec = np.array(value_vec).transpose(1, 0)
        train_embd_list.append(torch.cat((ehr_vec, batch_info, torch.tensor(value_vec).to(device)), dim=1))
        train_y_list += batch_y.squeeze().cpu().detach().tolist()
    train_embd_list = torch.cat(train_embd_list, dim=0).cpu().detach().numpy()

    test_embd_list = []
    test_y_list = []
    for i in range(0, len(x_test), batch_size):
        batch_g, batch_adj, batch_mask, batch_flag, batch_info = get_graph_batch(x_norm_test[i:i+batch_size], model.node_embd, idx2name, edge_pair, M)
        batch_y = torch.tensor(y_test[i:i+batch_size], dtype=torch.float32).to(device).unsqueeze(-1)
        pred, g_vec = model(batch_g, batch_adj, batch_mask)
        ehr_vec = torch.zeros(batch_info.size(0), g_vec.size(1)).to(device)
        ehr_vec[batch_flag == 1] = g_vec
        value_vec = []
        for (each_M, interval_start) in zip([M1,M2,M3,M4], [M1, M1*2+M2,M1*2+M2*2+M3,M1*2+M2*2+M3*2+M4]):
            for each_feat in range(interval_start, interval_start+each_M):
                if each_feat < M1*2:
                    cur_val = np.array(x_test[i:i+batch_size, each_feat])
                    cur_val[cur_val > 0] = 1
                    value_vec.append(cur_val)
                else:
                    value_vec.append(x_test[i:i+batch_size, each_feat])
        value_vec = np.array(value_vec).transpose(1, 0)
        test_embd_list.append(torch.cat((ehr_vec, batch_info, torch.tensor(value_vec).to(device)), dim=1))
        test_y_list += batch_y.squeeze().cpu().detach().tolist()
    test_embd_list = torch.cat(test_embd_list, dim=0).cpu().detach().numpy()

    output = Transforms.get_output()
    with output.filesystem().open('train_embd_list', 'wb') as f:
        pickle.dump(train_embd_list, f)
    with output.filesystem().open('train_y_list', 'wb') as f:
        pickle.dump(train_y_list, f)
    with output.filesystem().open('test_embd_list', 'wb') as f:
        pickle.dump(test_embd_list, f)
    with output.filesystem().open('test_y_list', 'wb') as f:
        pickle.dump(test_y_list, f)
    return output

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.193e22d1-1488-49a2-a2ce-6ab051b15376"),
    build_graph=Input(rid="ri.foundry.main.dataset.c68f658f-c149-499c-9b7e-c40f6c57ddf9"),
    feature_process=Input(rid="ri.foundry.main.dataset.55041bc9-f5d0-483a-b4a5-c13b0383c13b"),
    split_dataset=Input(rid="ri.foundry.main.dataset.a6c4b4c5-c9a7-46a0-a28f-13ef917c9262"),
    train_model=Input(rid="ri.foundry.main.dataset.6c407803-217d-40c2-b0d4-3d113d628301")
)
def gen_embd_ablation(split_dataset, build_graph, feature_process, train_model):
    with split_dataset.filesystem().open('x_train', 'rb') as f:
        x_train = pickle.load(f)
    with split_dataset.filesystem().open('x_test', 'rb') as f:
        x_test = pickle.load(f)
    with split_dataset.filesystem().open('y_train', 'rb') as f:
        y_train = pickle.load(f)
    with split_dataset.filesystem().open('y_test', 'rb') as f:
        y_test = pickle.load(f)
    with build_graph.filesystem().open('edge_pair', 'rb') as f:
        edge_pair = pickle.load(f)
    with build_graph.filesystem().open('idx2name', 'rb') as f:
        idx2name = pickle.load(f)
    with feature_process.filesystem().open('M', 'rb') as f:
        M = pickle.load(f)
    with train_model.filesystem().open('model', 'rb') as f:
        saved = pickle.load(f)
        
    M1, M2, M3, M4, M5, M6, M7 = M
    node_dim = 128
    hidden_dim = 128
    mlp_dim = 128
    batch_size = 256
    model = MedML(node_dim, hidden_dim, mlp_dim).to(device) 

    model.load_state_dict(saved)
    print('Load weights')
    model.eval()
    train_embd_list = []
    train_y_list = []
    for i in range(0, len(x_train), batch_size):
        batch_g, batch_adj, batch_mask, batch_flag, batch_info = get_graph_batch(x_train[i:i+batch_size], model.node_embd, idx2name, edge_pair, M)
        batch_y = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32).to(device).unsqueeze(-1)
        value_vec = []
        for (each_M, interval_start) in zip([M1,M2,M3,M4], [M1, M1*2+M2,M1*2+M2*2+M3,M1*2+M2*2+M3*2+M4]):
            for each_feat in range(interval_start, interval_start+each_M):
                value_vec.append(x_train[i:i+batch_size, each_feat])
        value_vec = np.array(value_vec).transpose(1, 0)
        train_embd_list.append(torch.cat((batch_info, torch.tensor(value_vec).to(device)), dim=1))
        train_y_list += batch_y.squeeze().cpu().detach().tolist()
    train_embd_list = torch.cat(train_embd_list, dim=0).cpu().detach().numpy()

    test_embd_list = []
    test_y_list = []
    for i in range(0, len(x_test), batch_size):
        batch_g, batch_adj, batch_mask, batch_flag, batch_info = get_graph_batch(x_test[i:i+batch_size], model.node_embd, idx2name, edge_pair, M)
        batch_y = torch.tensor(y_test[i:i+batch_size], dtype=torch.float32).to(device).unsqueeze(-1)
        value_vec = []
        for (each_M, interval_start) in zip([M1,M2,M3,M4], [M1, M1*2+M2,M1*2+M2*2+M3,M1*2+M2*2+M3*2+M4]):
            for each_feat in range(interval_start, interval_start+each_M):
                value_vec.append(x_test[i:i+batch_size, each_feat])
        value_vec = np.array(value_vec).transpose(1, 0)
        test_embd_list.append(torch.cat((batch_info, torch.tensor(value_vec).to(device)), dim=1))
        test_y_list += batch_y.squeeze().cpu().detach().tolist()
    test_embd_list = torch.cat(test_embd_list, dim=0).cpu().detach().numpy()

    output = Transforms.get_output()
    with output.filesystem().open('train_embd_list', 'wb') as f:
        pickle.dump(train_embd_list, f)
    with output.filesystem().open('train_y_list', 'wb') as f:
        pickle.dump(train_y_list, f)
    with output.filesystem().open('test_embd_list', 'wb') as f:
        pickle.dump(test_embd_list, f)
    with output.filesystem().open('test_y_list', 'wb') as f:
        pickle.dump(test_y_list, f)
    return output

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.0aa50f2e-7a23-4176-ab1f-af00630f6435"),
    build_graph=Input(rid="ri.foundry.main.dataset.c68f658f-c149-499c-9b7e-c40f6c57ddf9"),
    feature_process=Input(rid="ri.foundry.main.dataset.55041bc9-f5d0-483a-b4a5-c13b0383c13b"),
    train_model=Input(rid="ri.foundry.main.dataset.6c407803-217d-40c2-b0d4-3d113d628301")
)
def get_full_embd(feature_process, build_graph, train_model):
    with feature_process.filesystem().open('X', 'rb') as f:
        X = pickle.load(f)
    with feature_process.filesystem().open('y', 'rb') as f:
        y = pickle.load(f)
    with feature_process.filesystem().open('plist', 'rb') as f:
        plist = pickle.load(f)
    with feature_process.filesystem().open('M', 'rb') as f:
        M1, M2, M3, M4, M5, M6, M7 = pickle.load(f)
    print(M1, M2, M3, M4, M5, M6, M7)
    for (M, interval_start) in zip([M1,M2,M3,M4], [M1, M1*2+M2,M1*2+M2*2+M3,M1*2+M2*2+M3*2+M4]):
        for each_feat in range(interval_start, interval_start+M):
            mean = np.mean(X[:, each_feat])
            std = np.std(X[:, each_feat])
            if std != 0:
                X[:, each_feat] = (X[:, each_feat] - mean) / std
            else:
                X[:, each_feat] = np.zeros_like(X[:, each_feat])
                
    for each_feat in [-1,-2,-3]:
        mean = np.mean(X[:, each_feat])
        std = np.std(X[:, each_feat])
        if std != 0:
            X[:, each_feat] = (X[:, each_feat] - mean) / std
        else:
            X[:, each_feat] = np.zeros_like(X[:, each_feat])
    print ('feature normalzied')
    
    with build_graph.filesystem().open('edge_pair', 'rb') as f:
        edge_pair = pickle.load(f)
    with build_graph.filesystem().open('idx2name', 'rb') as f:
        idx2name = pickle.load(f)
    with feature_process.filesystem().open('M', 'rb') as f:
        M = pickle.load(f)
    with train_model.filesystem().open('model', 'rb') as f:
        saved = pickle.load(f)
        
    M1, M2, M3, M4, M5, M6, M7 = M
    node_dim = 128
    hidden_dim = 128
    mlp_dim = 128
    batch_size = 256
    model = MedML(node_dim, hidden_dim, mlp_dim).to(device) 

    model.load_state_dict(saved)
    print('Load weights')
    model.eval()
    embd_list = []
    y_list = []
    valid_att = []
    valid_embd = []
    valid_y = []
    valid_node = []
    valid_p = []
    for i in range(0, len(X), batch_size):
        batch_g, batch_adj, batch_mask, batch_flag, batch_info, batch_node = get_graph_batch(X[i:i+batch_size], model.node_embd, idx2name, edge_pair, M)
        batch_y = torch.tensor(y[i:i+batch_size], dtype=torch.float32).to(device).unsqueeze(-1)
        pred, g_vec, att = model(batch_g, batch_adj, batch_mask)
        valid_att += att
        valid_node += batch_node
        valid_p += np.array(plist[i:i+batch_size])[batch_flag==1].tolist()
        valid_embd.append(g_vec)
        valid_y.append(batch_y[batch_flag==1])
        ehr_vec = torch.zeros(batch_info.size(0), g_vec.size(1)).to(device)
        ehr_vec[batch_flag == 1] = g_vec
        value_vec = []
        for (each_M, interval_start) in zip([M1,M2,M3,M4], [M1, M1*2+M2,M1*2+M2*2+M3,M1*2+M2*2+M3*2+M4]):
            for each_feat in range(interval_start, interval_start+each_M):
                value_vec.append(X[i:i+batch_size, each_feat])
        value_vec = np.array(value_vec).transpose(1, 0)
        embd_list.append(torch.cat((ehr_vec, batch_info, torch.tensor(value_vec).to(device)), dim=1))
        y_list += batch_y.squeeze().cpu().detach().tolist()
    embd_list = torch.cat(embd_list, dim=0).cpu().detach().numpy()
    valid_embd = torch.cat(valid_embd, dim=0).cpu().detach().numpy()
    valid_y = torch.cat(valid_y, dim=0).cpu().detach().numpy()
    print(valid_embd.shape)
    print(valid_y.shape)
    output = Transforms.get_output()
    with output.filesystem().open('embd_list', 'wb') as f:
        pickle.dump(embd_list, f)
    with output.filesystem().open('y_list', 'wb') as f:
        pickle.dump(y_list, f)
    with output.filesystem().open('valid_att', 'wb') as f:
        pickle.dump(valid_att, f)
    with output.filesystem().open('valid_embd', 'wb') as f:
        pickle.dump(valid_embd, f)
    with output.filesystem().open('valid_y', 'wb') as f:
        pickle.dump(valid_y, f)
    with output.filesystem().open('valid_node', 'wb') as f:
        pickle.dump(valid_node, f)
    with output.filesystem().open('valid_p', 'wb') as f:
        pickle.dump(valid_p, f)
    return output

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.dc23cfc9-dcc2-4717-b303-b4b2da50fb03"),
    meas_f_step1=Input(rid="ri.foundry.main.dataset.10cd40f3-70c0-43e2-a92a-561f0d074e18")
)
def meas_f_step2(meas_f_step1):
    measurement_f_step1 = meas_f_step1
    f1 = measurement_f_step1.toPandas()
    
    # collect the dict
    Dict = {}
    for idx, item in enumerate(f1.origin_concept.unique()):
        Dict[item] = idx
    print (len(Dict))
    
    # person_dict
    person_measure_dict = {}
    for idx, (person, tmp) in enumerate(f1.groupby('person_id')):
        binary_vec = [0 for _ in range(len(Dict))]
        value_vec = [0 for _ in range(len(Dict))]
        for _, measurement, value in tmp.values:
            binary_vec[Dict[measurement]] = 1
            value_vec[Dict[measurement]] = value
        person_measure_dict[person] = [binary_vec, value_vec]
    
    # output
    output = Transforms.get_output()
    with output.filesystem().open('person_measure_dict', 'wb') as f:
        pickle.dump(person_measure_dict, f)
    with output.filesystem().open('meas_dict', 'wb') as f:
        pickle.dump(Dict, f)

    print (len(person_measure_dict))

    return output
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.01377f9c-b83e-4dba-9098-330c87d08a81"),
    l_concept=Input(rid="ri.foundry.main.dataset.b5d38a62-a97e-4410-ae71-7013d90712bf"),
    map_dict=Input(rid="ri.foundry.main.dataset.628c932c-c6e4-4e70-825b-e609a7efa9e0")
)
def meas_mapping(l_concept, map_dict):
    l_concept = l_concept
    concept_str = "spo2;creatinine;pao2;glucose;a1c;gfr;bmi;weight;weight loss;gestational age;myeloblast count"

    concept_list = concept_str.split(';')
    df_concat = None

    with map_dict.filesystem().open('map_dict', 'rb') as f:
        mdict = pickle.load(f)
    for each_concept in concept_list:
        if each_concept in mdict:
            df = l_concept.select('concept_id', 'concept_name').filter(l_concept.concept_id.isin(mdict[each_concept]))
            df = df.withColumn("from_set", lit('1'))
        else:
            split_feat = each_concept.split(" ")
            regex_str = ''
            for each_feat in split_feat:
                cur_regex = '(?=.*' + each_feat + ')'
                regex_str += cur_regex
            df = l_concept.select('concept_id', 'concept_name').where((l_concept.domain_id=='Measurement')).filter(col("concept_name").rlike(regex_str))
            df = df.withColumn("from_set", lit('0'))
        df = df.withColumn("origin_concept", lit(each_concept))
        if df_concat is None:
            df_concat = df
        else:
            df_concat = df_concat.union(df)
    return df_concat

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.b293ed4e-716c-4539-b9ff-addcfc3ccb67"),
    meas_f_step2=Input(rid="ri.foundry.main.dataset.dc23cfc9-dcc2-4717-b303-b4b2da50fb03")
)
def meas_output(meas_f_step2):
    with meas_f_step2.filesystem().open('meas_dict', 'rb') as f:
        Dict = pickle.load(f)
    df = pd.DataFrame(Dict.items(), columns=['feat', 'idx'])
    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.16401b3e-063d-4f8a-8493-36c631fecb27"),
    clean_person=Input(rid="ri.foundry.main.dataset.580dcb5b-4057-4db2-8d23-b29ad09fff7e"),
    location=Input(rid="ri.foundry.main.dataset.efac41e8-cc64-49bf-9007-d7e22a088318"),
    task_1_gt_ten_thousand=Input(rid="ri.foundry.main.dataset.4e4865dc-429f-4e34-8001-2c221a06a786")
)
def person_level_f(location, clean_person, task_1_gt_ten_thousand):
    task_1_goldstandard = task_1_gt_ten_thousand
    clean_person = clean_person
    # merge personT, locationT, task1gt
    # load data
    person_list = task_1_goldstandard[['person_id']]
    task2gt = task_1_goldstandard[['person_id','outcome', 'covid_index', 'outpatient_visit_start_date']].toPandas()
    print ('load task1 data')

    personT = clean_person[['person_id','location_id','year_of_birth','ethnicity_concept_name','gender_concept_name','race_concept_name']]
    personT = personT.join(person_list[['person_id']], on='person_id').toPandas()
    print ('load and clean personT data')

    locationT = location[['location_id', 'state']].toPandas()
    person_level_info = task2gt.merge(personT, on='person_id', how='left').merge(locationT, on='location_id', how='left')

    person_level_info['age'] = (np.floor((pd.to_datetime(person_level_info['outpatient_visit_start_date'], errors = 'coerce') - pd.to_datetime(person_level_info['year_of_birth'], format='%Y', errors = 'coerce')).dt.days / 365.25)).fillna(10).astype(int)
    # null fill
    person_level_info.year_of_birth = person_level_info.year_of_birth.fillna('Nan')
    person_level_info.ethnicity_concept_name = person_level_info.ethnicity_concept_name.fillna('Nan')
    person_level_info.gender_concept_name = person_level_info.gender_concept_name.fillna('Nan')
    person_level_info.race_concept_name = person_level_info.race_concept_name.fillna('Nan')
    person_level_info.state = person_level_info.state.fillna('Nan')

    # process personal info
    ethnicity2idx = {}
    for idx, ethnicity in enumerate(person_level_info.ethnicity_concept_name.unique()):
        ethnicity2idx[ethnicity] = idx
    gender2idx = {}
    for idx, gender in enumerate(person_level_info.gender_concept_name.unique()):
        gender2idx[gender] = idx
    race2idx = {}
    for idx, race in enumerate(person_level_info.race_concept_name.unique()):
        race2idx[race] = idx
    
    # person_leve f
    person_info_dict = {}
    for idx, (p, outcome, covid_index, hospitalization_date, loc, birthyear, ethnicity, gender, race, s, age) in enumerate(person_level_info.values):
        year_vec = [0 for _ in range(19)]
        ethnicity_vec = [0 for _ in range(len(ethnicity2idx))]
        gender_vec = [0 for _ in range(len(gender2idx))]
        race_vec = [0 for _ in range(len(race2idx))]
        cur_age = age if (age < 19 and age>=0) else 10
        year_vec[cur_age] = 1
        ethnicity_vec[ethnicity2idx[ethnicity]] = 1
        gender_vec[gender2idx[gender]] = 1
        race_vec[race2idx[race]] = 1
        # person_info_dict[p] = [year2idx[birthyear], ethnicity2idx[ethnicity], gender2idx[gender], race2idx[race], state2idx[s]]
        person_info_dict[p] = year_vec + ethnicity_vec + gender_vec + race_vec + []

    print(ethnicity2idx)
    print(gender2idx)
    print(race2idx)
    # output
    output = Transforms.get_output()
    with output.filesystem().open('person_info_dict', 'wb') as f:
        pickle.dump(person_info_dict, f)

    print (len(person_info_dict))

    return output
    

    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.5e1fa143-945c-44b2-bc56-ce4588baf04f"),
    proc_f_step1=Input(rid="ri.foundry.main.dataset.eb5b5c83-8a32-4a25-be3e-15628097ef47")
)
def proc_f_step2(proc_f_step1):
    procedure_f_step1 = proc_f_step1
    f1 =  procedure_f_step1.toPandas()
    
    # collect the dict
    Dict = {}
    for idx, item in enumerate(f1.origin_concept.unique()):
        Dict[item] = idx
    print (len(Dict))
    
    # person_dict
    person_prod_dict = {}
    for idx, (person, tmp) in enumerate(f1.groupby('person_id')):
        binary_vec = [0 for _ in range(len(Dict))]
        count_vec = [0 for _ in range(len(Dict))]
        for _, procedure, value in tmp.values:
            if value >= 2:
                binary_vec[Dict[procedure]] = 1
                count_vec[Dict[procedure]] = value
        person_prod_dict[person] = [binary_vec, count_vec]
    
    # output
    output = Transforms.get_output()
    with output.filesystem().open('person_prod_dict', 'wb') as f:
        pickle.dump(person_prod_dict, f)
    with output.filesystem().open('proc_dict', 'wb') as f:
        pickle.dump(Dict, f)
    
    print (len(person_prod_dict))

    return output
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.8742f851-61f0-4ab4-84e7-22afdf6dfae0"),
    l_concept=Input(rid="ri.foundry.main.dataset.b5d38a62-a97e-4410-ae71-7013d90712bf"),
    map_dict=Input(rid="ri.foundry.main.dataset.628c932c-c6e4-4e70-825b-e609a7efa9e0")
)
def proc_mapping(l_concept, map_dict):
    l_concept = l_concept
    concept_str = "solid organ transplant;gastrostomy tube placement;bone marrow biopsy;tumor resection;cardiac catheterization;central venous catheter;on ventilator"
    
    concept_list = concept_str.split(';')
    df_concat = None

    with map_dict.filesystem().open('map_dict', 'rb') as f:
        mdict = pickle.load(f)
    for each_concept in concept_list:
        if each_concept in mdict:
            df = l_concept.select('concept_id', 'concept_name').filter(l_concept.concept_id.isin(mdict[each_concept]))
            df = df.withColumn("from_set", lit('1'))
        else:
            split_feat = each_concept.split(" ")
            regex_str = ''
            for each_feat in split_feat:
                cur_regex = '(?=.*' + each_feat + ')'
                regex_str += cur_regex
            df = l_concept.select('concept_id', 'concept_name').where((l_concept.domain_id=='Procedure')).filter(col("concept_name").rlike(regex_str))
            df = df.withColumn("from_set", lit('0'))
        df = df.withColumn("origin_concept", lit(each_concept))
        if df_concat is None:
            df_concat = df
        else:
            df_concat = df_concat.union(df)
    return df_concat

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.aa5b29a2-62d8-4d47-a0ba-a5b1cac111ce"),
    proc_f_step2=Input(rid="ri.foundry.main.dataset.5e1fa143-945c-44b2-bc56-ce4588baf04f")
)
def proc_output(proc_f_step2):
    with proc_f_step2.filesystem().open('proc_dict', 'rb') as f:
        Dict = pickle.load(f)
    df = pd.DataFrame(Dict.items(), columns=['feat', 'idx'])
    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.a6c4b4c5-c9a7-46a0-a28f-13ef917c9262"),
    feature_process=Input(rid="ri.foundry.main.dataset.55041bc9-f5d0-483a-b4a5-c13b0383c13b")
)
def split_dataset(feature_process):
    with feature_process.filesystem().open('X', 'rb') as f:
        X = pickle.load(f)
    with feature_process.filesystem().open('y', 'rb') as f:
        y = pickle.load(f)
    with feature_process.filesystem().open('plist', 'rb') as f:
        plist = pickle.load(f)
    with feature_process.filesystem().open('M', 'rb') as f:
        M1, M2, M3, M4, M5, M6, M7 = pickle.load(f)
    print(M1, M2, M3, M4, M5, M6, M7)
    x_norm = np.array(X)

    norm_dict = {}
    for (M, interval_start) in zip([M1,M2,M3,M4], [M1, M1*2+M2,M1*2+M2*2+M3,M1*2+M2*2+M3*2+M4]):
        for each_feat in range(interval_start, interval_start+M):
            mean = np.mean(X[:, each_feat])
            std = np.std(X[:, each_feat])
            norm_dict[each_feat] = (mean, std)
            if std != 0:
                x_norm[:, each_feat] = (X[:, each_feat] - mean) / std
            else:
                x_norm[:, each_feat] = np.zeros_like(X[:, each_feat])
                
    for each_feat in [-1,-2,-3]:
        mean = np.mean(X[:, each_feat])
        std = np.std(X[:, each_feat])
        norm_dict[each_feat] = (mean, std)
        if std != 0:
            x_norm[:, each_feat] = (X[:, each_feat] - mean) / std
        else:
            x_norm[:, each_feat] = np.zeros_like(X[:, each_feat])
    print ('feature normalzied')
    print(norm_dict)
    # final output
    x_train, x_test, x_norm_train, x_norm_test, y_train, y_test, p_train, p_test = train_test_split(X, x_norm, y, plist, test_size=0.4, stratify=y, random_state=RANDOM_SEED)
    x_val, x_test, x_norm_val, x_norm_test, y_val, y_test, p_val, p_test = train_test_split(x_test, x_norm_test, y_test, p_test, test_size=0.75, stratify=y_test, random_state=RANDOM_SEED)
    sampled_x, sampled_y = x_train, y_train
    output = Transforms.get_output()
    print(x_train.shape)
    print(x_val.shape)
    print(x_test.shape)
    with output.filesystem().open('x_train', 'wb') as f:
        pickle.dump(x_train, f)
    with output.filesystem().open('x_val', 'wb') as f:
        pickle.dump(x_val, f)
    with output.filesystem().open('x_test', 'wb') as f:
        pickle.dump(x_test, f)
    with output.filesystem().open('x_norm_train', 'wb') as f:
        pickle.dump(x_train, f)
    with output.filesystem().open('x_norm_val', 'wb') as f:
        pickle.dump(x_val, f)
    with output.filesystem().open('x_norm_test', 'wb') as f:
        pickle.dump(x_test, f)
    with output.filesystem().open('y_train', 'wb') as f:
        pickle.dump(y_train, f)
    with output.filesystem().open('y_val', 'wb') as f:
        pickle.dump(y_val, f)
    with output.filesystem().open('y_test', 'wb') as f:
        pickle.dump(y_test, f)
    with output.filesystem().open('p_train', 'wb') as f:
        pickle.dump(p_train, f)
    with output.filesystem().open('p_val', 'wb') as f:
        pickle.dump(p_val, f)
    with output.filesystem().open('p_test', 'wb') as f:
        pickle.dump(p_test, f)
    with output.filesystem().open('norm_dict', 'wb') as f:
        pickle.dump(norm_dict, f)
    return output

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.7605bc24-43b1-4171-b6d8-93f9675b37fc"),
    feature_process=Input(rid="ri.foundry.main.dataset.55041bc9-f5d0-483a-b4a5-c13b0383c13b"),
    location=Input(rid="ri.foundry.main.dataset.efac41e8-cc64-49bf-9007-d7e22a088318"),
    person=Input(rid="ri.foundry.main.dataset.50cae11a-4afb-457d-99d4-55b4bc2cbe66")
)
def state_split(feature_process, location, person):
    with feature_process.filesystem().open('plist', 'rb') as f:
        plist = pickle.load(f)
    with feature_process.filesystem().open('M', 'rb') as f:
        M1, M2, M3, M4, M5, M6, M7 = pickle.load(f)

    record = person.filter(person.person_id.isin(plist)).filter(~person.location_id.isNull())
    record = record.join(location, record.location_id == location.location_id, 'inner').select(record['*'], location['state'])
    #record = record.withColumn("zipfix", record['zip'].substr(0,5))
    record = record.toPandas()
    
    states = [ 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
           'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
           'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
           'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
           'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

    abbr2name = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
    }
    name2abbr = {abbr2name[k].upper():k for k in abbr2name}
    
    state_cnt = {}
    state2pid = {}
    for idx, row in record.iterrows():
        if row['state'] in states:
            if row['state'] in state2pid:
                state2pid[row['state']].append(row['person_id'])
                state_cnt[row['state']] += 1
            else:
                state2pid[row['state']] = [row['person_id']]
                state_cnt[row['state']] = 1
        elif row['state'] in name2abbr:
            if name2abbr[row['state']] in state2pid:
                state2pid[name2abbr[row['state']]].append(row['person_id'])
                state_cnt[name2abbr[row['state']]] += 1
            else:
                state2pid[name2abbr[row['state']]] = [row['person_id']]
                state_cnt[name2abbr[row['state']]] = 1

    output = Transforms.get_output()
    with output.filesystem().open('state2pid', 'wb') as f:
        pickle.dump(state2pid, f)
    with output.filesystem().open('state_cnt', 'wb') as f:
        pickle.dump(state_cnt, f)
    return output
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.74f6d94b-92d5-43f5-a9cb-0f2cca3f12d1"),
    feature_process=Input(rid="ri.foundry.main.dataset.55041bc9-f5d0-483a-b4a5-c13b0383c13b"),
    task_1_goldstandard=Input(rid="ri.foundry.main.dataset.78c4cc3c-d0d9-4997-a851-62e1fb5f3e6b")
)
def temporal_cnt(feature_process, task_1_goldstandard):
    with feature_process.filesystem().open('plist', 'rb') as f:
        plist = pickle.load(f)
    with feature_process.filesystem().open('M', 'rb') as f:
        M1, M2, M3, M4, M5, M6, M7 = pickle.load(f)

    ratio = []
    cnt = []
    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2020-01-01'),pd.to_datetime('2020-04-01'))).toPandas()
    ratio.append(phase['outcome'].sum()/len(phase))
    cnt.append(len(phase))
    p1_list = np.array(list(set(phase['person_id'].tolist()).intersection(set(plist))))

    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2020-04-01'),pd.to_datetime('2020-07-01'))).toPandas()
    ratio.append(phase['outcome'].sum()/len(phase))
    cnt.append(len(phase))
    p2_list = np.array(list(set(phase['person_id'].tolist()).intersection(set(plist))))

    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2020-07-01'),pd.to_datetime('2020-10-01'))).toPandas()
    ratio.append(phase['outcome'].sum()/len(phase))
    cnt.append(len(phase))
    p3_list = np.array(list(set(phase['person_id'].tolist()).intersection(set(plist))))

    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2020-10-01'),pd.to_datetime('2021-01-01'))).toPandas()
    ratio.append(phase['outcome'].sum()/len(phase))
    cnt.append(len(phase))
    p4_list = np.array(list(set(phase['person_id'].tolist()).intersection(set(plist))))

    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2021-01-01'),pd.to_datetime('2021-04-01'))).toPandas()
    ratio.append(phase['outcome'].sum()/len(phase))
    cnt.append(len(phase))
    p5_list = np.array(list(set(phase['person_id'].tolist()).intersection(set(plist))))

    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2021-04-01'),pd.to_datetime('2021-07-01'))).toPandas()
    ratio.append(phase['outcome'].sum()/len(phase))
    cnt.append(len(phase))
    p6_list = np.array(list(set(phase['person_id'].tolist()).intersection(set(plist))))

    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2021-07-01'),pd.to_datetime('2021-10-01'))).toPandas()
    ratio.append(phase['outcome'].sum()/len(phase))
    cnt.append(len(phase))
    p7_list = np.array(list(set(phase['person_id'].tolist()).intersection(set(plist))))

    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2021-10-01'),pd.to_datetime('2022-01-01'))).toPandas()
    ratio.append(phase['outcome'].sum()/len(phase))
    cnt.append(len(phase))
    p8_list = np.array(list(set(phase['person_id'].tolist()).intersection(set(plist))))

    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2022-01-01'),pd.to_datetime('2022-04-01'))).toPandas()
    ratio.append(phase['outcome'].sum()/len(phase))
    cnt.append(len(phase))
    p9_list = np.array(list(set(phase['person_id'].tolist()).intersection(set(plist))))

    print(ratio)
    print(cnt)

    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.f2f90562-6291-4878-9da8-8dd9f5565bbf"),
    feature_process=Input(rid="ri.foundry.main.dataset.55041bc9-f5d0-483a-b4a5-c13b0383c13b"),
    task_1_goldstandard=Input(rid="ri.foundry.main.dataset.78c4cc3c-d0d9-4997-a851-62e1fb5f3e6b")
)
def temporal_split(feature_process, task_1_goldstandard):
    with feature_process.filesystem().open('plist', 'rb') as f:
        plist = pickle.load(f)
    with feature_process.filesystem().open('M', 'rb') as f:
        M1, M2, M3, M4, M5, M6, M7 = pickle.load(f)

    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2020-01-01'),pd.to_datetime('2020-04-01')))
    p1_list = np.array(list(set(phase.toPandas()['person_id'].tolist()).intersection(set(plist))))
    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2020-04-01'),pd.to_datetime('2020-07-01')))
    p2_list = np.array(list(set(phase.toPandas()['person_id'].tolist()).intersection(set(plist))))
    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2020-07-01'),pd.to_datetime('2020-10-01')))
    p3_list = np.array(list(set(phase.toPandas()['person_id'].tolist()).intersection(set(plist))))
    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2020-10-01'),pd.to_datetime('2021-01-01')))
    p4_list = np.array(list(set(phase.toPandas()['person_id'].tolist()).intersection(set(plist))))
    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2021-01-01'),pd.to_datetime('2021-04-01')))
    p5_list = np.array(list(set(phase.toPandas()['person_id'].tolist()).intersection(set(plist))))
    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2021-04-01'),pd.to_datetime('2021-07-01')))
    p6_list = np.array(list(set(phase.toPandas()['person_id'].tolist()).intersection(set(plist))))
    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2021-07-01'),pd.to_datetime('2021-10-01')))
    p7_list = np.array(list(set(phase.toPandas()['person_id'].tolist()).intersection(set(plist))))
    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2021-10-01'),pd.to_datetime('2022-01-01')))
    p8_list = np.array(list(set(phase.toPandas()['person_id'].tolist()).intersection(set(plist))))
    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2022-01-01'),pd.to_datetime('2022-04-01')))
    p9_list = np.array(list(set(phase.toPandas()['person_id'].tolist()).intersection(set(plist))))

    phase_list = [p1_list,p2_list,p3_list,p4_list,p5_list,p6_list,p7_list,p8_list]
    idx_list = []
    for each_l in phase_list:
        cur_idx = []
        for each_p in each_l:
            cur_idx.append(plist.index(each_p))
        idx_list.append(cur_idx)
        print(len(each_l))
    
    output = Transforms.get_output()
    with output.filesystem().open('idx_list', 'wb') as f:
        pickle.dump(idx_list, f)
    return output
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.45ac07ba-9fe0-4a38-8a52-df81eef5a656"),
    feature_process=Input(rid="ri.foundry.main.dataset.55041bc9-f5d0-483a-b4a5-c13b0383c13b"),
    task_1_goldstandard=Input(rid="ri.foundry.main.dataset.78c4cc3c-d0d9-4997-a851-62e1fb5f3e6b")
)
def temporal_split_plist(feature_process, task_1_goldstandard):
    with feature_process.filesystem().open('plist', 'rb') as f:
        plist = pickle.load(f)
    with feature_process.filesystem().open('M', 'rb') as f:
        M1, M2, M3, M4, M5, M6, M7 = pickle.load(f)

    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2020-01-01'),pd.to_datetime('2020-04-01')))
    p1_list = np.array(list(set(phase.toPandas()['person_id'].tolist()).intersection(set(plist))))
    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2020-04-01'),pd.to_datetime('2020-07-01')))
    p2_list = np.array(list(set(phase.toPandas()['person_id'].tolist()).intersection(set(plist))))
    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2020-07-01'),pd.to_datetime('2020-10-01')))
    p3_list = np.array(list(set(phase.toPandas()['person_id'].tolist()).intersection(set(plist))))
    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2020-10-01'),pd.to_datetime('2021-01-01')))
    p4_list = np.array(list(set(phase.toPandas()['person_id'].tolist()).intersection(set(plist))))
    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2021-01-01'),pd.to_datetime('2021-04-01')))
    p5_list = np.array(list(set(phase.toPandas()['person_id'].tolist()).intersection(set(plist))))
    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2021-04-01'),pd.to_datetime('2021-07-01')))
    p6_list = np.array(list(set(phase.toPandas()['person_id'].tolist()).intersection(set(plist))))
    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2021-07-01'),pd.to_datetime('2021-10-01')))
    p7_list = np.array(list(set(phase.toPandas()['person_id'].tolist()).intersection(set(plist))))
    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2021-10-01'),pd.to_datetime('2022-01-01')))
    p8_list = np.array(list(set(phase.toPandas()['person_id'].tolist()).intersection(set(plist))))
    phase = task_1_goldstandard.filter(~task_1_goldstandard.covid_index.isNull()).filter(task_1_goldstandard.covid_index.between(pd.to_datetime('2022-01-01'),pd.to_datetime('2022-04-01')))
    p9_list = np.array(list(set(phase.toPandas()['person_id'].tolist()).intersection(set(plist))))

    phase_list = [p1_list,p2_list,p3_list,p4_list,p5_list,p6_list,p7_list,p8_list]
    
    output = Transforms.get_output()
    with output.filesystem().open('phase_list', 'wb') as f:
        pickle.dump(phase_list, f)
    return output
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.6298dc25-b85c-462a-9f05-24aa7aec9b88"),
    cmb_process_full=Input(rid="ri.foundry.main.dataset.b8fddf32-e938-4144-8602-8deb3db06d1a"),
    feature_process=Input(rid="ri.foundry.main.dataset.55041bc9-f5d0-483a-b4a5-c13b0383c13b"),
    temporal_split=Input(rid="ri.foundry.main.dataset.f2f90562-6291-4878-9da8-8dd9f5565bbf")
)
def temporal_train(temporal_split, feature_process, cmb_process_full):
    with temporal_split.filesystem().open('idx_list', 'rb') as f:
        idx_list = pickle.load(f)
    with cmb_process_full.filesystem().open('final_embd', 'rb') as f:
        embd_list = pickle.load(f)
    with cmb_process_full.filesystem().open('y_list', 'rb') as f:
        y_list = np.array(pickle.load(f))

    auroc = []
    auprc = []
    minpse = []
    for i in range(2,8):
        print('Start %d/7'%i)
        train_embd = np.array([embd_list[idx, :] for idx in list(idx_list[i-1])+list(idx_list[i-2])])
        train_y = np.array([y_list[idx] for idx in list(idx_list[i-1])+list(idx_list[i-2])])
        test_embd = np.array([embd_list[idx, :] for idx in list(idx_list[i])])
        test_y = np.array([y_list[idx] for idx in list(idx_list[i])])

        xgb = GradientBoostingClassifier(random_state=RANDOM_SEED)

        xgb.fit(train_embd, train_y)
        cl_xgb = CalibratedClassifierCV(xgb, cv=5)
        cl_xgb.fit(train_embd, train_y)
        preds = cl_xgb.predict_proba(test_embd)[:, 1]
        ret = print_metrics_binary(test_y, preds, verbose=1)
        auroc.append(ret['auroc'])
        auprc.append(ret['auprc'])
        minpse.append(ret['minpse'])
    print(auroc)
    print(auprc)
    print(minpse)

        

@transform_pandas(
    Output(rid="ri.vector.main.execute.f7b89683-477c-49f3-9f6c-5757047561bb"),
    cmb_process_full=Input(rid="ri.foundry.main.dataset.b8fddf32-e938-4144-8602-8deb3db06d1a"),
    feature_process=Input(rid="ri.foundry.main.dataset.55041bc9-f5d0-483a-b4a5-c13b0383c13b"),
    get_full_embd=Input(rid="ri.foundry.main.dataset.0aa50f2e-7a23-4176-ab1f-af00630f6435"),
    temporal_split=Input(rid="ri.foundry.main.dataset.f2f90562-6291-4878-9da8-8dd9f5565bbf")
)
def temporal_train_abl(temporal_split, feature_process, get_full_embd, cmb_process_full):
    with temporal_split.filesystem().open('idx_list', 'rb') as f:
        idx_list = pickle.load(f)
    with get_full_embd.filesystem().open('embd_list', 'rb') as f:
        embd_list = pickle.load(f)
    with get_full_embd.filesystem().open('y_list', 'rb') as f:
        y_list = np.array(pickle.load(f))

    auroc = []
    auprc = []
    minpse = []
    for i in range(2,8):
        print('Start %d/7'%i)
        train_embd = np.array([embd_list[idx, :] for idx in list(idx_list[i-1])+list(idx_list[i-2])])
        train_y = np.array([y_list[idx] for idx in list(idx_list[i-1])+list(idx_list[i-2])])
        test_embd = np.array([embd_list[idx, :] for idx in list(idx_list[i])])
        test_y = np.array([y_list[idx] for idx in list(idx_list[i])])

        xgb = XGBClassifier(nthread=4,
        n_jobs=-1,
        seed=RANDOM_SEED)

        xgb.fit(train_embd, train_y)
        cl_xgb = CalibratedClassifierCV(xgb, cv=5)
        cl_xgb.fit(train_embd, train_y)
        preds = cl_xgb.predict_proba(test_embd)[:, 1]
        ret = print_metrics_binary(test_y, preds, verbose=1)
        auroc.append(ret['auroc'])
        auprc.append(ret['auprc'])
        minpse.append(ret['minpse'])
    print(auroc)
    print(auprc)
    print(minpse)

        

@transform_pandas(
    Output(rid="ri.vector.main.execute.465a1958-08a1-404f-b5ad-69da36669ab8"),
    gen_embd_ablation=Input(rid="ri.foundry.main.dataset.193e22d1-1488-49a2-a2ce-6ab051b15376")
)
def train_abl(gen_embd_ablation):
    with gen_embd_ablation.filesystem().open('train_embd_list', 'rb') as f:
        train_embd_list = pickle.load(f)
    with gen_embd_ablation.filesystem().open('test_embd_list', 'rb') as f:
        test_embd_list = pickle.load(f)
    with gen_embd_ablation.filesystem().open('train_y_list', 'rb') as f:
        train_y_list = pickle.load(f)
    with gen_embd_ablation.filesystem().open('test_y_list', 'rb') as f:
        test_y_list = pickle.load(f)
    
    # print('\n----XGB----')
    # xgb = XGBClassifier(nthread=4,
    #     n_jobs=-1,
    #     seed=RANDOM_SEED)

    # xgb.fit(train_embd_list, train_y_list)
    # cl_xgb = CalibratedClassifierCV(xgb, cv=5)
    # cl_xgb.fit(train_embd_list, train_y_list)
    # preds = cl_xgb.predict_proba(test_embd_list)[:, 1]
    # ret = print_metrics_binary(test_y_list, preds)
    # print(ret)

    print('\n----GBDT----')
    clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
    clf.fit(train_embd_list, train_y_list)
    # cl_clf = CalibratedClassifierCV(clf, cv=5)
    # cl_clf.fit(train_embd_list, train_y_list)
    preds = clf.predict_proba(test_embd_list)[:, 1]
    ret = print_metrics_binary(test_y_list, preds)
    fpr, tpr, _ = metrics.roc_curve(test_y_list, preds)
    prc, rec, _ = precision_recall_curve(test_y_list, preds)
    fpr = [fpr[i] for i in range(0, len(fpr), 2)]
    tpr = [tpr[i] for i in range(0, len(tpr), 2)]
    prc = [prc[i] for i in range(0, len(prc), 2)]
    rec = [rec[i] for i in range(0, len(rec), 2)]
    print(list(fpr))
    print(list(tpr))
    print(list(prc))
    print(list(rec))
    print (ret)
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.9dff932c-af86-4d24-bbae-f19ac4dc8d61"),
    feature_process=Input(rid="ri.foundry.main.dataset.55041bc9-f5d0-483a-b4a5-c13b0383c13b")
)
def train_and_prediction(feature_process):
    feature_process = feature_process
    with feature_process.filesystem().open('X', 'rb') as f:
        X = pickle.load(f)
    with feature_process.filesystem().open('y', 'rb') as f:
        y = pickle.load(f)
    with feature_process.filesystem().open('M', 'rb') as f:
        M1, M2, M3, M4, M5, M6, M7 = pickle.load(f)

    print ('X y loaded')
    
    # feature normalization
    # for interval_start in [M, 3*M, 5*M]:
    # for (M, interval_start) in zip([min(M1,setK1),min(M2,setK2),min(M3,setK3),min(M4,setK4)], \
    #             [min(M1,setK1), min(M1,setK1)*2+min(M2,setK2), min(M1,setK1)*2+min(M2,setK2)*2+min(M3,setK3), \
    #             min(M1,setK1)*2+min(M2,setK2)*2+min(M3,setK3)*2+min(M4,setK4)]):
    for (M, interval_start) in zip([M1,M2,M3,M4], [M1, M1*2+M2,M1*2+M2*2+M3,M1*2+M2*2+M3*2+M4]):
        tmp = X[:, interval_start:interval_start+M]
        MIN = np.percentile(tmp, 5, axis=0)
        MAX = np.percentile(tmp, 95, axis=0)
        tmp = (tmp - MIN) / (MAX - MIN + 1e-9)
        X[:, interval_start:interval_start+M] = tmp

    for idx in [-1, -2, -3]:
        tmp = X[:, idx]
        MIN = np.percentile(tmp, 5)
        MAX = np.percentile(tmp, 95)
        tmp = (tmp - MIN) / (MAX - MIN + 1e-9)
        X[:, idx] = tmp
    print ('feature normalzied')

    # final output
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    xgb = XGBClassifier(
            max_depth=4,#1
            n_estimators=170, #1
            gamma=0,#1
            subsample=0.8,#1
            scale_pos_weight=1,
            colsample_bytree=0.8,#1
            min_child_weight=3,#1
            reg_lambda=45.555,#1
            learning_rate =0.1,
            objective='binary:logistic',
            nthread=4,
            n_jobs=-1,
            seed=RANDOM_SEED)
    xgb_calibrated = CalibratedClassifierCV(xgb, cv=5, method='isotonic')
    rf = RandomForestClassifier(n_estimators=100, max_depth=13, min_samples_split=120, min_samples_leaf=10, max_features=55, n_jobs=-1, random_state=RANDOM_SEED)
    rf_calibrated = CalibratedClassifierCV(rf, cv=5, method='isotonic')
    estimators = [('xgb', xgb_calibrated), ('rf', rf_calibrated)]

    
    for ratio in np.linspace(0.1, 1.2, 20):
        print ('================== result for ratio: {} ============='.format(ratio))
        clf = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        sampled_x, sampled_y = balanced_sample_maker(x_train, y_train, ratio=ratio)
        clf.fit(sampled_x, sampled_y)
        preds = clf.predict_proba(x_test)[:, 1]
        print_metrics_binary(y_test, preds)
        print ()

    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.6c407803-217d-40c2-b0d4-3d113d628301"),
    build_graph=Input(rid="ri.foundry.main.dataset.c68f658f-c149-499c-9b7e-c40f6c57ddf9"),
    feature_process=Input(rid="ri.foundry.main.dataset.55041bc9-f5d0-483a-b4a5-c13b0383c13b"),
    split_dataset=Input(rid="ri.foundry.main.dataset.a6c4b4c5-c9a7-46a0-a28f-13ef917c9262")
)
def train_model(split_dataset, build_graph, feature_process):
    with split_dataset.filesystem().open('x_norm_train', 'rb') as f:
        x_train = pickle.load(f)
    with split_dataset.filesystem().open('x_norm_val', 'rb') as f:
        x_val = pickle.load(f)
    with split_dataset.filesystem().open('y_train', 'rb') as f:
        y_train = pickle.load(f)
    with split_dataset.filesystem().open('y_val', 'rb') as f:
        y_val = pickle.load(f)
    with build_graph.filesystem().open('edge_pair', 'rb') as f:
        edge_pair = pickle.load(f)
    with build_graph.filesystem().open('idx2name', 'rb') as f:
        idx2name = pickle.load(f)
    with feature_process.filesystem().open('M', 'rb') as f:
        M = pickle.load(f)
    node_dim = 128
    hidden_dim = 128
    mlp_dim = 128

    model = MedML(node_dim, hidden_dim, mlp_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss(reduction='sum')
    print(model)

    epoch = 40
    batch_size = 256
    best_loss = 0
    output = Transforms.get_output()
    for each_epoch in range(epoch):
        train_loss = 0
        model.train()
        for i in range(0, len(x_train), batch_size):
            batch_g, batch_adj, batch_mask, batch_flag, batch_info = get_graph_batch(x_train[i:i+batch_size], model.node_embd, idx2name, edge_pair, M)
            batch_y = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32).to(device).unsqueeze(-1)[np.array(batch_flag)==1]
            pred, _ = model(batch_g, batch_adj, batch_mask)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        val_loss = 0
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for i in range(0, len(x_val), batch_size):
                batch_g, batch_adj, batch_mask, batch_flag, batch_info = get_graph_batch(x_val[i:i+batch_size], model.node_embd, idx2name, edge_pair, M)
                batch_y = torch.tensor(y_val[i:i+batch_size], dtype=torch.float32).to(device).unsqueeze(-1)[np.array(batch_flag)==1]
                pred, _ = model(batch_g, batch_adj, batch_mask)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
                y_true += list(batch_y.cpu().detach().numpy().reshape(-1))
                y_pred += list(pred.cpu().detach().numpy().reshape(-1))
    
        ret = print_metrics_binary(y_true, y_pred, verbose=0)
        if ret['auprc'] > best_loss:
            best_loss = ret['auprc']
            with output.filesystem().open('model', 'wb') as f:
                pickle.dump(model.state_dict(), f)
        print('Epoch: {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f} Val ROC: {:.4f} Val PRC: {:.4f}'.format(each_epoch, train_loss/len(x_train), val_loss/len(x_val), ret['auroc'], ret['auprc']))
    return output

@transform_pandas(
    Output(rid="ri.vector.main.execute.8f07ca82-8637-4149-bc0b-53d3275c2e20"),
    gen_embd=Input(rid="ri.foundry.main.dataset.6bdab2d9-0632-438c-a342-43851da7ee07")
)
def train_xgb_0(gen_embd):
    with gen_embd.filesystem().open('train_embd_list', 'rb') as f:
        train_embd_list = pickle.load(f)
    with gen_embd.filesystem().open('test_embd_list', 'rb') as f:
        test_embd_list = pickle.load(f)
    with gen_embd.filesystem().open('train_y_list', 'rb') as f:
        train_y_list = pickle.load(f)
    with gen_embd.filesystem().open('test_y_list', 'rb') as f:
        test_y_list = pickle.load(f)
    
    # print('\n----XGB----')
    # xgb = XGBClassifier(nthread=4,
    #     n_jobs=-1,
    #     seed=RANDOM_SEED)

    # xgb.fit(train_embd_list, train_y_list)
    # cl_xgb = CalibratedClassifierCV(xgb, cv=5)
    # cl_xgb.fit(train_embd_list, train_y_list)
    # preds = cl_xgb.predict_proba(test_embd_list)[:, 1]
    # ret = print_metrics_binary(test_y_list, preds)
    # print(ret)

    print('\n----GBDT----')
    clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
    clf.fit(train_embd_list, train_y_list)
    cl_clf = CalibratedClassifierCV(clf, cv=5)
    cl_clf.fit(train_embd_list, train_y_list)
    preds = cl_clf.predict_proba(test_embd_list)[:, 1]
    ret = print_metrics_binary(test_y_list, preds)
    fpr, tpr, _ = metrics.roc_curve(test_y_list, preds)
    prc, rec, _ = precision_recall_curve(test_y_list, preds)
    #fpr = [fpr[i] for i in range(0, len(fpr), 2)]
    #tpr = [tpr[i] for i in range(0, len(tpr), 2)]
    #prc = [prc[i] for i in range(0, len(prc), 2)]
    #rec = [rec[i] for i in range(0, len(rec), 2)]
    print(list(fpr.round(5)))
    print(list(tpr.round(5)))
    print(list(prc.round(5)))
    print(list(rec.round(5)))
    print (ret)

@transform_pandas(
    Output(rid="ri.vector.main.execute.a08bd702-1974-4821-ae8c-a4004636984d"),
    gen_cmb_embd=Input(rid="ri.foundry.main.dataset.b664b449-52e1-483c-ac43-8d1a156ba6c0")
)
def train_xgb_cmb(gen_cmb_embd):
    with gen_cmb_embd.filesystem().open('train_embd_list_cmb', 'rb') as f:
        train_embd_list = pickle.load(f)
    with gen_cmb_embd.filesystem().open('val_embd_list_cmb', 'rb') as f:
        val_embd_list = pickle.load(f)
    with gen_cmb_embd.filesystem().open('test_embd_list_cmb', 'rb') as f:
        test_embd_list = pickle.load(f)
    with gen_cmb_embd.filesystem().open('train_y_list_cmb', 'rb') as f:
        train_y_list = pickle.load(f)
    with gen_cmb_embd.filesystem().open('val_y_list_cmb', 'rb') as f:
        val_y_list = pickle.load(f)
    with gen_cmb_embd.filesystem().open('test_y_list_cmb', 'rb') as f:
        test_y_list = pickle.load(f)
    
    # print('\n----XGB----')
    # xgb = XGBClassifier(nthread=4,
    #     n_jobs=-1,
    #     seed=RANDOM_SEED)

    # xgb.fit(train_embd_list, train_y_list)
    # cl_xgb = CalibratedClassifierCV(xgb, cv=5)
    # cl_xgb.fit(train_embd_list, train_y_list)
    # preds = cl_xgb.predict_proba(test_embd_list)[:, 1]
    # ret = print_metrics_binary(test_y_list, preds)
    # print(ret)

    print('\n----GBDT----')
    clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
    clf.fit(train_embd_list, train_y_list)
    cl_clf = CalibratedClassifierCV(clf, cv=5)
    cl_clf.fit(train_embd_list, train_y_list)
    preds = cl_clf.predict_proba(test_embd_list)[:, 1]
    ret = print_metrics_binary(test_y_list, preds)
    print (ret)

@transform_pandas(
    Output(rid="ri.vector.main.execute.63aa4a2a-65da-4811-8011-52efce4f4e42"),
    split_dataset=Input(rid="ri.foundry.main.dataset.a6c4b4c5-c9a7-46a0-a28f-13ef917c9262")
)
def unnamed(split_dataset):
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.e423deb9-2f23-407b-b624-a359a6565263"),
    feature_process=Input(rid="ri.foundry.main.dataset.55041bc9-f5d0-483a-b4a5-c13b0383c13b")
)
def unnamed_2(feature_process):
    with feature_process.filesystem().open('y', 'rb') as f:
        y = pickle.load(f)
    print(len(y[y==1]))
    print(len(y[y==0]))

@transform_pandas(
    Output(rid="ri.vector.main.execute.8c68a789-5cba-4280-a87a-dcba5ad51687"),
    build_graph=Input(rid="ri.foundry.main.dataset.c68f658f-c149-499c-9b7e-c40f6c57ddf9"),
    feature_process=Input(rid="ri.foundry.main.dataset.55041bc9-f5d0-483a-b4a5-c13b0383c13b"),
    split_dataset=Input(rid="ri.foundry.main.dataset.a6c4b4c5-c9a7-46a0-a28f-13ef917c9262"),
    train_model=Input(rid="ri.foundry.main.dataset.6c407803-217d-40c2-b0d4-3d113d628301")
)
def unnamed_3(split_dataset, build_graph, feature_process, train_model):
    with split_dataset.filesystem().open('x_val', 'rb') as f:
        x_val = pickle.load(f)
    with split_dataset.filesystem().open('y_val', 'rb') as f:
        y_val = pickle.load(f)
    with build_graph.filesystem().open('edge_pair', 'rb') as f:
        edge_pair = pickle.load(f)
    with build_graph.filesystem().open('idx2name', 'rb') as f:
        idx2name = pickle.load(f)
    with feature_process.filesystem().open('M', 'rb') as f:
        M = pickle.load(f)
    with train_model.filesystem().open('model', 'rb') as f:
        saved = pickle.load(f)
    node_dim = 128
    hidden_dim = 128
    mlp_dim = 128

    model = MedML(node_dim, hidden_dim, mlp_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss(reduction='sum')
    print(model)

    def get_graph_batch2(raw_x, node_embd, idx2name, edge_pair, M):
        print('------Start------\n')
        drug_idx2name, meas_idx2name, cond_idx2name, proc_idx2name = idx2name
        M1, M2, M3, M4, M5, M6, M7 = M
        cond_vec = raw_x[:, :M1]
        cond_val = raw_x[:, M1:2*M1]
        cond_idx = np.argwhere(cond_vec == 1)
        cond_vec = np.array([[cond_idx[idx, 0], feat_list.index(cond_idx2name[cond_idx[idx, 1]])] for idx in range(len(cond_idx))])
        if cond_vec.shape[0] > 0:
            cond_vec = np.concatenate([cond_vec, np.expand_dims(cond_val[cond_idx[:, 0], cond_idx[:, 1]], axis=-1), ], axis=1)
        else:
            cond_vec = np.array([[-1, -1, -1]])
        
        meas_vec = raw_x[:, 2*M1:2*M1+M2]
        meas_val = raw_x[:, 2*M1+M2:2*M1+2*M2]
        meas_idx = np.argwhere(meas_vec == 1)
        meas_vec = np.array([[meas_idx[idx, 0], feat_list.index(meas_idx2name[meas_idx[idx, 1]])] for idx in range(len(meas_idx))])
        if meas_vec.shape[0] > 0:
            meas_vec = np.concatenate([meas_vec, np.expand_dims(meas_val[meas_idx[:, 0], meas_idx[:, 1]], axis=-1), ], axis=1)
        else:
            meas_vec = np.array([[-1, -1, -1]])
        
        proc_vec = raw_x[:, 2*M1+2*M2:2*M1+2*M2+M3]
        proc_val = raw_x[:, 2*M1+2*M2+M3:2*M1+2*M2+2*M3]
        proc_idx = np.argwhere(proc_vec == 1)
        proc_vec = np.array([[proc_idx[idx, 0], feat_list.index(proc_idx2name[proc_idx[idx, 1]])] for idx in range(len(proc_idx))])
        if proc_vec.shape[0] > 0:
            proc_vec = np.concatenate([proc_vec, np.expand_dims(proc_val[proc_idx[:, 0], proc_idx[:, 1]], axis=-1), ], axis=1)
        else:
            proc_vec = np.array([[-1, -1, -1]])
        
        drug_vec = raw_x[:, 2*M1+2*M2+2*M3:2*M1+2*M2+2*M3+M4]
        drug_val = raw_x[:, 2*M1+2*M2+2*M3+M4:2*M1+2*M2+2*M3+2*M4]
        drug_idx = np.argwhere(drug_vec == 1)
        drug_vec = np.array([[drug_idx[idx, 0], feat_list.index(drug_idx2name[drug_idx[idx, 1]])] for idx in range(len(drug_idx))])
        if drug_vec.shape[0] > 0:
            drug_vec = np.concatenate([drug_vec, np.expand_dims(drug_val[drug_idx[:, 0], drug_idx[:, 1]], axis=-1), ], axis=1)
        else:
            drug_vec = np.array([[-1, -1, -1]])

        graph_vec = np.concatenate([cond_vec, meas_vec, proc_vec, drug_vec], axis=0)
        # number of patient, id of feature, value of feature
        batch_g = []
        adj_list= []
        batch_size = 0
        batch_flag = []
        batch_info = torch.tensor(raw_x[:, 2*M1+2*M2+2*M3+2*M4:], dtype=torch.float32).to(device)
        for i in range(len(raw_x)):
            cur_p = graph_vec[graph_vec[:, 0] == i, 1:]
            if cur_p.shape[0] > 0:
                cur_nodes = np.array(cur_p[:, 0], dtype=np.int32)
                cur_val = node_embd[cur_nodes]
                cur_val = torch.cat([torch.tensor(cur_p[:, 1], dtype=torch.float32).unsqueeze(-1).to(device), cur_val], dim=-1)
                batch_g.append(cur_val)
                cur_adj = np.eye(cur_nodes.shape[0])
                for i in range(cur_nodes.shape[0]):
                    for j in range(i+1, cur_nodes.shape[0]):
                        if (cur_nodes[i], cur_nodes[j]) in edge_pair:
                            cur_adj[i, j] = 1
                            cur_adj[j, i] = 1
                adj_list.append(cur_adj)
                batch_size += len(cur_nodes)
                batch_flag.append(1)
            else:
                batch_flag.append(0)
        print(batch_flag)
        batch_adj = torch.zeros((batch_size, batch_size), dtype=torch.int).to(device)
        cum_size = 0
        batch_mask = []

        for i in range(len(adj_list)):
            cur_size = len(adj_list[i])
            batch_adj[cum_size:cum_size+cur_size, cum_size:cum_size+cur_size] = torch.tensor(adj_list[i], dtype=torch.int).to(device)
            batch_mask.append([cum_size, cur_size])
            cum_size += cur_size
        batch_mask = torch.tensor(batch_mask, dtype=torch.int).to(device)
        batch_g = torch.cat(batch_g, dim=0)
        return batch_g, batch_adj, batch_mask, batch_flag, batch_info
    batch_g, batch_adj, batch_mask, batch_flag, batch_info = get_graph_batch2(x_val[10504:10506], model.node_embd, idx2name, edge_pair, M)
    model.load_state_dict(saved)
    print('Load weights')
    model.eval()
    pred, g_vec = model(batch_g, batch_adj, batch_mask)
    batch_flag = np.array(batch_flag)
    print(batch_flag.shape)
    print(g_vec.shape)
    print(g_vec[batch_flag == 1])

@transform_pandas(
    Output(rid="ri.vector.main.execute.9ddf230d-31b1-4a80-b9c2-c381d2bc608e"),
    gen_embd=Input(rid="ri.foundry.main.dataset.6bdab2d9-0632-438c-a342-43851da7ee07")
)
def unnamed_4(gen_embd):
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.598fae22-91c6-46ad-b027-e8621ec2d3ba"),
    feature_process=Input(rid="ri.foundry.main.dataset.55041bc9-f5d0-483a-b4a5-c13b0383c13b")
)
def unnamed_6(feature_process):
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ec0df1ca-df03-431d-ac19-3aed08ef0bcc"),
    task_1_gt_ten_thousand=Input(rid="ri.foundry.main.dataset.4e4865dc-429f-4e34-8001-2c221a06a786"),
    visit_occurrence=Input(rid="ri.foundry.main.dataset.911d0bb2-c56e-46bd-af4f-8d9611183bb7")
)
def visit_seq_vec( visit_occurrence, task_1_gt_ten_thousand):
    task_1_goldstandard = task_1_gt_ten_thousand
    person_list = task_1_goldstandard[['person_id', 'outpatient_visit_start_date']]
    visitT = visit_occurrence[['person_id','visit_start_date','visit_occurrence_id']]
    visitT = visitT.join(person_list, on='person_id').orderBy(["person_id", "visit_start_date"], ascending=[0, 1]).toPandas()
    visitT = visitT[(pd.to_datetime(visitT.outpatient_visit_start_date) - pd.to_datetime(visitT.visit_start_date)).apply(lambda x: x.days) >= 0]

    # person-level visit feature
    person_visit_seq = {}
    for idx, (person, tmp) in enumerate(visitT.groupby('person_id')):
        seq = ((pd.to_datetime(tmp.visit_start_date) - pd.to_datetime(tmp.visit_start_date.iloc[0])).apply(lambda x: x.days)).values.tolist()

        empty_vec = []
        # counts
        empty_vec.append(len(seq))
        # avg
        empty_vec.append(np.mean(np.diff(seq)))

        person_visit_seq[person] = np.nan_to_num(empty_vec).tolist()

    # output 
    output = Transforms.get_output()
    with output.filesystem().open('person_visit_seq_vec', 'wb') as f:
        pickle.dump(person_visit_seq, f)

    print (len(person_visit_seq))

    print ()
    
    return output

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.f2d70200-9d12-4cc6-a1c5-7fd33a4c5f57"),
    clean_visit=Input(rid="ri.foundry.main.dataset.33f9e714-5208-4061-bcef-fb0bace6124e")
)
def visit_type_vec(clean_visit):
    clean_visit = clean_visit
    clean_visit = clean_visit.toPandas()

    # visit type map
    type2idx = {}
    for idx, visit_type in enumerate(clean_visit.visit_concept_name.unique()):
        type2idx[visit_type] = idx

    # patient visit cnt vec
    patient_visit_cnt_vec = {}
    for person, tmp in clean_visit.groupby('person_id'):
        empty_vec = [0 for _ in range(len(type2idx))]
        for visit_type, tmp2 in tmp.groupby('visit_concept_name'):
            empty_vec[type2idx[visit_type]] = len(tmp2)
        patient_visit_cnt_vec[person] = empty_vec
    print(type2idx)
    # output 
    output = Transforms.get_output()
    with output.filesystem().open('patient_visit_cnt_vec', 'wb') as f:
        pickle.dump(patient_visit_cnt_vec, f)
    
    print (len(patient_visit_cnt_vec))
    
    return output

    
    
    

