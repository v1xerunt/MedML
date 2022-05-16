import numpy as np
import torch
from sklearn import metrics
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