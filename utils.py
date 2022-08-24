import numpy as np
import torch
from sklearn import metrics
import os
import random

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



def get_graph_batch(feat_x, demo_x, node_embd, edge_pair, device, directed=False):
    batch_g = []
    adj_list= []
    batch_size = 0
    batch_flag = []
    batch_demo = torch.tensor(demo_x, dtype=torch.float32).to(device)
    for i in range(len(feat_x)):
        cur_p = feat_x[i, :]
        cur_idx = np.argwhere(cur_p >= 1).reshape(-1)
        if cur_idx.shape[0] > 0:
            cur_val = node_embd[cur_idx]
            cur_val = torch.cat([torch.tensor(cur_p[cur_idx], dtype=torch.float32).unsqueeze(-1).to(device), cur_val], dim=-1)
            batch_g.append(cur_val)
            cur_adj = np.eye(cur_idx.shape[0])
            for i in range(cur_idx.shape[0]):
                for j in range(cur_idx.shape[0]):
                    if (cur_idx[i], cur_idx[j]) in edge_pair:
                        cur_adj[i, j] = 1
                        if directed == False:
                            cur_adj[j, i] = 1
            adj_list.append(cur_adj)
            batch_size += len(cur_idx)
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
    return batch_g, batch_adj, batch_mask, np.array(batch_flag), batch_demo