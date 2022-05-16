from pyspark.sql.functions import lower, col,lit
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
import os
import random
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from utils import *
from model import *


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


def train_xgb_0(gen_embd):
    with gen_embd.filesystem().open('train_embd_list', 'rb') as f:
        train_embd_list = pickle.load(f)
    with gen_embd.filesystem().open('test_embd_list', 'rb') as f:
        test_embd_list = pickle.load(f)
    with gen_embd.filesystem().open('train_y_list', 'rb') as f:
        train_y_list = pickle.load(f)
    with gen_embd.filesystem().open('test_y_list', 'rb') as f:
        test_y_list = pickle.load(f)

    print('\n----GBDT----')
    clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
    clf.fit(train_embd_list, train_y_list)
    cl_clf = CalibratedClassifierCV(clf, cv=5)
    cl_clf.fit(train_embd_list, train_y_list)
    preds = cl_clf.predict_proba(test_embd_list)[:, 1]
    ret = print_metrics_binary(test_y_list, preds)
    fpr, tpr, _ = metrics.roc_curve(test_y_list, preds)
    prc, rec, _ = precision_recall_curve(test_y_list, preds)
    print (ret)