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

x_train = pickle.load(open('./data/x_train.pkl', 'rb'))
x_val = pickle.load(open('./data/x_val.pkl', 'rb'))
x_test = pickle.load(open('./data/x_test.pkl', 'rb'))

demo_train = pickle.load(open('./data/demo_train.pkl', 'rb'))
demo_val = pickle.load(open('./data/demo_val.pkl', 'rb'))
demo_test = pickle.load(open('./data/demo_test.pkl', 'rb'))

y_train = pickle.load(open('./data/y_train.pkl', 'rb'))
y_val = pickle.load(open('./data/y_val.pkl', 'rb'))
y_test = pickle.load(open('./data/y_test.pkl', 'rb'))

edge_pair = pickle.load(open('./data/edge_pair.pkl', 'rb'))

feat_dim = x_train.shape[-1]
node_dim = 128
hidden_dim = 128
mlp_dim = 128
num_layers = 1
epoch = 10
batch_size = 128


model = MedML(feat_dim, node_dim, hidden_dim, mlp_dim, num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss(reduction='sum')
print(model)


# Train the GAT model
best_loss = 0
for each_epoch in range(epoch):
    train_loss = 0
    model.train()
    for i in range(0, len(x_train), batch_size):
        batch_g, batch_adj, batch_mask, batch_flag, batch_demo = get_graph_batch(x_train[i:i+batch_size], demo_train[i:i+batch_size], model.node_embd, edge_pair, device, directed=False)
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
            batch_g, batch_adj, batch_mask, batch_flag, batch_demo = get_graph_batch(x_val[i:i+batch_size], demo_val[i:i+batch_size], model.node_embd, edge_pair, device, directed=False)
            batch_y = torch.tensor(y_val[i:i+batch_size], dtype=torch.float32).to(device).unsqueeze(-1)[np.array(batch_flag)==1]
            pred, _ = model(batch_g, batch_adj, batch_mask)
            loss = criterion(pred, batch_y)
            val_loss += loss.item()
            y_true += list(batch_y.cpu().detach().numpy().reshape(-1))
            y_pred += list(pred.cpu().detach().numpy().reshape(-1))

    ret = print_metrics_binary(y_true, y_pred, verbose=0)
    if ret['auprc'] > best_loss:
        best_loss = ret['auprc']
        torch.save(model.state_dict(), './model/saved_weights.pkl')
    print('Epoch: {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f} Val ROC: {:.4f} Val PRC: {:.4f}'.format(each_epoch, train_loss/len(x_train), val_loss/len(x_val), ret['auroc'], ret['auprc']))

# Generate embeddings using trained MedML model
model.load_state_dict(torch.load('./model/saved_weights.pkl'))
print('Load weights')
model.eval()
train_embd = []
for i in range(0, len(x_train), batch_size):
    batch_g, batch_adj, batch_mask, batch_flag, batch_demo = get_graph_batch(x_train[i:i+batch_size], demo_train[i:i+batch_size], model.node_embd, edge_pair, device, directed=False)
    pred, g_vec = model(batch_g, batch_adj, batch_mask)
    ehr_vec = torch.zeros(batch_demo.size(0), g_vec.size(1)).to(device)
    ehr_vec[batch_flag == 1] = g_vec
    origin_val_vec = torch.tensor(x_train[i:i+batch_size], dtype=torch.float32).to(device)
    embd_vec = torch.cat([ehr_vec, origin_val_vec, batch_demo], dim=-1)
    train_embd.append(embd_vec)
train_embd = torch.cat(train_embd, dim=0).cpu().detach().numpy()

test_embd = []
for i in range(0, len(x_test), batch_size):
    batch_g, batch_adj, batch_mask, batch_flag, batch_demo = get_graph_batch(x_test[i:i+batch_size], demo_test[i:i+batch_size], model.node_embd, edge_pair, device, directed=False)
    pred, g_vec = model(batch_g, batch_adj, batch_mask)
    ehr_vec = torch.zeros(batch_demo.size(0), g_vec.size(1)).to(device)
    ehr_vec[batch_flag == 1] = g_vec
    origin_val_vec = torch.tensor(x_test[i:i+batch_size], dtype=torch.float32).to(device)
    embd_vec = torch.cat([ehr_vec, origin_val_vec, batch_demo], dim=-1)
    test_embd.append(embd_vec)
test_embd = torch.cat(test_embd, dim=0).cpu().detach().numpy()

# Train and test with downstream models

clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
clf.fit(train_embd, y_train)
cl_clf = CalibratedClassifierCV(clf, cv=5)
cl_clf.fit(train_embd, y_train)
preds = cl_clf.predict_proba(test_embd)[:, 1]
ret = print_metrics_binary(y_test, preds)
print (ret)