import numpy as np
import scipy.sparse 
from sklearn.preprocessing import OneHotEncoder
import pickle as pkl
import sys
import scipy.io as scio
from sklearn.metrics import roc_curve, f1_score

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

path = '/home/hangni/HeCo-main/data/dblp/'

#adjs
adjs = []
# #pa
# adj = np.genfromtxt('/home/hangni/HeCo-main/data/dblp/pa.txt', dtype = int)
# row = adj[:, 0]
# col = adj[:, 1]
# data = []
# for i in range(adj.shape[0]):
#     data.append(1.0)
# data = np.asarray(data)
# adj_s = scipy.sparse.coo_matrix((data, (row, col)), shape=(adj.shape[0], adj.shape[0]))
# adjs.append(adj_s)

# #pc
# adj = np.genfromtxt('/home/hangni/HeCo-main/data/dblp/pc.txt', dtype = int)
# row = adj[:, 0]
# col = adj[:, 1]
# data = []
# for i in range(adj.shape[0]):
#     data.append(1.0)
# data = np.asarray(data)
# adj_s = scipy.sparse.coo_matrix((data, (row, col)), shape=(adj.shape[0], adj.shape[0]))
# adjs.append(adj_s)

# #pt
# adj = np.genfromtxt('/home/hangni/HeCo-main/data/dblp/pt.txt', dtype = int)
# row = adj[:, 0]
# col = adj[:, 1]
# data = []
# for i in range(adj.shape[0]):
#     data.append(1.0)
# data = np.asarray(data)
# adj_s = scipy.sparse.coo_matrix((data, (row, col)), shape=(adj.shape[0], adj.shape[0]))
# adjs.append(adj_s)

#APA
apa = np.load(path + 'apa.npz')
APA = scipy.sparse.coo_matrix((apa['data'].astype(float), (apa['row'], apa['col'])), shape=(apa['shape'][0], apa['shape'][1]))
# APA = (APA.A).astype(float)
adjs.append(APA)

#APCPA
apcpa = np.load(path + 'apcpa.npz')
APCPA = scipy.sparse.coo_matrix((apcpa['data'].astype(float), (apcpa['row'], apcpa['col'])), shape=(apcpa['shape'][0], apcpa['shape'][1]))
# APCPA = (APCPA.A).astype(float)
adjs.append(APCPA)

#APTPA
aptpa = np.load(path + 'aptpa.npz')
APTPA = scipy.sparse.coo_matrix((aptpa['data'].astype(float), (aptpa['row'], aptpa['col'])), shape=(aptpa['shape'][0], aptpa['shape'][1]))
# APTPA = (APTPA.A).astype(float)
adjs.append(APTPA)

#feature
feat = np.load(path + 'a_feat.npz')
feature = scipy.sparse.csr_matrix((feat['data'].astype(float),feat['indices'], feat['indptr']), shape=(feat['shape'][0], feat['shape'][1]))
# feature = (feature.A).astype(float)
print('feature:{}'.format(feature))

#label
labels = np.load(path + 'labels.npy')
label = encode_onehot(labels).astype(int)
print("label:{}".format(label))

#idx_20
test_idx_20 = np.load('/home/hangni/HeCo-main/data/my_data/dblp/test_20.npy')
train_idx_20 = np.load('/home/hangni/HeCo-main/data/my_data/dblp/train_20.npy')
val_idx_20 = np.load('/home/hangni/HeCo-main/data/my_data/dblp/val_20.npy')

#idx_40
test_idx_40 = np.load('/home/hangni/HeCo-main/data/my_data/dblp/test_40.npy')
train_idx_40 = np.load('/home/hangni/HeCo-main/data/my_data/dblp/train_40.npy')
val_idx_40 = np.load('/home/hangni/HeCo-main/data/my_data/dblp/val_40.npy')

#idx_60
test_idx_60 = np.load('/home/hangni/HeCo-main/data/my_data/dblp/test_60.npy')
train_idx_60 = np.load('/home/hangni/HeCo-main/data/my_data/dblp/train_60.npy')
val_idx_60 = np.load('/home/hangni/HeCo-main/data/my_data/dblp/val_60.npy')

#idx_eval
test_idx_eval = np.load('/home/hangni/HeCo-main/data/my_data/dblp/eval_test_40.npy')
train_idx_eval = np.load('/home/hangni/HeCo-main/data/my_data/dblp/eval_train_40.npy')
val_idx_eval = np.load('/home/hangni/HeCo-main/data/my_data/dblp/eval_val_40.npy')

idx_train_list = []
idx_val_list = []
idx_test_list = []
idx_train_list.append(train_idx_eval)
idx_train_list.append(train_idx_20)
idx_train_list.append(train_idx_40)
idx_train_list.append(train_idx_60)
idx_test_list.append(test_idx_eval)
idx_test_list.append(test_idx_20)
idx_test_list.append(test_idx_40)
idx_test_list.append(test_idx_60)
idx_val_list.append(val_idx_eval)
idx_val_list.append(val_idx_20)
idx_val_list.append(val_idx_40)
idx_val_list.append(val_idx_60)

file_data = {'adjs':adjs, 'features':feature, 'labels':label,
            'idx_train_list':idx_train_list, 'idx_val_list':idx_val_list, 'idx_test_list':idx_test_list
            }
pkl.dump(file_data, open('dblp.pkl',"wb"), protocol=4)
print('saved')