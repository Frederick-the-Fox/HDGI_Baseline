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

path = '/home/hangni/HeCo-main/data/imdb/'

#adjs
adjs = []

#MAM
mam = np.load(path + 'mam.npz')
MAM = scipy.sparse.coo_matrix((mam['data'].astype(float), (mam['row'], mam['col'])), shape=(mam['shape'][0], mam['shape'][1]))
adjs.append(MAM)

#MDM
mdm = np.load(path + 'mdm.npz')
MDM = scipy.sparse.coo_matrix((mdm['data'].astype(float), (mdm['row'], mdm['col'])), shape=(mdm['shape'][0], mdm['shape'][1]))
adjs.append(MDM)

#MKM
mkm = np.load(path + 'mkm.npz')
MKM = scipy.sparse.coo_matrix((mkm['data'].astype(float), (mkm['row'], mkm['col'])), shape=(mkm['shape'][0], mkm['shape'][1]))
adjs.append(MKM)

# #feature
# feat = np.load('/home/hangni/WangYC/imdb_process/m_feat.npz')
# feature = scipy.sparse.coo_matrix((feat['data'].astype(float), (feat['row'], feat['col'])), shape=(feat['shape'][0], feat['shape'][1]))
# # feature = feature.A
# print('feature:{}'.format(feature))

# #label
# labels = np.load(path + 'labels.npy')
# label = encode_onehot(labels).astype(int)
# # print("label:{}".format(label))

#feature
feat = np.load('/home/hangni/WangYC/imdb_process/m_feat.npz')
feature = scipy.sparse.coo_matrix((feat['data'].astype(float), (feat['row'], feat['col'])), shape=(feat['shape'][0], feat['shape'][1]))
print("feature:{}".format(feature))

#label
labels = np.genfromtxt('/home/hangni/WangYC/imdb_process/m_label.txt')
label = encode_onehot(labels[:, 1]).astype(int)
print('label', label)

#idx_20
test_idx_20 = np.load('/home/hangni/HeCo-main/data/my_data/imdb/test_20.npy')
train_idx_20 = np.load('/home/hangni/HeCo-main/data/my_data/imdb/train_20.npy')
val_idx_20 = np.load('/home/hangni/HeCo-main/data/my_data/imdb/val_20.npy')

#idx_40
test_idx_40 = np.load('/home/hangni/HeCo-main/data/my_data/imdb/test_40.npy')
train_idx_40 = np.load('/home/hangni/HeCo-main/data/my_data/imdb/train_40.npy')
val_idx_40 = np.load('/home/hangni/HeCo-main/data/my_data/imdb/val_40.npy')

#idx_60
test_idx_60 = np.load('/home/hangni/HeCo-main/data/my_data/imdb/test_60.npy')
train_idx_60 = np.load('/home/hangni/HeCo-main/data/my_data/imdb/train_60.npy')
val_idx_60 = np.load('/home/hangni/HeCo-main/data/my_data/imdb/val_60.npy')

#idx_eval
test_idx_eval = np.load('/home/hangni/HeCo-main/data/my_data/imdb/eval_test_40.npy')
train_idx_eval = np.load('/home/hangni/HeCo-main/data/my_data/imdb/eval_train_40.npy')
val_idx_eval = np.load('/home/hangni/HeCo-main/data/my_data/imdb/eval_val_40.npy')

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
pkl.dump(file_data, open('imdb.pkl',"wb"), protocol=4)
print('saved')