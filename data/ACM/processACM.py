import scipy.io as scio
import scipy.sparse as sp
import pickle 
import numpy as np 

dataFile = 'ACM3025.mat'
data = scio.loadmat(dataFile)
#a = data['feature']
features = sp.coo_matrix(data['feature'])

with open('paper_features_1870.pickle', 'wb') as f:
         pickle.dump(features, f)
f.close

labels = data['label']
labels = np.where(labels)[1]
index_label = []
index = 0
for label in labels:
    index_label.append(str(index)+","+str(label))
    index += 1 
with open('./index_label.txt', 'w') as idx_label:
        for item in index_label:
            idx_label.write(item)
            idx_label.write('\n')         
idx_label.close    

for adj_type in ['PTP','PLP','PAP']:
    adj_matrix = sp.coo_matrix(data[adj_type])
    with open('{}_adj.pickle'.format(adj_type), 'wb') as f:
         pickle.dump(adj_matrix, f)
    f.close

homo_movie_adj = sp.coo_matrix(data['PTP']) + sp.coo_matrix(data['PLP']) + sp.coo_matrix(data['PAP'])
with open('homo_movie_adj.pickle', 'wb') as f:
         pickle.dump(homo_movie_adj, f)
f.close 