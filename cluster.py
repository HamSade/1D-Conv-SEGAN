# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:50:01 2018

@author: hamed
"""

import torch
import  numpy as np
import pickle
from time import time

#%% Loading kmeans object and args
kmeans_obj_filename = "saved_models/kmeans_obj.pkl"
kmeans_args_filename = "saved_models/kmeans_args.pkl"

with open(kmeans_obj_filename, 'rb') as f:  
    kmeans_obj = pickle.load(f)
with open(kmeans_args_filename, 'rb') as f:  
    kmeans_args = pickle.load(f)
    
#%% AE model and args loading
print("Loading AE model started"); start=time()
checkpoint = torch.load(kmeans_args.ae_load_path)
print("Loading AE model finished in {} secs".format(time()-start))

model = checkpoint['model']
model.eval()
args_ae = checkpoint['args']

#%% Load data & Find embedding for all data points
device = torch.device('cuda' if torch.cuda.device_count() != 0 else 'cpu')
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

model = model.cuda(device)
test_data = Tensor( np.random.rand(args_ae.batch_size, args_ae.n_times, args_ae.n_feats) )  #TODO: fix pls

print("Embedding started")
start = time()
with torch.no_grad():
    _, _, emb_test_data, cls_last_layer_ = model(test_data)
print("Embedding finished {} minutes to embed all training data points".format( 1/60. * ( time()-start ) )) 

#%% Training Kmeans obj
print("MiniBatch-Kmeans started")
start = time()
if kmeans_args.last_layer:
    print('*'*20, "Classifier last layer was selected")
    cls_last_layer_ = emb_test_data.cpu().numpy()
    kmeans_obj.predict(cls_last_layer_)
else:
    print('*'*20, "Embeddings were selected")
    emb_train_data = emb_test_data.cpu().numpy()
    kmeans_obj.predict(emb_test_data)
print("MiniBatch-Kmeans finished after {} seconds".format( 1/60. * ( time()-start ) ))
