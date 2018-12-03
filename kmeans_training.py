# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:03:17 2018

@author: hamed
"""
from __future__ import unicode_literals, print_function, division

import torch
import numpy as np
#import pdb

import numpy
from sklearn.cluster import MiniBatchKMeans
import pickle
import argparse
from time import time

#from models import AECONV
import colored_traceback; colored_traceback.add_hook()

'''Here we perform Kmeans training'''

#%%
parser = argparse.ArgumentParser()

parser.add_argument('--K',  type=int, default=5)  #TODO: set 
parser.add_argument('--batch_size',  type=int, default=32)  #TODO: set
parser.add_argument('--max_iter',  type=int, default=100)  #TODO: set

parser.add_argument('--ae_load_path', type=str, default="saved_models/ae_model.chkpt")
parser.add_argument('--save_path', type=str, default="saved_models/")
parser.add_argument('--last_layer', type=bool, default=False)  #TODO: determine you want kmeans on latent code or classifier last layer

args = parser.parse_args()
print("args = ", args)

#%% AE model and args loading
print("Loading model started"); start=time()
checkpoint = torch.load(args.ae_load_path)
print("Loading model finished in {} secs".format(time()-start))

model = checkpoint['model']
model.eval()
args_ae = checkpoint['args']

#%% Load data & Find embedding for all data points
device = torch.device('cuda' if torch.cuda.device_count() != 0 else 'cpu')
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

model = model.cuda(device)
train_data = Tensor( np.random.rand(args_ae.batch_size, args_ae.n_times, args_ae.n_feats) )  #TODO: fix pls

print("Embedding started")
start = time()
with torch.no_grad():
    _, _, emb_train_data, cls_last_layer_ = model(train_data)
print("Embedding finished {} minutes to embed all training data points".format( 1/60. * ( time()-start ) )) 

#%% Kmeans obj definition
# kmeans_obj attributes =  cluster_centers_, labels_, inertia_
kmeans_obj = MiniBatchKMeans(n_clusters=args.K, init='k-means++', max_iter=args.max_iter,
                                 batch_size=args.batch_size, verbose=0, compute_labels=True,
                                 random_state=0, tol=0.0, max_no_improvement=10,
                                 init_size=None, n_init=3, reassignment_ratio=0.01)

#%% Training Kmeans obj
print("MiniBatch-Kmeans started")
start = time()
if args.last_layer:
    print('*'*20, "Classifier last layer was selected")
    cls_last_layer_ = emb_train_data.cpu().numpy()
    kmeans_obj.fit(cls_last_layer_)
else:
    print('*'*20, "Embeddings were selected")
    emb_train_data = emb_train_data.cpu().numpy()
    kmeans_obj.fit(emb_train_data)
print("MiniBatch-Kmeans finished after {} seconds".format( 1/60. * ( time()-start ) ))

#%% Saving Kmeans op obj
kmeans_filename = args.save_path + "kmeans_obj.pkl"
args_filename = args.save_path + "kmeans_args.pkl"
with open(kmeans_filename, 'wb') as f:  
    pickle.dump(kmeans_obj, f)
with open(args_filename, 'wb') as f:  
    pickle.dump(args, f)
