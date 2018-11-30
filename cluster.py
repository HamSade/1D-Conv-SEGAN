# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:50:01 2018

@author: hamed
"""

import  numpy as np

import kmeans+ as km


#%%
pkl_filename = "saved_models/kmeans_op.pkl"

test_data = 
with open(pkl_filename, 'rb') as file:  
    pickle_model = pickle.load(file)
    
    

#%% Load data & Find embedding for all data points
device = torch.device('cuda' if torch.cuda.device_count() != 0 else 'cpu')
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

model = model.to(device)
train_data = Tensor( np.random.rand(args.batch_size, args.n_times, args.n_feats) )  #TODO: fix pls

print("Embedding started")
start = time()
with torch.no_grad():
    emb_train_data, _ = model(train_data)
print("Embedding finished {} minutes to embed all training data points".format( 1/60. * ( time()-start ) ))

emb_train_data = emb_train_data.cpu().numpy()


#%%