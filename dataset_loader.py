# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 16:29:43 2018

@author: hamed
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

#%%
class dataset_class(Dataset):
    def __init__(self, phase="train"):
        
        if phase == "train":
            self.dataset = np.random.rand(1000, 7, 600)  #TODO: fix. Just for demonstration  
            self.targets = np.random.randint(low=0, high=2, size=(1000)) #TODO: fix. Just for demonstration  
        else: #test
            self.dataset = np.random.rand(100, 7, 600) #TODO: fix. Just for demonstration 
            self.targets = np.random.randint(low=0, high=2, size=(1000)) #TODO: fix. Just for demonstration 
            
    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return self.dataset[idx]
    
#%% 
def data_loader(dataset__, batch_size = 32 , shuffle=True, num_workers=0):
#    if shuffle:
#        dataset__.data.random_shuffle()
    return DataLoader(dataset__, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=num_workers)
