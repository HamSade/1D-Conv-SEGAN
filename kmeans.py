# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:03:17 2018

@author: hamed
"""


from __future__ import unicode_literals, print_function, division
import random

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import numpy
import sklearn

from models import AECONV

#%%

n_times, n_feats, n_latent, device, n_classes=2

model = AECONV(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()