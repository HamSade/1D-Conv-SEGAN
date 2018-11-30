# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:19:12 2018

@author: hamed
"""

# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
#import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%%
class AECONV(nn.Module):    
    def __init__(self, n_times, n_feats, n_latent, device, n_classes=2):
        
        super(AECONV, self).__init__()
        self.device = device
        self.encoder = EncoderCNN(n_times, n_feats, n_latent).cuda(device=device)
        self.decoder = DecoderCNN(n_times, n_feats, n_latent).cuda(device=device)
        self.classifier = Classifier(n_times, n_latent, n_classes)
        
    ################################################################################         
    def forward(self, x):    
        x = x.float().cuda(device=self.device)
        enc_out = self.encoder(x)   
        class_ = self.classifier(enc_out)            
        out = self.decoder(enc_out)
        return out, class_
        
#%%
def init_weights(self):
        """Initialize weights for convolution layers using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)
                
#%% The Encoder
class EncoderCNN(nn.Module):
    """ Encoder gets a seq batch of size [bs x n_years x n_feats]
     We would keep the sequence length (n_times) CONSTANT through convs"""
    def __init__(self, n_times, n_feats, n_latent):
        super(EncoderCNN, self).__init__()
                
        h1 = int(n_feats//2) #number of channels for the first conv layer, EX: n_feats = 600 ==> h1=300
        h2 = int(n_feats//4) #EX: n_feats = 600 ==> h1=300
        h3 = int(n_feats//8) #EX: n_feats = 600 ==> h1=300
        n_ker = 3 #kernel size
        n_pad = int( (n_ker-1)/2)
        
        self.enc1 = nn.Conv1d(in_channels=n_feats, out_channels=h1, kernel_size=n_ker, stride=1, padding=n_pad)  # [bs x n_times x h1]
        self.enc1_nl = nn.PReLU()
        self.enc2 = nn.Conv1d(in_channels=h1, out_channels=h2, kernel_size=n_ker, stride=1, padding=n_pad)  # [bs x n_times x h2]
        self.enc2_nl = nn.PReLU()
        self.enc3 = nn.Conv1d(in_channels=h2, out_channels=h3, kernel_size=n_ker, stride=1, padding=n_pad)  # [bs x n_times x h3]
        self.enc3_nl = nn.PReLU()
        self.enc4 = nn.Conv1d(in_channels=h3, out_channels=n_latent, kernel_size=n_ker, stride=1, padding=n_pad)  # [bs x n_times x n_latent]
        
        # initialize weights
        self.init_weights()
        
    def forward(self, x, hidden):
        x = x.transpose(1,2) #[bs, n_times, n_feats] ---> [bs, n_feats, n_times]
        output = x #.contiguous().view(1, 1, -1)
        
        output = self.enc1(output)
        output = self.enc1_nl(output)
        output = self.enc2(output)
        output = self.enc2_nl(output)
        output = self.enc3(output)
        output = self.enc3_nl(output)
        output = self.enc4_nl(output)
        
        return output #[bs, n_latent, n_times]

#%%
class DecoderCNN(nn.Module):
    def __init__(self, n_times, n_feats, n_latent):
        super(DecoderCNN, self).__init__()

        h1 = int(n_feats//2) #number of channels for the first conv layer, EX: n_feats = 600 ==> h1=300
        h2 = int(n_feats//4) #EX: n_feats = 600 ==> h2= 150
        h3 = int(n_feats//8) #EX: n_feats = 600 ==> h1=75
        n_ker = 3 #kernel size
        n_pad = int((n_ker-1)/2)
        
        self.dec4 = nn.Conv1d(in_channels=n_latent, out_channels=h3, kernel_size=n_ker, stride=1, padding=n_pad)  # [bs x h3 x  n_times]
        self.dec4_nl = nn.PReLU()
        self.dec3 = nn.Conv1d(in_channels=h3, out_channels=h2, kernel_size=n_ker, stride=1, padding=n_pad)  # [bs x h2 x n_times]
        self.dec3_nl = nn.PReLU()
        self.dec2 = nn.Conv1d(in_channels=h2, out_channels=h1, kernel_size=n_ker, stride=1, padding=n_pad)  # [bs x h1 x n_times]
        self.dec2_nl = nn.PReLU()
        self.dec1 = nn.Conv1d(in_channels=h1, out_channels=n_feats, kernel_size=n_ker, stride=1, padding=n_pad)  # [bs x n_feats x n_times]
        
        # initialize weights
        self.init_weights()
        
    def forward(self, x):
        output = x
        
        output = self.dec4(output)
        output = self.dec4_nl(output)
        output = self.dec3(output)
        output = self.dec3_nl(output)
        output = self.dec2(output)
        output = self.dec2_nl(output)
        output = self.dec1(output)

        return output.transpose(1,2) #[bs, n_times, n_latent]
        
#%%
class Classifier(nn.Module):
    def __init__(self, n_times, n_latent, n_classes):
        super(Classifier, self).__init__()   
        
        # Encoder output = #[bs, n_latent, n_times]
        n_latent_times = n_times * n_latent
        h1 = int(n_latent_times//2)
        h2 = int(n_latent_times//4)
        self.fnn1 = nn.Linear(n_latent_times, h1)
        self.fnn2 = nn.Linear(h1, h2)
        self.fnn3 = nn.Linear(h2, n_classes)
        self.hid_softmax = nn.Softmax(dim=1)

    def forward(self, h):        
        h = self.fnn1(h)
        h = self.fnn2(h)
        h = self.fnn3(h)
        class_ = self.hid_softmax(h)
        return class_
                   
