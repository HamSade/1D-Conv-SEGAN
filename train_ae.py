# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:36:39 2018

@author: hamed
"""

from __future__ import absolute_import
from __future__ import print_function
###########################################
import argparse
import os

###########################################
import torch
from torch import optim
#from torch.autograd import Variable
#import numpy as np
from time import time

###########################################
from dataset_loader import dataset_class, data_loader  #TODO: Mat
from models import AECONV

###########################################
import pdb
import colored_traceback; colored_traceback.add_hook()

#%% Args setup
parser = argparse.ArgumentParser()

parser.add_argument('--n_times',  type=int, default=7)  #TODO: set
parser.add_argument('--n_feats',  type=int, default=600)  #TODO: set
parser.add_argument('--n_latent',  type=int, default=50)  #TODO: set
parser.add_argument('--n_classes',  type=int, default=2)  #TODO: set

################## Data
parser.add_argument('--data', type=str, help='Path to the data directory',
                    default=os.path.join(os.path.dirname(__file__), '../data/...')) #TODO: to be completed for data loading
parser.add_argument('--log_dir', type=str, help='Directory relative which all output files are stored',
                    default='/log/') 
parser.add_argument('--small_part', dest='small_part', action='store_false')    

################## Training                                                    
parser.add_argument('--n_iters', type=int, default=100000)                    
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.9,help='beta_1 for Adam')
parser.add_argument('--beta_2', type=float, default=0.999,
                    help='beta_2 for Adam')  
                    
parser.add_argument('--print_every', type=int, default=50)
parser.add_argument('--save_every', type=int, default=1000)
parser.add_argument('--save_mode', type=str, default="best")
parser.add_argument('--log_train_file', type=str, default="log/train.log")
parser.add_argument('--save_path', type=str, default="saved_models/ae_model")
  
    
#################### Finalizing args
#parser.set_defaults(small_part=False)
args = parser.parse_args()
print("args = ", args)

#%% ############### Training the Model

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def train(x, targets, model, opts, losses):
    
    enc_opt, dec_opt, cls_opt = opts
    ae_loss, cls_loss = losses    
    
    enc_opt.zero_grad()
    dec_opt.zero_grad()
    cls_opt.zero_grad()
   
    dec_out, cls_ = model(x, ae_loss)

    ae_loss_  = ae_loss(x, dec_out)
    cls_loss_ = cls_loss(cls_, targets)
    
    ae_loss_.backward()#retain_graph=True)
    dec_opt.step() #TODO: the order matters
    enc_opt.step()
    
    cls_loss_.backward()#retain_graph=True)
    cls_opt.step() #TODO: the order matters
    enc_opt.step()

    return ae_loss_.item(), cls_loss_.item()
    
#%%
def trainIters(train_data, train_targets, model):
    start = time()
    print_loss_ae = 0  # Reset every print_every
        
    ae_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    print(ae_params)
    
    # SGD
#    enc_opt = optim.SGD(model.encoder.parameters(), lr=learning_rate)
#    dec_opt = optim.SGD(model.decoder.parameters(), lr=learning_rate)  
#    cls_opt = optim.SGD(model.decoder.parameters(), lr=learning_rate) 
    
    # ADAM
    enc_opt    = optim.Adam( model.encoder.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    dec_opt    = optim.Adam( model.decoder.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    cls_opt    = optim.Adam( model.classifier.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        
    opts = (enc_opt, dec_opt, cls_opt)
    
    ae_loss = torch.nn.MSELoss()
    cls_loss = torch.nn.BCELoss()  #TODO: change if required based off your targets representation
    losses = (ae_loss, cls_loss)
    
    print_loss_ae = 0
    
    # Logging and saving
    with open(args.log_train_file, 'a') as log_tf:
        pass # Just to remove previous logs
            
    ## Training iterations
    for step in range(1, args.n_iters + 1):
            
        ae_loss_, cls_loss_ = train(train_data, train_targets, model, opts, losses)
        
        # Logging
        if args.log_train_file:
            with open(args.log_train_file, 'a') as log_tf:
                log_tf.write('{iteration},{ae_loss:3.6f},{cls_loss:3.6f}\n'.format(
                    iteration=step, ae_loss=ae_loss_, cls_loss=cls_loss_))
                    
        # AE loss print
        print_loss_ae += ae_loss
        if step % args.print_every == 0:
            print("iteration = ", step)
            print("progress= {}, ae_loss= {}, cls_loss= {}, time_elapsed= {}".format(step*100./args.n_iters, ae_loss_,
                  cls_loss_, time()-start))
                  
        # Saving model              
        if step  % args.save_every == 0:
            checkpoint = {'model': model.state_dict(),                    
                          'args': args,
                          'iteration': step}

            if args.save_mode == 'all':
                model_name = args.save_path + '_ael_{ae_loss:2.6f}_cll_{cls_loss:2.6f}.chkpt'.format(
                ael=ae_loss_, cls_loss=cls_loss_)
                torch.save(checkpoint, model_name)
                
            elif args.save_mode == 'best':
                model_name = args.save_path + '.chkpt'
                torch.save(checkpoint, model_name)

#%% Training
def main():
    
    # Loading Data #TODO: Needs to be changed it according your loader
    dataset_ = dataset_class(phase="train")
    train_data, train_targets = data_loader(dataset_, args.batch_size) #TODO: Assuming train_data is the actual batch sample of size [bs x n_times x n_feats] maybe after next(iter(data_loader_obj))
     
    device = torch.device('cuda' if torch.cuda.device_count() != 0 else 'cpu')    
    model = AECONV(args.K, args.input_size, args.hidden_size, args.output_size, device)#.cuda(device=device)
    #training
    trainIters(train_data, train_targets, model)
                            
#%%
if __name__ == "__main__":
    main()
    
    
    