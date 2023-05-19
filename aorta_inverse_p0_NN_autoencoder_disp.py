# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 19:32:48 2023

@author: liang
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import sys
sys.path.append("c3d8")
sys.path.append("mesh")
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark=True
from PolyhedronMesh import PolyhedronMesh
from aorta_mesh import get_solid_mesh_cfg
import time
#%%
class Encoder(torch.nn.Module):
    def __init__(self, h_dim, c_dim, n_layers):
        super().__init__()
        layer=[torch.nn.Linear(10000*3, h_dim)]
        for n in range(0, n_layers):
            layer.append(torch.nn.Linear(h_dim, h_dim))
            layer.append(torch.nn.Softplus())
        layer.append(torch.nn.Linear(h_dim, c_dim))
        self.layer=torch.nn.Sequential(*layer)
    def forward(self, x):
        #x.shape (10000, 3)
        x=x.view(1, -1)
        y=self.layer(x)
        return y
#%%
class Decoder(torch.nn.Module):
    def __init__(self, h_dim, c_dim, n_layers):
        super().__init__()
        layer=[torch.nn.Linear(c_dim, h_dim)]
        for n in range(0, n_layers):
            layer.append(torch.nn.Linear(h_dim, h_dim))
            layer.append(torch.nn.Softplus())
        layer.append(torch.nn.Linear(h_dim, 10000*3))
        self.layer=torch.nn.Sequential(*layer)
    def forward(self, x):
        #x.shape (?, in_dim)
        y=self.layer(x)
        y=y.reshape(10000,3)
        return y
#%%
def train_val_test_split(folder, train_percent, val_percent=0.1):
    not_converged=48
    id_test=[24, 150, 168, 171, 174, 192, 318]

    id_other=[]
    for n in range(0, 343):
        if (n != not_converged) and (n not in id_test):
            id_other.append(n)

    rng=np.random.RandomState(0)#seed=0
    rng.shuffle(id_other)

    id_val=id_other[0:int(342*val_percent)]
    id_train=id_other[int(342*val_percent):int(342*(train_percent+val_percent))]
    id_test.extend(id_other[int(342*(train_percent+val_percent)):])

    filelist_train_p0=[]
    filelist_train_p10=[]
    for idx in id_train:
        filelist_train_p0.append(folder+'matMean/'+'p0_'+str(idx)+'_solid_matMean_p20_i0.pt')
        filelist_train_p10.append(folder+'matMean/'+'p0_'+str(idx)+'_solid_matMean_p20_i10.pt')

    filelist_val_p0=[]
    filelist_val_p10=[]
    for idx in id_val:
        filelist_val_p0.append(folder+'matMean/'+'p0_'+str(idx)+'_solid_matMean_p20_i0.pt')
        filelist_val_p10.append(folder+'matMean/'+'p0_'+str(idx)+'_solid_matMean_p20_i10.pt')

    filelist_test_p0=[]
    filelist_test_p10=[]
    for idx in id_test:
        filelist_test_p0.append(folder+'matMean/'+'p0_'+str(idx)+'_solid_matMean_p20_i0.pt')
        filelist_test_p10.append(folder+'matMean/'+'p0_'+str(idx)+'_solid_matMean_p20_i10.pt')

    return (filelist_train_p0, filelist_train_p10,
            filelist_val_p0,   filelist_val_p10,
            filelist_test_p0,  filelist_test_p10)
#%%
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=1, type=int)
parser.add_argument('--dtype', default="float32", type=str)
parser.add_argument('--path', default='./app3/', type=str)
parser.add_argument('--shell_mesh', default='data/bav17_AortaModel_P0_best', type=str)
parser.add_argument('--mesh_tube', default='data/aorta_tube_solid_1layers', type=str)
parser.add_argument('--folder_data', default='data/343c1.5/', type=str)
parser.add_argument('--folder_result', default='result/PyFEA_NN_P0/', type=str)
parser.add_argument('--max_epochs', default=10000, type=int)
parser.add_argument('--lr_decay_interval', default=100, type=int)
parser.add_argument('--lr_decay', default=None, type=float)
parser.add_argument('--lr_init', default=1e-4, type=float)
parser.add_argument('--lr_min', default=1e-6, type=float)
parser.add_argument('--encoder_net', default='Encoder(128,8,1)', type=str)
parser.add_argument('--decoder_net', default='Decoder(128,8,1)', type=str)
parser.add_argument('--train_percent', default=0.8, type=float)
parser.add_argument('--u_aug', default='False', type=str)
arg = parser.parse_args()
#%%
if arg.lr_decay is None:
    arg.lr_decay=np.exp(np.log(arg.lr_min/arg.lr_init)/(arg.max_epochs//arg.lr_decay_interval-1))
#%%
arg.folder_data=arg.path+arg.folder_data
arg.mesh_tube=arg.path+arg.mesh_tube
arg.shell_mesh=arg.path+arg.shell_mesh
print(arg)
#%%
device=torch.device("cuda:"+str(arg.cuda))
#device=torch.device("cpu")
if arg.dtype == "float64":
    dtype=torch.float64
elif arg.dtype == "float32":
    dtype=torch.float32
else:
    raise ValueError("unkown dtype:"+arg.dtype)
#%%
filename_shell=arg.shell_mesh+".pt"
n_layers=1
if '4layer' in arg.mesh_tube:
    n_layers=4
(boundary0, boundary1, Element_surface_pressure, Element_surface_free)=get_solid_mesh_cfg(filename_shell, n_layers)
#%%
mesh_tube=PolyhedronMesh()
mesh_tube.load_from_torch(arg.mesh_tube+".pt")
NodeTube=mesh_tube.node.to(dtype).to(device)
Element=mesh_tube.element.to(device)
Element_surface_pressure=Element_surface_pressure.to(device)
mask=torch.ones_like(NodeTube)
mask[boundary0]=0
mask[boundary1]=0
#%%
def load_mesh(filename_p0, filename_px, dtype=dtype, device=device):
    with torch.no_grad():
        mesh_p0=PolyhedronMesh()
        mesh_p0.load_from_torch(filename_p0)
        X=mesh_p0.node.to(dtype).to(device)
        mesh_px=PolyhedronMesh()
        mesh_px.load_from_torch(filename_px)
        x=mesh_px.node.to(dtype).to(device)
    return X, x
#%%
def load_all(filelist_p0, filelist_px, dtype, device):
    X_all=[]; x_all=[]
    for filename_p0, filename_px in zip(filelist_p0, filelist_px):
        try:
            X, x=load_mesh(filename_p0, filename_px, dtype, device)
            X_all.append(X)
            x_all.append(x)
        except:
            print("cannot load", filename_px)
    return X_all, x_all
#%%
(filelist_train_p0, filelist_train_p10,
 filelist_val_p0,   filelist_val_p10,
 filelist_test_p0,  filelist_test_p10)=train_val_test_split(arg.folder_data, arg.train_percent)

X_train, x_train=load_all(filelist_train_p0, filelist_train_p10, dtype, device)
X_val, x_val=load_all(filelist_val_p0, filelist_val_p10, dtype, device)
X_test, x_test=load_all(filelist_test_p0, filelist_test_p10, dtype, device)
print("train", len(X_train), "val", len(X_val), "test", len(X_test))
#%%
data_p0=[]
for X in X_train:
    data_p0.append(X.view(1, -1).cpu())
data_p0=torch.cat(data_p0, dim=0)
MeanShape_p0=data_p0.mean(dim=0).view(10000,3).to(dtype).to(device)
del X, data_p0
#%%
data_px=[]
for x in x_train:
    data_px.append(x.view(1, -1).cpu())
data_px=torch.cat(data_px, dim=0)
MeanShape_px=data_px.mean(dim=0).view(10000,3).to(dtype).to(device)
del x, data_px
#%%
#X_train.append(MeanShape_p0)
#x_train.append(MeanShape_px)
#X_train.append(torch.zeros_like(MeanShape_p0))
#x_train.append(torch.zeros_like(MeanShape_px))
#%%
encoder_net=eval(arg.encoder_net).to(dtype).to(device)
decoder_net=eval(arg.decoder_net).to(dtype).to(device)
#%%
Code=encoder_net(0*MeanShape_p0)
u_field=decoder_net(Code)
#sys.exit()
#%%
def test(X_list, x_list):
    encoder_net.eval()
    decoder_net.eval()
    with torch.no_grad():
        mrse_mean=[]
        mrse_max=[]
        for X, x in zip(X_list, x_list):
            u_true=x-X
            c=encoder_net(u_true)
            u_pred=decoder_net(c)
            u_pred=u_pred*mask
            mrse=((u_pred-u_true)**2).sum(dim=1).sqrt()
            mrse_mean.append(mrse.mean().item())
            mrse_max.append(mrse.max().item())
        mrse_mean=np.mean(mrse_mean)
        mrse_max=np.max(mrse_max)
    return mrse_mean, mrse_max
#%%
from torch.optim import Adamax, Adam
lr=arg.lr_init
#%%
def update_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr
#%%
mrse_list_train=[]
mrse_list_val=[]
mrse_list_test=[]
#%%
optimizer=Adam(list(encoder_net.parameters())+list(decoder_net.parameters()), lr=lr)
#%%
idxlist=np.arange(0, len(X_train))
for epoch in range(0, arg.max_epochs):
    t0=time.time()
    encoder_net.train()
    decoder_net.train()
    np.random.shuffle(idxlist)
    for n in range(0, len(X_train)):
        X=X_train[idxlist[n]]
        x=x_train[idxlist[n]]
        u_true=x-X
        #----------------------------------
        if arg.u_aug == 'True':
            if np.random.rand() > 0.5:
                u_true*=np.random.rand()
        #----------------------------------
        c=encoder_net(u_true)
        u_pred=decoder_net(c)
        loss=((u_pred-u_true)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        if arg.u_aug == 'True':
            torch.nn.utils.clip_grad_norm_(encoder_net.parameters(), max_norm=1, norm_type=2)
            torch.nn.utils.clip_grad_norm_(decoder_net.parameters(), max_norm=1, norm_type=2)
        optimizer.step()
    #-------------------
    if (epoch+1)%arg.lr_decay_interval == 0:
        lr=lr*arg.lr_decay
        lr=max(lr, arg.lr_min)
        update_lr(optimizer, lr)
    #-------------------

    mrse_train=test(X_train, x_train)
    mrse_val=test(X_val, x_val)
    mrse_test=test(X_test, x_test)

    mrse_list_train.append(mrse_train)
    mrse_list_val.append(mrse_val)
    mrse_list_test.append(mrse_test)

    t1=time.time()
    print("epoch", epoch, "time", t1-t0)
    print("train: mrse", *mrse_train)
    print("val:   mrse", *mrse_val)
    print("test:  mrse", *mrse_test)
#%%
import os
folder_result=arg.path+arg.folder_result+arg.encoder_net+'_'+arg.decoder_net+'/'+str(arg.train_percent)+'/'
if arg.u_aug == 'True':
    folder_result+='u_aug/'
folder_result_test=folder_result+'/test/'
if os.path.exists(folder_result_test) == False:
    os.makedirs(folder_result_test)
filename_save=folder_result+arg.encoder_net+'_'+arg.decoder_net
#%%
if arg.max_epochs > 0:
    torch.save({"arg":arg,
                "encoder_model_state":encoder_net.state_dict(),
                "decoder_model_state":decoder_net.state_dict(),
                "mrse_train":mrse_list_train,
                "mrse_val":mrse_list_val,
                "mrse_test":mrse_list_test,
                "MeanShape_px":MeanShape_px,
                "MeanShape_p0":MeanShape_p0},
               filename_save+".pt")
    print("saved:", filename_save)
#%%

