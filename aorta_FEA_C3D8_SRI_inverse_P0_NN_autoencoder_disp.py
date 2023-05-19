# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 19:39:51 2021

@author: liang
"""

import sys
sys.path.append("c3d8")
sys.path.append("mesh")
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
import torch
from AortaFEModel_C3D8_SRI import AortaFEModel, cal_potential_energy
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
matMean=torch.load('./app3/125mat.pt')['mean_mat_str']
#%%
encoder='Encoder(128,16,1)'
decoder='Decoder(128,16,1)'
folder_net="./app3/result/PyFEA_NN_P0/"+encoder+'_'+decoder+'/0.8/'
#['24', '150', '168', '171', '174', '192', '318']
idx='318'
mesh_px='./app3/data/343c1.5/matMean/p0_'+idx+'_solid_matMean_p20_i10'
mesh_p0_true='./app3/data/343c1.5_fast/matMean/p0_'+idx+'_solid_matMean_p20_i0'
mesh_p0_pred=folder_net+'p0_'+idx+'_solid_matMean_p20_i10_inverse_p0_NN'
pressure=10
#%%
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--dtype', default="float64", type=str)
parser.add_argument('--n_layers', default=1, type=int)
parser.add_argument('--shell_template', default='./app3/data/bav17_AortaModel_P0_best', type=str)
parser.add_argument('--mesh_px', default=mesh_px, type=str)
parser.add_argument('--mesh_p0_pred', default=mesh_p0_pred, type=str)
parser.add_argument('--mesh_p0_true', default=mesh_p0_true, type=str)#DEBUG
parser.add_argument('--mat', default=matMean, type=str)
parser.add_argument('--pressure', default=pressure, type=float)
parser.add_argument('--max_iter', default=100, type=int)
parser.add_argument('--folder_net', default=folder_net, type=str)
parser.add_argument('--encoder', default=encoder, type=str)
parser.add_argument('--decoder', default=decoder, type=str)
arg = parser.parse_args()
print(arg)
#%%
if arg.cuda < 0:
    device=torch.device("cpu")
else:
    device=torch.device("cuda:"+str(arg.cuda))
if arg.dtype == "float64":
    dtype=torch.float64
elif arg.dtype == "float32":
    dtype=torch.float32
else:
    raise ValueError("unkown dtype:"+arg.dtype)
#%% load net
filename_state=arg.folder_net+arg.encoder+'_'+arg.decoder+".pt"
state=torch.load(filename_state, map_location='cpu')
encoder=eval(arg.encoder)
encoder.load_state_dict(state['encoder_model_state'])
encoder=encoder.to(dtype).to(device)
#-----------------------------------
decoder=eval(arg.decoder)
decoder.load_state_dict(state['decoder_model_state'])
decoder=decoder.to(dtype).to(device)
#-----------------------------------
MeanShape_p0=state['MeanShape_p0'].to(dtype).to(device)
MeanShape_px=state['MeanShape_px'].to(dtype).to(device)
#%%
filename_shell=arg.shell_template+".pt"
(boundary0, boundary1, Element_surface_pressure, Element_surface_free)=get_solid_mesh_cfg(filename_shell, arg.n_layers)
#%%
Mesh_x=PolyhedronMesh()
Mesh_x.load_from_torch(arg.mesh_px+".pt")
Node_x=Mesh_x.node.to(dtype).to(device)
Element=Mesh_x.element.to(device)
Element_surface_pressure=Element_surface_pressure.to(device)
mask=torch.ones_like(Node_x)
mask[boundary0]=0
mask[boundary1]=0
#%% DEBUG
Mesh_X_true=PolyhedronMesh()
Mesh_X_true.load_from_torch(arg.mesh_p0_true+".pt")
Node_X_true=Mesh_X_true.node.to(dtype).to(device)
#%%
from Mat_GOH_SRI import cal_1pk_stress, cal_cauchy_stress
Mat=[float(m) for m in arg.mat.split(",")]
Mat[4]=np.pi*(Mat[4]/180)
Mat=torch.tensor([Mat], dtype=dtype, device=device)
aorta_model=AortaFEModel(Node_x, Element, None, boundary0, boundary1, Element_surface_pressure,
                         Mat, cal_1pk_stress, cal_cauchy_stress, dtype, device, mode='inverse_p0')
aorta_model.element_orientation_on_deformed_mesh=False
pressure=arg.pressure
#%%
def loss1a_function(force_int, force_ext):
    loss=((force_int-force_ext)**2).sum(dim=1).sqrt().mean()
    return loss
#%%
def loss1b_function(force_int, force_ext):
    loss=((force_int-force_ext)**2).sum()
    return loss
#%%
loss_list=[]
TPE_list=[]
Rmax_list=[]
time_list=[]
t0=time.time()
#%%
#Code=encoder(0*Node_x)
#u_field=decoder(Code)
#%%
from FE_lbfgs_ori import LBFGS
Code=encoder(MeanShape_px-MeanShape_p0)
Code=Code.clone().detach();
Code.requires_grad=True
optimizer = LBFGS([Code], lr=0.01,
                  line_search_fn="backtracking", #try strong_wolfe to see if it can get a better result (smaller loss)
                  t_max=0.5, tolerance_grad=1e-3, tolerance_change=1e-20, history_size =100, max_iter=1)
optimizer.set_backtracking(t_list=[0.5, 0.1, 0.05, 0.01, 0.001])
#%%
for iter1 in range(0, arg.max_iter):
    def closure(loss1_fn=loss1b_function, return_all=False):
        u_field=decoder(Code)
        u_field=u_field*mask
        Node_X=Node_x-u_field
        aorta_model.set_u_field(u_field.detach(), requires_grad=False)
        out =aorta_model.cal_energy_and_force(pressure)
        TPE=out['TPE1']
        force_int=out['force_int']
        force_ext=out['force_ext']
        loss1=loss1_fn(force_int, force_ext)
        force_res=force_int-force_ext
        loss2=cal_potential_energy(force_res, u_field)
        #loss=loss1
        loss=loss2
        if loss.requires_grad==True:
            optimizer.zero_grad()
            loss.backward()
        if return_all==False:
            return loss
            #return TPE #for line search
        else:
            force_int=force_int.detach()
            force_ext=force_ext.detach()
            force_int_of_element=out['force_int_of_element'].detach()
            Node_X=Node_X.detach()
            return float(loss), float(TPE), force_int, force_ext, force_int_of_element, Node_X
    opt_cond=optimizer.step(closure)

    loss, TPE, force_int, force_ext, force_int_of_element, Node_X=closure(loss1_fn=loss1a_function, return_all=True)

    force_avg=(force_int_of_element**2).sum(dim=2).sqrt().mean()
    force_res=((force_int-force_ext)**2).sum(dim=1).sqrt()
    R=force_res/(force_avg+1e-10)
    Rmean=R.mean().item()
    Rmax=R.max().item()
    Rmax_list.append(Rmax)

    loss_list.append(loss)
    TPE_list.append(TPE)
    t1=time.time()
    time_list.append(t1-t0)

    error=((Node_X-Node_X_true)**2).sum(dim=1).sqrt()
    node_error_mean=error.mean().item()
    node_error_max=error.max().item()

    if (iter1+1)%100 == 0 or opt_cond==True:
        print(iter1, t1-t0, loss, TPE, Rmean, Rmax, node_error_mean, node_error_max)
        display.clear_output(wait=False)
        fig, ax = plt.subplots()
        ax.plot(np.array(loss_list)/max(loss_list), 'r')
        ax.set_ylim(0, 1)
        ax.grid(True)
        display.display(fig)
        plt.close(fig)

    if opt_cond == True:
        print(iter1, t1-t0, loss, TPE, Rmean, Rmax, node_error_mean, node_error_max)
        print("break: opt_cond is True, time", t1-t0)
        break
#%%
#'''
mesh_p0_pred=PolyhedronMesh()
mesh_p0_pred.element=Element
mesh_p0_pred.node=Node_X
filename_save=arg.mesh_p0_pred
mesh_p0_pred.mesh_data['time']=time_list
mesh_p0_pred.mesh_data['loss']=loss_list
mesh_p0_pred.mesh_data['TPE']=TPE_list
mesh_p0_pred.mesh_data['Rmax']=Rmax_list
mesh_p0_pred.save_by_torch(filename_save+".pt")
mesh_p0_pred.save_by_vtk(filename_save+".vtk")
print("saved:", filename_save)
#'''