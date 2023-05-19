# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 18:04:25 2022

@author: liang
"""
#%%
#import os
#os.environ["MKL_NUM_THREADS"] = "?"
#%%
import sys
sys.path.append("c3d8")
sys.path.append("mesh")
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
import torch
from PolyhedronMesh import PolyhedronMesh
import time
import os
#%%
matA="200, 0, 1, 0.3333, 0, 1e5"
matB="50, 1000, 10, 0.3333, 0, 1e5"
matC="50, 1000, 10, 0.1, 60, 1e5"
mat_sd="1e5, 0, 1, 0.3333, 0, 1e5"
all_mat=torch.load('./app1/125mat.pt')['mat_str']
mat95=all_mat[95]
mat10=all_mat[10]
mat24=all_mat[24]
mat64=all_mat[64]
matMean=torch.load('./app1/125mat.pt')['mean_mat_str']
#%%
shell_template_str='./app3/data/bav17_AortaModel_P0_best'
mesh_px_str='./app3/data/343c1.5/matMean/p0_174_solid_matMean_p20_i10'
mesh_p0_pred_str='./app3/result/BD/p0_174_solid_matMean_p20_i10_BD'
mat_str=matMean
px_pressure=10
#%%
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=1, type=int)
parser.add_argument('--n_layers', default=1, type=int)
parser.add_argument('--mesh_px', default=mesh_px_str, type=str)
parser.add_argument('--mesh_p0_pred', default=mesh_p0_pred_str, type=str)
parser.add_argument('--shell_template', default=shell_template_str, type=str)
parser.add_argument('--mat', default=mat_str, type=str)
parser.add_argument('--pressure', default=px_pressure, type=float)
parser.add_argument('--max_iterA', default=0, type=int)
parser.add_argument('--max_iterB', default=20, type=int)
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--loss1', default=0.1, type=float)
parser.add_argument('--Rmax', default=0.005, type=float)
parser.add_argument('--max_step_size', default=0.01, type=float)
parser.add_argument('--save_by_vtk', default='True', type=str)
parser.add_argument('--save_all_iterations', default='True', type=str)
arg = parser.parse_args()
print(arg)
#%%
mesh_px=PolyhedronMesh()
mesh_px.load_from_torch(arg.mesh_px+'.pt')
Node_x=mesh_px.node
#%%
name='BD'+str(arg.max_iterA)+'A'+str(arg.max_iterB)+'B'+str(arg.alpha)+'a'
#%%
@torch.no_grad()
def run_one_iteration(pressure_t, file_mesh_p0_t, file_mesh_px_t, file_mesh_p0_t_next,
                      max_step_size, alpha, final_iteration):
    #-----------------------------------------------
    cmd=("python aorta_FEA_C3D8_SRI_quasi_newton_R1.py"
         +" --cuda "+str(arg.cuda)
         +" --pressure "+str(pressure_t)
         +' --mat ' + '"' + arg.mat +'"'
         +" --mesh_p0 "+file_mesh_p0_t
         +" --mesh_px "+file_mesh_px_t
         +' --save_by_vtk '+arg.save_by_vtk
         +' --shell_template '+arg.shell_template
         +' --n_layers '+str(arg.n_layers)
         +' --plot False'
         +' --loss1 '+str(arg.loss1)
         +' --Rmax '+str(arg.Rmax)
         +' --max_step_size '+str(max_step_size)
         )
    os.system(cmd)
    if final_iteration == True:
        return
    #-----------------------------------------------
    mesh_p0_t=PolyhedronMesh()
    mesh_p0_t.load_from_torch(file_mesh_p0_t+'.pt')
    Node_p0_t=mesh_p0_t.node
    mesh_px_t=PolyhedronMesh()
    mesh_px_t.load_from_torch(file_mesh_px_t+'.pt')
    Node_px_t=mesh_px_t.node
    #-----------------------------------------------
    Node_p0_t_next=Node_p0_t+alpha*(Node_x-Node_px_t)
    #-----------------------------------------------
    mesh_p0_t_next=PolyhedronMesh()
    mesh_p0_t_next.element=mesh_px.element
    mesh_p0_t_next.node=Node_p0_t_next
    mesh_p0_t_next.save_by_torch(file_mesh_p0_t_next+'.pt')
    if arg.save_by_vtk == 'True':
        mesh_p0_t_next.save_by_vtk(file_mesh_p0_t_next+'.vtk')
#%% increase pressure gradually
for t in range(0, arg.max_iterA):
    if t == 0:
        file_mesh_p0_t=arg.mesh_px
    else:
        file_mesh_p0_t=arg.mesh_p0_pred+'_'+name+'_p0_t'+str(t)
    file_mesh_px_t=arg.mesh_p0_pred+'_'+name+'_px_t'+str(t)
    file_mesh_p0_t_next=arg.mesh_p0_pred+'_'+name+'_p0_t'+str(t+1)
    #-----------------------------------------------
    if t < arg.max_iterA:
        pressure_t=arg.pressure*(t+1)/arg.max_iterA
    else:
        pressure_t=arg.pressure
    if arg.max_step_size > 0.1 or arg.max_iterA==1:
        max_step_size=arg.max_step_size
    else:
        #decrease max_step_size from 0.1 to arg.max_step_size
        max_step_size=0.1+(arg.max_step_size-0.1)*t/(arg.max_iterA-1)
    final_iteration=False
    if t==arg.max_iterA-1 and arg.max_iterB==0:
        final_iteration=True
    run_one_iteration(pressure_t, file_mesh_p0_t, file_mesh_px_t, file_mesh_p0_t_next,
                      max_step_size=max_step_size, alpha=1, final_iteration=final_iteration)
#%% refine
t_refine_start=arg.max_iterA
t_refine_end=arg.max_iterA+arg.max_iterB
#%%
for t in range(t_refine_start, t_refine_end):
    if t==t_refine_start and t_refine_start == 0:
        file_mesh_p0_t=arg.mesh_px
    else:
        file_mesh_p0_t=arg.mesh_p0_pred+'_'+name+'_p0_t'+str(t)
    file_mesh_px_t=arg.mesh_p0_pred+'_'+name+'_px_t'+str(t)
    file_mesh_p0_t_next=arg.mesh_p0_pred+'_'+name+'_p0_t'+str(t+1)
    pressure_t=arg.pressure
    final_iteration=False
    if t==t_refine_end-1:
        final_iteration=True
    run_one_iteration(pressure_t, file_mesh_p0_t, file_mesh_px_t, file_mesh_p0_t_next,
                      max_step_size=arg.max_step_size, alpha=arg.alpha, final_iteration=final_iteration)
#%%
if arg.save_all_iterations == 'False':
    try:
        os.remove(arg.mesh_p0_pred+'_'+name+'_px_t0.pt')
        os.remove(arg.mesh_p0_pred+'_'+name+'_px_t0.vtk')
    except:
        pass
    for t in range(1, arg.max_iterA+arg.max_iterB-1):
        file_mesh_p0_t=arg.mesh_p0_pred+'_'+name+'_p0_t'+str(t)
        file_mesh_px_t=arg.mesh_p0_pred+'_'+name+'_px_t'+str(t)
        try:
            os.remove(file_mesh_px_t+'.pt')
            os.remove(file_mesh_p0_t+'.pt')
            os.remove(file_mesh_px_t+'.vtk')
            os.remove(file_mesh_p0_t+'.vtk')
        except:
            pass
