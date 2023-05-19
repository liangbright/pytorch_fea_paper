# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 05:06:42 2021

@author: liang
"""
import os
import torch
import numpy as np
#%%
all_mat=torch.load('./app1/125mat.pt')['mat_str']
#%%
def get_must_points(delta):
    #pressure=20, delta=0.01: 100 must points
    t=0
    n=0
    t_str="0"
    while t<1:
        n=n+1
        t=delta*n
        ts='{:.2f}'.format(t)
        t_str=t_str+","+ts
    return t_str
#%%
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default='1', type=str)
parser.add_argument('--delta', default=0.01, type=float)
parser.add_argument('--pressure', default=20, type=int)
parser.add_argument('--save_by_vtk', default='False', type=str)
arg = parser.parse_args()
print(arg)
#%%
t_str=get_must_points(arg.delta)
print('t_str',t_str)

n_layers=1

seed='0'

matMean=torch.load('./app1/125mat.pt')['mean_mat_str']

mat_model='GOH_SRI'
#mat_model='GOH_Jv'
#mat_model='GOH_3Field'
#mat_model='GOH_Fbar'

mat_id='Mean'
mat=matMean

for k in [24,150,168,171,174,192,318]:
    mesh_p0="./app1/p0_"+str(k)+"_solid"
    mesh_px="./app1/pyfea/p0_"+str(k)+"_solid_mat"+mat_id+"_p"+str(arg.pressure)
    if mat_model != 'GOH_SRI':
        mesh_px+='_'+mat_model
    if os.path.exists(mesh_px+'.pt') == True or k ==48:
        print('file exits, skip')
        #continue

    shell_template="./app1/bav17_AortaModel_P0_best"
    cmd=("python aorta_FEA_C3D8_SRI_quasi_newton_R1.py"
         +" --cuda "+arg.cuda
         +" --pressure "+str(arg.pressure)
         +' --mat ' + '"' + mat +'"'
         +' --mat_model ' + '"' + mat_model +'"'
         +" --mesh_p0 "+mesh_p0
         +" --mesh_px "+mesh_px
         +' --must_points '+'"'+t_str+'"'
         +' --random_seed '+seed
         +' --save_by_vtk '+arg.save_by_vtk
         +' --shell_template '+shell_template
         +' --n_layers '+str(n_layers)
         +' --plot False'
         +' --loss1 0.01'
         +' --Rmax 0.005'
         )
    print(cmd)
    os.system(cmd)
    #break