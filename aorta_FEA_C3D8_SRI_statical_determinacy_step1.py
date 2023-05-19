# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:10:41 2022

@author: liang
"""
import torch
#%%
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--mat_id', default=0, type=int)
parser.add_argument('--shape_id', default=171, type=int)#['24', '150', '168', '171', '174', '192', '318']
arg = parser.parse_args()
[print(i) for i in vars(arg).items()];
#%%
#'''
mat_model='GOH_SRI'
matMean=torch.load('./app4/p0_true/125mat.pt')['mean_mat_str']
mat_name='matMean'
mat_true=matMean
mat_sd_c0="1e4"
mat_sd=mat_sd_c0+", 0, 1, 0, 0, 1e5"
print('mat_name', mat_name)
print('mat_sd', mat_sd)
#'''
#%%
cuda=str(arg.cuda)
idx=str(arg.shape_id)
filename_p0="./app4/p0_true/p0_"+idx+"_solid"
filename_p10="./app4/result/p0_"+idx+"_solid_"+mat_name+"_p10"
filename_p16="./app4/result/p0_"+idx+"_solid_"+mat_name+"_p16"
filename_p10sd_inverse_p0="./app4/result/p0_"+idx+"_solid_"+mat_name+"_p10_inverse_p0sd_"+mat_sd_c0
filename_p16sd_inverse_p0="./app4/result/p0_"+idx+"_solid_"+mat_name+"_p16_inverse_p0sd_"+mat_sd_c0
filename_p10sd="./app4/result/p0_"+idx+"_solid_"+mat_name+"_p10_sd_"+mat_sd_c0
filename_p16sd="./app4/result//p0_"+idx+"_solid_"+mat_name+"_p16_sd_"+mat_sd_c0
shell_template_str="./app4/p0_true/bav17_AortaModel_P0_best"
#%%
import os
#%%
if not os.path.isfile(filename_p10+'.pt'):
    cmd_p10=("python aorta_FEA_C3D8_SRI_quasi_newton_R1.py"
             +" --cuda "+cuda
             +" --pressure 10"
             +" --mat "+'"'+mat_true+'"'
             +" --mat_model "+mat_model
             +" --mesh_p0 "+filename_p0
             +" --mesh_px "+filename_p10
             +' --save_by_vtk True'
             +' --shell_template '+shell_template_str
             +' --n_layers 1'
             +' --loss1 0.00001'
             +' --Rmax  0.0001'
             +' --min_step_size 0.0000000001'
             +' --max_step_size 0.01'
             +' --save_all_stress True'
             )
    os.system(cmd_p10)
#'''
#%%
if not os.path.isfile(filename_p16+'.pt'):
    cmd_p16=("python aorta_FEA_C3D8_SRI_quasi_newton_R1.py"
             +" --cuda "+cuda
             +" --pressure 16"
             +" --mat "+'"'+mat_true+'"'
             +" --mat_model "+mat_model
             +" --mesh_p0 "+filename_p0
             +" --mesh_px "+filename_p16
             +' --save_by_vtk True'
             +' --shell_template '+shell_template_str
             +' --n_layers 1'
             +' --loss1 0.00001'
             +' --Rmax  0.0001'
             +' --min_step_size 0.0000000001'
             +' --max_step_size 0.01'
             +' --save_all_stress True'
             )
    os.system(cmd_p16)
#'''
#%% slow but it will converge
'''
cmd_p10_p0sd=("python aorta_FEA_C3D8_SRI_inverse_P0_R2b.py"
              +" --cuda "+cuda
              +" --pressure 10"
              +" --mat "+'"'+mat_sd+'"'
              +" --mat_model "+mat_model
              +" --mesh_px "+filename_p10
              +" --mesh_p0_pred "+filename_p10_inverse_p0sd
              +' --mesh_p0_init ""'
              +' --shell_template '+shell_template_str
              +' --loss1 0.00001'
              +' --Rmax  0.0001'
              +' --max_iter1 1'
              +' --max_iter2 10001'
              )
os.system(cmd_p10_p0sd)
'''
#%% faster but it may not converge if mat_sd_c0 is too large (e.g., 1e5) or too small (e.g., 100)
#'''
cmd_p10_p0sd=("python aorta_FEA_C3D8_SRI_quasi_newton_inverse_p0_R1.py"
             +" --cuda "+cuda
             +" --pressure 10"
             +" --mat "+'"'+mat_sd+'"'
             +" --mat_model "+mat_model
             +" --mesh_px "+filename_p10
             +" --mesh_p0 "+filename_p10sd_inverse_p0
             +' --mesh_p0_init ""'
             +' --shell_template '+shell_template_str
             +' --loss1 0.00001'
             +' --Rmax  0.0001'
             +' --max_step_size 0.1'
             +' --max_t_linesearch 0.5'
             )
os.system(cmd_p10_p0sd)
#'''
#%%
cmd_p16_p0sd=("python aorta_FEA_C3D8_SRI_quasi_newton_inverse_p0_R1.py"
             +" --cuda "+cuda
             +" --pressure 16"
             +" --mat "+'"'+mat_sd+'"'
             +" --mat_model "+mat_model
             +" --mesh_px "+filename_p16
             +" --mesh_p0 "+filename_p16sd_inverse_p0
             +' --mesh_p0_init ""'
             +' --shell_template '+shell_template_str
             +' --loss1 0.00001'
             +' --Rmax  0.0001'
             +' --max_step_size 0.1'
             +' --max_t_linesearch 0.5'
             )
os.system(cmd_p16_p0sd)
#%%
cmd_p10sd=("python aorta_FEA_C3D8_SRI_quasi_newton_R1.py"
           +" --cuda "+cuda
           +" --pressure 10"
           +" --mat "+'"'+mat_sd+'"'
           +" --mat_model "+mat_model
           +" --mesh_p0 "+filename_p10sd_inverse_p0
           +" --mesh_px "+filename_p10sd
           +' --save_by_vtk True'
           +' --shell_template '+shell_template_str
           +' --n_layers 1'
           +' --loss1 0.00001'
           +' --Rmax  0.0001'
           +' --max_step_size 0.1'
           +' --save_all_stress True'
           )
os.system(cmd_p10sd)
#%%
cmd_p16sd=("python aorta_FEA_C3D8_SRI_quasi_newton_R1.py"
           +" --cuda "+cuda
           +" --pressure 16"
           +" --mat "+'"'+mat_sd+'"'
           +" --mat_model "+mat_model
           +" --mesh_p0 "+filename_p16sd_inverse_p0
           +" --mesh_px "+filename_p16sd
           +' --save_by_vtk True'
           +' --shell_template '+shell_template_str
           +' --n_layers 1'
           +' --loss1 0.00001'
           +' --Rmax  0.0001'
           +' --max_step_size 0.1'
           +' --save_all_stress True'
           )
os.system(cmd_p16sd)
#%% check error
import sys
sys.path.append("mesh")
from PolyhedronMesh import PolyhedronMesh
mesh_pL=PolyhedronMesh()
mesh_pL.load_from_torch(filename_p10+".pt")
mesh_pH=PolyhedronMesh()
mesh_pH.load_from_torch(filename_p16+".pt")
S_pL_true=mesh_pL.mesh_data['S']
S_pH_true=mesh_pH.mesh_data['S']
#--------------------------------------------------
mesh_pLsd=PolyhedronMesh()
mesh_pLsd.load_from_torch(filename_p10sd+".pt")
mesh_pHsd=PolyhedronMesh()
mesh_pHsd.load_from_torch(filename_p16sd+".pt")
node_error_pL=((mesh_pLsd.node-mesh_pL.node)**2).sum(dim=1).sqrt()
node_error_pH=((mesh_pHsd.node-mesh_pH.node)**2).sum(dim=1).sqrt()
S_pL_sd=mesh_pLsd.mesh_data['S']
S_pH_sd=mesh_pHsd.mesh_data['S']
S_error_pL=(S_pL_true.mean(dim=1)-S_pL_sd.mean(dim=1)).abs()/S_pL_true.mean(dim=1).abs().mean()
S_error_pL=S_error_pL.mean(dim=(1,2))
S_error_pH=(S_pH_true.mean(dim=1)-S_pH_sd.mean(dim=1)).abs()/S_pH_true.mean(dim=1).abs().mean()
S_error_pH=S_error_pH.mean(dim=(1,2))
mesh_pLsd.element_data['S_error']=S_error_pL.view(-1,1)
mesh_pHsd.element_data['S_error']=S_error_pH.view(-1,1)
mesh_pLsd.node_data['node_error']=node_error_pL.view(-1,1)
mesh_pHsd.node_data['node_error']=node_error_pH.view(-1,1)
#--------------------------------------------------
mesh_pLsd.save_by_torch(filename_p10sd+".pt")
mesh_pLsd.save_by_vtk(filename_p10sd+".vtk")
mesh_pHsd.save_by_torch(filename_p16sd+".pt")
mesh_pHsd.save_by_vtk(filename_p16sd+".vtk")
#%%
