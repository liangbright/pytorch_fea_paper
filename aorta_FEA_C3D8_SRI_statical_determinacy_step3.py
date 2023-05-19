# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:11:14 2023

@author: liang
"""
import os
import argparse
#%%
shell_template_str='./app4/p0_true/bav17_AortaModel_P0_best'
pessure_Low=10
pessure_High=16
mat_model='GOH_SRI'
#%%
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--beta', default=1, type=float)#1: use force_loss, 0: not use force_loss
parser.add_argument('--alpha', default=1, type=float)#1: use stress_loss, 0: not use stress_loss
parser.add_argument('--nFa', default=1, type=int) # the number of integration points
parser.add_argument('--use_Fb', default='False', type=str)
parser.add_argument('--use_w_loss', default='False', type=str)
parser.add_argument('--g_w_loss', default=0, type=float)
parser.add_argument('--g_divisor', default=3, type=int)
parser.add_argument('--g_threshold', default=5, type=float)
parser.add_argument('--use_S_true', default='False', type=str)#for DEBUG
parser.add_argument('--max_iter', default=100000, type=int)#100000 is a good number
parser.add_argument('--sd_c0', default='1e4', type=str)
parser.add_argument('--mat_init', default='none', type=str)
parser.add_argument('--mat_name', default='matMean', type=str)
parser.add_argument('--shape_id', default=171, type=int)#['24', '150', '168', '171', '174', '192', '318']
arg = parser.parse_args()
#%%
mat_init=arg.mat_init
if len(arg.mat_name) == 0:
    raise ValueError('mat_name is empty')
mat_name=arg.mat_name
#--------------------------------
idx=str(arg.shape_id)
filename_p0="./app4/p0_true/p0_"+idx+"_solid"
filename_pL="./app4/result/p0_"+idx+"_solid_"+mat_name+"_p10"
filename_pH="./app4/result/p0_"+idx+"_solid_"+mat_name+"_p16"
filename_pLsd="./app4/result/p0_"+idx+"_solid_"+mat_name+"_p10_sd"
filename_pHsd="./app4/result/p0_"+idx+"_solid_"+mat_name+"_p16_sd"
if len(arg.sd_c0) > 0:
    filename_pLsd+='_'+arg.sd_c0
    filename_pHsd+='_'+arg.sd_c0
#--------------------------------
cmd=("python aorta_FEA_C3D8_SRI_inverse_mat_in_vivo_statical_determinacy.py"
     +' --cuda '+str(arg.cuda)
     +' --shell_template '+shell_template_str
     +' --mesh_pL '+filename_pL
     +' --mesh_pH '+filename_pH
     +' --mesh_pLsd '+filename_pLsd
     +' --mesh_pHsd '+filename_pHsd
     +' --pessureL '+str(pessure_Low)
     +' --pessureH '+str(pessure_High)
     +' --mat_model '+mat_model
     +' --nFa '+str(arg.nFa)
     +' --use_Fb '+str(arg.use_Fb)
     +' --use_w_loss '+str(arg.use_w_loss)
     +' --g_w_loss '+str(arg.g_w_loss)
     +' --g_divisor '+str(arg.g_divisor)
     +' --g_threshold '+str(arg.g_threshold)
     +' --use_S_true '+str(arg.use_S_true)
     +' --max_iter '+str(arg.max_iter)
     +' --mat_init '+mat_init
     +' --alpha '+str(arg.alpha)
     +' --beta '+str(arg.beta)
     )
os.system(cmd)