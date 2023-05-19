# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 23:52:43 2022

@author: liang
"""

import os
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--idx', default=171, type=int)# [24, 150, 168, 171, 174, 192, 318]
arg = parser.parse_args()
#%%
folder_data='./app3/data/343c1.5_125mat/matMean/'
folder_result='./app3/result/PyFEA_P0/'
shell_template_str='./app3/data/bav17_AortaModel_P0_best'
#%% [24, 150, 168, 171, 174, 192, 318]
for idx in [arg.idx]:
    mesh_px_str=folder_data+'p0_'+str(idx)+'_solid_matMean_p20_i10'
    mesh_p0_pred_str=folder_result+'p0_'+str(idx)+'_solid_matMean_p20_i10_inverse_p0'
    cmd=("python aorta_FEA_C3D8_SRI_inverse_P0_R2b.py"
         +" --cuda "+str(arg.cuda)
         +" --pressure 10"
         +" --mesh_px "+mesh_px_str
         +" --mesh_p0_pred "+mesh_p0_pred_str
         +' --mesh_p0_init ""'
         +' --shell_template '+shell_template_str
         +' --loss1 0.1'
         +' --Rmax 0.005'
         +' --max_iter1 100'
         +' --max_iter2 10001'
         )
    os.system(cmd)
#%% refine
for idx in [arg.idx]:
    mesh_px_str=folder_data+'p0_'+str(idx)+'_solid_matMean_p20_i10'
    mesh_p0_init_str=folder_result+'p0_'+str(idx)+'_solid_matMean_p20_i10_inverse_p0'
    mesh_p0_pred_str=folder_result+'p0_'+str(idx)+'_solid_matMean_p20_i10_inverse_p0_refine'
    cmd=("python aorta_FEA_C3D8_SRI_inverse_P0_R2b.py"
         +" --cuda "+str(arg.cuda)
         +" --pressure 10"
         +" --mesh_px "+mesh_px_str
         +" --mesh_p0_pred "+mesh_p0_pred_str
         +" --mesh_p0_init "+mesh_p0_init_str
         +' --shell_template '+shell_template_str
         +' --loss1 0.001'
         +' --Rmax 0.005'
         +' --max_iter1 1'
         +' --max_iter2 10001'
         )
    os.system(cmd)