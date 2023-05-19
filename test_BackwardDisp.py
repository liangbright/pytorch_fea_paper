# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 00:23:11 2022

@author: liang
"""

import os
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=0, type=int)
arg = parser.parse_args()
print(arg)
#%%
folder_data='./app3/data/343c1.5/matMean/'
folder_result='/app3/result/BD/'
shell_template_str='./app3/data/bav17_AortaModel_P0_best'
#%%
for idx in [24, 150, 168, 171, 174, 192, 318]:
    mesh_px_str=folder_data+'p0_'+str(idx)+'_solid_matMean_p20_i10'
    mesh_p0_pred_str=folder_result+'p0_'+str(idx)+'_solid_matMean_p20_i10_BD'
    cmd=("python aorta_FEA_C3D8_SRI_quasi_newton_BackwardDisp.py"
         +" --cuda "+str(arg.cuda)
         +" --pressure 10"
         +" --mesh_px "+mesh_px_str
         +" --mesh_p0_pred "+mesh_p0_pred_str
         +' --shell_template '+shell_template_str
         +' --loss1 0.1'
         +' --Rmax 0.005'
         +' --max_iterA 0'
         +' --max_iterB 20'
         +' --alpha 0.5'
         )
    os.system(cmd)
