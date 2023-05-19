# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 20:32:22 2023

@author: liang
"""

import pandas as pd
import torch
import numpy as np

folder="./app2/"
data=[[] for _ in range(7) ]
matRef=[120, 6000, 60, 1/3, 90, 1e5]
idx_list=[24,150,168,171,174,192,318]
for n in range(0, 7):
    idx=str(idx_list[n])
    result=torch.load(folder+'P0_'+idx+'_solid_matMean_p20_i50_ex_vivo_mat.pt')
    Mat_true=result['Mat_ture'].reshape(-1)
    Mat_pred=result['Mat_pred'].reshape(-1)
    error=np.abs(Mat_pred-Mat_true)/matRef
    for m in range(0,5):
        data[n].append(error[m])
    data[n].append(result['time'][-1])

data=pd.DataFrame(data, columns=['m0','m1','m2','m3','m4','t'], index=[24,150,168,171,174,192,318])
with pd.option_context('display.float_format', '{:0.10f}'.format):
    print(data)
data.to_csv('./app2/table/ex_vivo_mat_7shapes_result.csv')