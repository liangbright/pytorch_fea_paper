# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 10:18:57 2023

@author: liang
"""
#%%
import torch
import numpy as np
matMean=torch.load('./app4/p0_true/125mat.pt')['mean_mat'][0:5]
matRef=[120, 6000, 60, 1/3, 90]
path='./app4/result/'
id_list=['24', '150', '168', '171', '174', '192', '318']
mat_listA1=[]
error_listA1=[]
mat_listA8=[]
error_listA8=[]
mat_listB=[]
error_listB=[]
for k in range(0, 7):
    idx=id_list[k]
    dataA1=torch.load(path+'p0_'+idx+'_solid_matMean_p10_sd_1e4_invivo_mat_sd_b0.0a1.0n1.pt', map_location='cpu')
    dataA8=torch.load(path+'p0_'+idx+'_solid_matMean_p10_sd_1e4_invivo_mat_sd_b0.0a1.0n8.pt', map_location='cpu')
    dataB=torch.load(path+'p0_'+idx+'_solid_matMean_p10_sd_1e4_invivo_mat_sd_b1.0a1.0n8.pt', map_location='cpu')

    data_p10=torch.load(path+'p0_'+idx+'_solid_matMean_p10.pt', map_location='cpu')
    data_p16=torch.load(path+'p0_'+idx+'_solid_matMean_p16.pt', map_location='cpu')
    data_p10sd=torch.load(path+'p0_'+idx+'_solid_matMean_p10_sd_1e4.pt', map_location='cpu')
    data_p16sd=torch.load(path+'p0_'+idx+'_solid_matMean_p16_sd_1e4.pt', map_location='cpu')

    S_p10=data_p10['element_data']['S']
    S_p16=data_p16['element_data']['S']
    S_p10sd=data_p10sd['element_data']['S']
    S_p16sd=data_p16sd['element_data']['S']

    S_p10_error=float(((S_p10sd-S_p10)**2).sum(dim=1).sqrt().mean()/(S_p10**2).sum(dim=1).sqrt().max())
    S_p16_error=float(((S_p16sd-S_p16)**2).sum(dim=1).sqrt().mean()/(S_p16**2).sum(dim=1).sqrt().max())

    matA1=dataA1['mesh_data']['Mat'][-1].tolist()[0][0:5]
    matA1[4]*=180/np.pi
    matA8=dataA8['mesh_data']['Mat'][-1].tolist()[0][0:5]
    matA8[4]*=180/np.pi
    matB=dataB['mesh_data']['Mat'][-1].tolist()[0][0:5]
    matB[4]*=180/np.pi
    mat_listA1.append(matA1)
    errorA1=np.abs(np.array(matA1)-np.array(matMean))/np.array(matRef)
    error_listA1.append(errorA1.tolist()+[S_p10_error, S_p16_error])
    mat_listA8.append(matA8)
    errorA8=np.abs(np.array(matA8)-np.array(matMean))/np.array(matRef)
    error_listA8.append(errorA8.tolist()+[S_p10_error, S_p16_error])
    mat_listB.append(matB)
    errorB=np.abs(np.array(matB)-np.array(matMean))/np.array(matRef)
    error_listB.append(errorB.tolist()+[S_p10_error, S_p16_error])

print("matMean", matMean.tolist())
#%%
import pandas as pd
df_matA1=pd.DataFrame(mat_listA1, columns=['m0','m1','m2','m3','m4'],
                     index=[24,150,168,171,174,192,318])
df_matA1.to_csv('./app4/table/mat_listA1.csv')
print(df_matA1)

dfA1=pd.DataFrame(error_listA1, columns=['m0','m1','m2','m3','m4', 'error_p10', 'error_p16'],
                index=[24,150,168,171,174,192,318])
dfA1.loc[len(dfA1.index)]=dfA1.mean(axis=0).values
dfA1.to_csv('./app4/table/in_vivo_mat_7shapes_resultA1.csv')
print(dfA1) #without residual force terms, using the average from the 8 integration points, Table 4

df_matA8=pd.DataFrame(mat_listA8, columns=['m0','m1','m2','m3','m4'],
                     index=[24,150,168,171,174,192,318])
df_matA8.to_csv('./app4/table/mat_listA8.csv')
print(df_matA8)

dfA8=pd.DataFrame(error_listA8, columns=['m0','m1','m2','m3','m4', 'error_p10', 'error_p16'],
                index=[24,150,168,171,174,192,318])
dfA8.loc[len(dfA8.index)]=dfA8.mean(axis=0).values
dfA8.to_csv('./app4/table/in_vivo_mat_7shapes_resultA8.csv')
print(dfA8) #without residual force terms, using 8 integration points, not shown in the paper

df_matB=pd.DataFrame(mat_listB, columns=['m0','m1','m2','m3','m4'],
                     index=[24,150,168,171,174,192,318])
df_matB.to_csv('./app4/table/mat_listB.csv')
print(df_matB)

dfB=pd.DataFrame(error_listB, columns=['m0','m1','m2','m3','m4', 'error_p10', 'error_p16'],
                index=[24,150,168,171,174,192,318])
dfB.loc[len(dfB.index)]=dfB.mean(axis=0).values
dfB.to_csv('./app4/table/in_vivo_mat_7shapes_resultB.csv')
print(dfB) #with residual force terms, Table 4
