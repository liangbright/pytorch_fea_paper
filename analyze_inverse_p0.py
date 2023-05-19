# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 23:46:48 2022

@author: liang
"""
import torch
#%%
node_error_p0_mean_listB=[]
node_error_p0_max_listB=[]
MAPE_p10_listB=[]
APE_p10_listB=[]
time_cost_listB=[]

node_error_p0_mean_listC=[]
node_error_p0_max_listC=[]
MAPE_p10_listC=[]
APE_p10_listC=[]
time_cost_listC=[]

node_error_p0_mean_listD=[]
node_error_p0_max_listD=[]
MAPE_p10_listD=[]
APE_p10_listD=[]
time_cost_listD=[]

id_list=['24', '150', '168', '171', '174', '192', '318']
#%% [24, 150, 168, 171, 174, 192, 318]
for id in id_list:
    #%%
    mesh_p0=torch.load('./app3/data/343c1.5/p0_'+id+'_solid.pt', map_location='cpu')
    mesh_p10=torch.load('./app3/data/343c1.5/matMean/p0_'+id+'_solid_matMean_p20_i10.pt', map_location='cpu')
    disp_max=((mesh_p10['node']-mesh_p0['node'])**2).sum(dim=1).sqrt().max().item()
    #%%
    mesh_p0B=torch.load('./app3/result/PyFEA_P0/p0_'+id+'_solid_matMean_p20_i10_inverse_p0.pt', map_location='cpu')
    time_cost_listB.append(mesh_p0B['mesh_data']['time'])
    try: #refine may not be necessary, so the refine file may not exist
        mesh_p0B=torch.load('./app3/result/PyFEA_P0/p0_'+id+'_solid_matMean_p20_i10_inverse_p0_refine.pt', map_location='cpu')
    except:
        pass
    errorB=((mesh_p0['node']-mesh_p0B['node'])**2).sum(dim=1).sqrt()
    print('p0 errorB', errorB.max().item()/disp_max, errorB.mean().item()/disp_max)
    node_error_p0_mean_listB.append(errorB.mean().item()/disp_max)
    node_error_p0_max_listB.append(errorB.max().item()/disp_max)
    #%%
    mesh_p0C_px=torch.load('./app3/result/BD/p0_'+id+'_solid_matMean_p20_i10_BD_BD0A20B0.5a_px_t1.pt', map_location='cpu')
    time_cost_listC.append(mesh_p0C_px['mesh_data']['time'][-1]*20)
    mesh_p0C=torch.load('./app3/result/BD/p0_'+id+'_solid_matMean_p20_i10_BD_BD0A20B0.5a_p0_t20.pt', map_location='cpu')
    errorC=((mesh_p0['node']-mesh_p0C['node'])**2).sum(dim=1).sqrt()
    print('p0 errorC', errorC.max().item()/disp_max, errorC.mean().item()/disp_max)
    node_error_p0_mean_listC.append(errorC.mean().item()/disp_max)
    node_error_p0_max_listC.append(errorC.max().item()/disp_max)
    #%%
    netD='Encoder(128,16,1)_Decoder(128,16,1)'
    mesh_p0D=torch.load('./app3/result/PyFEA_NN_P0/'+netD+'/0.8/p0_'+id+'_solid_matMean_p20_i10_inverse_p0_NN.pt', map_location='cpu')
    time_cost_listD.append(mesh_p0D['mesh_data']['time'][-1])
    errorD=((mesh_p0['node']-mesh_p0D['node'])**2).sum(dim=1).sqrt()
    print('p0 errorD', errorD.max().item()/disp_max, errorD.mean().item()/disp_max)
    node_error_p0_mean_listD.append(errorD.mean().item()/disp_max)
    node_error_p0_max_listD.append(errorD.max().item()/disp_max)
    #%%
    mesh_p10=torch.load('./app3/data/343c1.5/matMean/p0_'+id+'_solid_matMean_p20_i10.pt', map_location='cpu')
    VM_max_p10=mesh_p10['element_data']['VM'].max().item()
    #%%
    try: #refine may not be necessary, so the refine file may not exist
        mesh_p10B=torch.load('./app3/result/PyFEA_P0/p0_'+id+'_solid_matMean_p20_i10_inverse_p0_refine_to_p10.pt', map_location='cpu')
    except:
        mesh_p10B=torch.load('./app3/result/PyFEA_P0/p0_'+id+'_solid_matMean_p20_i10_inverse_p0_to_p10.pt', map_location='cpu')
    errorB=(mesh_p10['element_data']['VM']-mesh_p10B['element_data']['VM']).abs().view(-1)
    print('VM errorB%', errorB.max().item()/VM_max_p10, errorB.mean().item()/VM_max_p10)
    VM_max_p10B=mesh_p10B['element_data']['VM'].max().item()
    print('VM_max errorB%', abs(VM_max_p10-VM_max_p10B)/VM_max_p10)
    MAPE_p10_listB.append(errorB.mean().item()/VM_max_p10B)
    APE_p10_listB.append(abs(VM_max_p10-VM_max_p10B)/VM_max_p10)
    #%%
    mesh_p10C=torch.load('./app3/result/BD/p0_'+id+'_solid_matMean_p20_i10_BD_BD0A20B0.5a_p0_t20_to_p10.pt', map_location='cpu')
    errorC=(mesh_p10['element_data']['VM']-mesh_p10C['element_data']['VM']).abs().view(-1)
    print('VM errorC%', errorC.max().item()/VM_max_p10, errorC.mean().item()/VM_max_p10)
    VM_max_p10C=mesh_p10C['element_data']['VM'].max().item()
    print('VM_max errorC%', abs(VM_max_p10-VM_max_p10C)/VM_max_p10)
    MAPE_p10_listC.append(errorC.mean().item()/VM_max_p10)
    APE_p10_listC.append(abs(VM_max_p10-VM_max_p10C)/VM_max_p10)
    #%%
    mesh_p10D=torch.load('./app3/result/PyFEA_NN_P0/'+netD+'/0.8/p0_'+id+'_solid_matMean_p20_i10_inverse_p0_NN_to_p10.pt', map_location='cpu')
    errorD=(mesh_p10['element_data']['VM']-mesh_p10D['element_data']['VM']).abs().view(-1)
    print('VM errorD%', errorD.max().item()/VM_max_p10, errorD.mean().item()/VM_max_p10)
    VM_max_p10D=mesh_p10D['element_data']['VM'].max().item()
    print('VM_max errorD%', abs(VM_max_p10-VM_max_p10D)/VM_max_p10)
    MAPE_p10_listD.append(errorD.mean().item()/VM_max_p10)
    APE_p10_listD.append(abs(VM_max_p10-VM_max_p10D)/VM_max_p10)
#%%
import pandas as pd
import numpy as np

df=pd.DataFrame()
df['method']=['PyFEA-NN-P0', 'PyFEA-P0', 'BD']
df['node_error']=[np.mean(node_error_p0_mean_listD),
                  np.mean(node_error_p0_mean_listB),
                  np.mean(node_error_p0_mean_listC)]
df['stress_error']=[np.mean(MAPE_p10_listD),
                    np.mean(MAPE_p10_listB),
                    np.mean(MAPE_p10_listC)]
df['peak_stress_error']=[np.mean(APE_p10_listD),
                         np.mean(APE_p10_listB),
                         np.mean(APE_p10_listC)]
df['Time(avg)']=[np.mean(time_cost_listD),
                 np.mean(time_cost_listB),
                 np.mean(time_cost_listC)]
df.to_csv('./app3/table/inverse_p0_result_comparison.csv', index=False)
print(df)



