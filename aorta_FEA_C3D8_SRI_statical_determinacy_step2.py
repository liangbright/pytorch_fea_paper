# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:10:41 2022

@author: liang
"""
import torch
#%%
shape_id=171
mat_model='GOH_SRI'
matMean=torch.load('./app4/p0_true/125mat.pt')['mean_mat_str']
mat_name='matMean'
print('mat_name', mat_name)
#%%
cuda='0'
folder="./app4/result"
idx=str(shape_id)
filename_pL=folder+"/p0_"+idx+"_solid_"+mat_name+"_p10"
filename_pH=folder+"/p0_"+idx+"_solid_"+mat_name+"_p16"
filename_pLsd=folder+"/p0_"+idx+"_solid_"+mat_name+"_p10_sd"
filename_pHsd=folder+"/p0_"+idx+"_solid_"+mat_name+"_p16_sd"
#%%
def cal_S_diff_old(S_list):
    #S_list.shape (?, 4950, 8, 3, 3)
    S=S_list.mean(dim=2)
    Smean=S.mean(dim=0, keepdim=True)
    temp=(S-Smean).abs().max(dim=0)[0]
    S_diff=temp/Smean.abs().mean()
    #S_diff=S.std(dim=0)/S.mean(dim=0).abs().mean()
    #print(S_diff.shape) # (4950,3,3)
    S_diff=S_diff.mean(dim=(1,2))
    return S_diff
#%%
def cal_S_diff_old2(S_list):
    #S_list.shape (?, 4950, 8, 3, 3)
    S=S_list.mean(dim=2) #(?,4950,3,3)
    Smean=S.mean(dim=0)  #(4950,3,3)
    Smax=S.max(dim=0)[0] #(4950,3,3)
    Smin=S.min(dim=0)[0] #(4950,3,3)
    S_diff=0.5*(Smax-Smin)/Smean.abs().mean(dim=(1,2), keepdim=True)
    S_diff=S_diff.mean(dim=(1,2))
    return S_diff
#%%
def cal_S_diff(S_list):
    #S_list.shape (?, 4950, 8, 3, 3)
    S=S_list.mean(dim=2) #(?,4950,3,3)
    Smean=S.mean(dim=0)  #(4950,3,3)
    Sstd=S.std(dim=0)    #(4950,3,3)
    S_diff=Sstd/Smean.abs().mean(dim=(1,2), keepdim=True)
    S_diff=S_diff.mean(dim=(1,2))
    return S_diff
#%%
mat_sd_c0_list=['1e3', '2e3', '3e3', '4e3', '5e3', '6e3', '7e3', '8e3', '9e3', '1e4', '2e4', '3e4', '4e4']
#%%
import sys
sys.path.append("../../../MLFEA/code/mesh")
from PolyhedronMesh import PolyhedronMesh
#--------------------------------------------------
S_list_pL=[]
node_list_pL=[]
for mat_sd_c0 in mat_sd_c0_list:
    try:
        mesh_sd=PolyhedronMesh()
        mesh_sd.load_from_torch(filename_pLsd+'_'+mat_sd_c0+".pt")
        S=mesh_sd.mesh_data['S']
        S_list_pL.append(S.view(1,*S.shape))
        node=mesh_sd.node
        node_list_pL.append(node.view(1,node.shape[0],node.shape[1]))
        del mesh_sd, S, node
    except:
        print("cannot load: mat_sd_c0", mat_sd_c0)
node_list_pL=torch.cat(node_list_pL,dim=0)
node_diff_pL=node_list_pL.std(dim=0).sum(dim=1)
S_list_pL=torch.cat(S_list_pL, dim=0)
S_diff_pL=cal_S_diff(S_list_pL)
S_pL_sd=S_list_pL.mean(dim=0)
#%%
S_list_pH=[]
node_list_pH=[]
for mat_sd_c0 in mat_sd_c0_list:
    try:
        mesh_sd=PolyhedronMesh()
        mesh_sd.load_from_torch(filename_pHsd+'_'+mat_sd_c0+".pt")
        S=mesh_sd.mesh_data['S']
        S_list_pH.append(S.view(1,*S.shape))
        node=mesh_sd.node
        node_list_pH.append(node.view(1,node.shape[0],node.shape[1]))
        del mesh_sd, S, node
    except:
        print("cannot load: mat_sd_c0", mat_sd_c0)
node_list_pH=torch.cat(node_list_pH,dim=0)
node_diff_pH=node_list_pH.std(dim=0).sum(dim=1)
S_list_pH=torch.cat(S_list_pH, dim=0)
S_diff_pH=cal_S_diff(S_list_pH)
S_pH_sd=S_list_pH.mean(dim=0)
#%%
S_diff=0.5*(S_diff_pL+S_diff_pH)
from statsmodels.robust.scale import huber
g_mean, g_std=huber(S_diff.numpy())
g=g_mean+g_std
print("g", g, "g_mean", g_mean, "g_std", g_std)
S_weight=torch.exp(-S_diff**2/(2*g**2))
import matplotlib.pyplot as plt
fig, ax = plt.subplots(3,1,constrained_layout=True)
ax[0].hist(S_diff_pL.numpy(), bins=100)
ax[0].set_title('S_diff_pL')
ax[1].hist(S_diff_pH.numpy(), bins=100)
ax[1].set_title('S_diff_pH')
ax[2].hist(1-S_weight.numpy(), bins=100)
ax[2].set_title('1-S_weight')
print("S_diff: mean", S_diff.mean().item(), "max", S_diff.max().item())
#%%
mesh_pL=PolyhedronMesh()
mesh_pL.load_from_torch(filename_pL+".pt")
mesh_pH=PolyhedronMesh()
mesh_pH.load_from_torch(filename_pH+".pt")
#------DEBUG------
S_pL_true=mesh_pL.mesh_data['S']
S_pH_true=mesh_pH.mesh_data['S']
S_diff_pL_true=(S_pL_true.mean(dim=1)-S_pL_sd.mean(dim=1)).abs()/S_pL_true.abs().mean(dim=(1,2,3)).view(-1,1,1)
S_diff_pL_true=S_diff_pL_true.mean(dim=(1,2))
S_diff_pH_true=(S_pH_true.mean(dim=1)-S_pH_sd.mean(dim=1)).abs()/S_pH_true.abs().mean(dim=(1,2,3)).view(-1,1,1)
S_diff_pH_true=S_diff_pH_true.mean(dim=(1,2))
S_diff_true=(S_diff_pL_true+S_diff_pH_true)/2
h_mean, h_std=huber(S_diff_true.numpy())
h=h_mean+h_std
print("h", h, "h_mean", h_mean, "h_std", h_std)
S_weight_true=torch.exp(-S_diff_true**2/(2*0.02**2))
import matplotlib.pyplot as plt
fig, ax = plt.subplots(3,1,constrained_layout=True)
ax[0].hist(S_diff_pL_true.numpy(), bins=100)
ax[0].set_title('S_diff_pL_true')
ax[1].hist(S_diff_pH_true.numpy(), bins=100)
ax[1].set_title('S_diff_pH_true')
ax[2].hist(1-S_weight_true.numpy(), bins=100)
ax[2].set_title('1-S_weight_true')
print("S_diff_true: mean", S_diff_true.mean().item(), "max", S_diff_true.max().item())
#%%
fig, ax = plt.subplots()
ax.plot(S_diff.numpy(), S_diff_true.numpy(), '.', markersize=0.1)
#%%
mesh_pLsd=PolyhedronMesh()
mesh_pLsd.node=node_list_pL.mean(dim=0)
mesh_pLsd.element=mesh_pL.element
mesh_pLsd.mesh_data['S']=S_pL_sd
mesh_pLsd.node_data['node_diff']=node_diff_pL.view(-1,1)
mesh_pLsd.element_data['S_diff']=S_diff_pL.view(-1,1)
mesh_pLsd.element_data['S_weight']=S_weight.view(-1,1)
mesh_pLsd.element_data['S_diff_true']=S_diff_pL_true.view(-1,1)
mesh_pLsd.element_data['S_weight_true']=S_weight_true.view(-1,1)
mesh_pLsd.save_by_vtk(filename_pLsd+'.vtk')
mesh_pLsd.save_by_torch(filename_pLsd+'.pt')

mesh_pHsd=PolyhedronMesh()
mesh_pHsd.node=node_list_pH.mean(dim=0)
mesh_pHsd.element=mesh_pH.element
mesh_pHsd.mesh_data['S']=S_pH_sd
mesh_pHsd.node_data['node_diff']=node_diff_pH.view(-1,1)
mesh_pHsd.element_data['S_diff']=S_diff_pH.view(-1,1)
mesh_pHsd.element_data['S_weight']=S_weight.view(-1,1)
mesh_pHsd.element_data['S_diff_true']=S_diff_pH_true.view(-1,1)
mesh_pHsd.element_data['S_weight_true']=S_weight_true.view(-1,1)
mesh_pHsd.save_by_vtk(filename_pHsd+'.vtk')
mesh_pHsd.save_by_torch(filename_pHsd+'.pt')


