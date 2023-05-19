import sys
sys.path.append("c3d8")
sys.path.append("mesh")
import torch
import numpy as np
from PolyhedronMesh import PolyhedronMesh
#%%
def get_node(filename):
    file = open(filename, 'r')
    Lines = file.readlines()
    file.close()
    node=[]
    for line in Lines:
        node.append([float(a) for a in line.split(",")])
    node=torch.tensor(node, dtype=torch.float64)
    return node
#%%
def get_stress(filename, M):
    file = open(filename, 'r')
    Lines = file.readlines()
    file.close()
    stress=[]
    for m in range(0, M):
        S=torch.zeros((8,3,3), dtype=torch.float64)
        for i in range(0, 8):
            j=8*m+i
            S11, S22, S33, S12, S13, S23=[float(a) for a in Lines[j].split(",")]
            S[i,0,0]=S11
            S[i,1,1]=S22
            S[i,2,2]=S33
            S[i,0,1]=S12; S[i,1,0]=S12
            S[i,0,2]=S13; S[i,2,0]=S13
            S[i,1,2]=S23; S[i,2,1]=S23
        S=S.mean(dim=0, keepdim=True)
        stress.append(S)
    stress=torch.cat(stress, dim=0)
    return stress
#%% log
def get_time_cost(filename):
    file = open(filename, 'r')
    Lines = file.readlines()
    file.close()
    t0=None
    t1=None
    for n in range(0, len(Lines)):
        line=Lines[n]
        if "Begin Analysis Input File Processor" in line:
            #6/18/2022 5:12:00 PM
            temp=Lines[n+1]
            temp=temp.replace('\n', '')
            temp=temp.split(" ")
            t0=[float(a) for a in temp[-2].split(":")]
            #print(t0)
        if "Run SMASimUtility.exe" in line:
            temp=Lines[n+1]
            temp=temp.replace('\n', '')
            temp=temp.split(" ")
            t1=[float(a) for a in temp[-2].split(":")]
            #print(t1)
            break
    if t0 is not None and t1 is not None:
        if t0[0] > t1[0]:
            #t0:[12.0, 59.0, 37.0] 12:59:37 AM
            #t1:[1.0, 3.0, 3.0]     1:03:03 AM
            t1[0]=t0[0]+1
        t=(t1[0]-t0[0])*60*60+(t1[1]-t0[1])*60+t1[2]-t0[2]
    else:
        t=-1
    return t
#%% log
def check_convergence(filename):
    try:
        file = open(filename, 'r')
        Lines = file.readlines()
        file.close()
    except:
        return False
    if "errors" in Lines[-1]:
        return False
    else:
        return True
#%%
matMean=torch.load('./app1/125mat.pt')['mean_mat']
matMean[4]=np.pi*(matMean[4]/180)
matMean=torch.tensor([matMean], dtype=torch.float64)
#%%
from aorta_mesh import get_solid_mesh_cfg
filename_shell='./app1/bav17_AortaModel_P0_best.pt'
(boundary0, boundary1, Element_surface_pressure, Element_surface_free)=get_solid_mesh_cfg(filename_shell, n_layers=1)
#%%
data_path="./app1/pyfea/"
abaqus_path="./app1/abaqus/"
#%% read abaqus results
'''
meshA=PolyhedronMesh()
meshA.load_from_torch(data_path+"p0_0_solid_matMean_p20_i90.pt")
for n in range(0, 0):
    if check_convergence(abaqus_path+str(n)+".log") == False:
        print('not converged:', n)
        continue
    meshB=PolyhedronMesh()
    meshB.node=get_node(abaqus_path+str(n)+"U.txt")
    meshB.element=meshA.element
    S_element=get_stress(abaqus_path+str(n)+"S.txt", meshB.element.shape[0])
    VM_element=cal_von_mises_stress(S_element)
    S_node=cal_attribute_on_node(meshB.node.shape[0], meshB.element, S_element)
    VM_node=cal_von_mises_stress(S_node)
    meshB.element_data['S']=S_element.view(-1,9).detach().cpu()
    meshB.element_data['VM']=VM_element.view(-1,1).detach().cpu()
    meshB.node_data['S']=S_node.view(-1,9).detach().cpu()
    meshB.node_data['VM']=VM_node.view(-1,1).detach().cpu()
    meshB.mesh_data['time']=get_time_cost(abaqus_path+str(n)+".log")
    meshB.save_by_vtk(abaqus_path+"p0_"+str(n)+"_solid_matMean_p18.vtk")
    meshB.save_by_torch(abaqus_path+"p0_"+str(n)+"_solid_matMean_p18.pt")
#'''
#%%
id_list=[]
loss1_list=[]
node_diff=[]
stress_diff=[]
peak_stress_diff=[]
time_costA=[]
time_costB=[]
for n in [24,150,168,171,174,192,318]:
    if n==48:
        continue
    id_list.append(n)
    mesh_p0=PolyhedronMesh()
    mesh_p0.load_from_torch(data_path+"p0_"+str(n)+"_solid_matMean_p20_i0.pt")

    meshA=PolyhedronMesh()#pytorch_fea
    meshA.load_from_torch(data_path+"p0_"+str(n)+"_solid_matMean_p20_i90.pt")#SRI
    #meshA.load_from_torch(data_path+"p0_"+str(n)+"_solid_matMean_p20_GOH_Jv_i90.pt")
    #meshA.load_from_torch(data_path+"p0_"+str(n)+"_solid_matMean_p20_GOH_3Field_i90.pt")
    #meshA.load_from_torch(data_path+"p0_"+str(n)+"_solid_matMean_p20_GOH_Fbar_i90.pt")

    meshB=PolyhedronMesh()#abaqus
    meshB.load_from_torch(abaqus_path+"p0_"+str(n)+"_solid_matMean_p18.pt")

    disp_max=((meshB.node-mesh_p0.node)**2).sum(dim=1).sqrt().max().item()
    print("disp_max", disp_max)
    node_diff.append(((meshA.node-meshB.node)**2).sum(dim=1).sqrt().mean().item()/disp_max)
    #VM_mean=meshB.element_data['VM'].abs().mean().item()
    VM_max=meshB.element_data['VM'].abs().max().item()
    stress_diff.append((meshA.element_data['VM']-meshB.element_data['VM']).abs().mean().item()/VM_max)
    peak_stress_diff.append((meshA.element_data['VM'].max()-VM_max).abs().item()/VM_max)

    time_costA.append(meshA.mesh_data['time'][-1])
    try:
        time_costB.append(meshB.mesh_data['time'][-1])
    except:
        time_costB.append(meshB.mesh_data['time'])
   # break

time_costA=np.array(time_costA)
time_costB=np.array(time_costB)
time_cost=time_costA/time_costB
#%%
import pandas as pd
df=pd.DataFrame()
df['node_diff']=node_diff
df['stress_diff']=stress_diff
df['peak_stress_diff']=peak_stress_diff
df['Time']=time_cost
print(df)
df.to_csv("./app1/table/7shapes_compared_to_abaqus.csv", index=False)
