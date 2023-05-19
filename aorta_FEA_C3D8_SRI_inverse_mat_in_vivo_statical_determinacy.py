import sys
sys.path.append("c3d8")
sys.path.append("mesh")
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
import torch
#import torch.nn.functional as nnF
from torch.linalg import det
from torch import log
from AortaFEModel_C3D8_SRI import cal_element_orientation, cal_pressure_force
from FEModel_C3D8_SRI_fiber import cal_F_tensor_1i, cal_F_tensor_8i, cal_cauchy_stress_force_inverse
from FEModel_C3D8_SRI_fiber import cal_d_sf_dx_and_dx_dr_8i, cal_d_sf_dx_and_dx_dr_1i
from FEModel_C3D8_SRI_fiber import cal_attribute_on_node
from von_mises_stress import cal_von_mises_stress
from polar_decomposition import polar_decomposition
from PolyhedronMesh import PolyhedronMesh
from QuadMesh import QuadMesh
from aorta_mesh import get_solid_mesh_cfg
from FE_lbfgs_ori import LBFGS
import time
#%%
shell_template_str='./app4/p0_true/bav17_AortaModel_P0_best'
idx='24'
mat_name='matMean'
mat_model='GOH_SRI'
mat_init=''
filename_p0="./app4/p0_true/p0_"+idx+"_solid"
filename_pL="./app4/result/p0_"+idx+"_solid_"+mat_name+"_p10"
filename_pH="./app4/result/p0_"+idx+"_solid_"+mat_name+"_p16"
filename_pLsd="./app4/result/p0_"+idx+"_solid_"+mat_name+"_p10_sd_1e4"
filename_pHsd="./app4/result/p0_"+idx+"_solid_"+mat_name+"_p16_sd_1e4"
pessure_Low=10
pessure_High=16
#%%
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=1, type=int)
parser.add_argument('--dtype', default="float64", type=str)
parser.add_argument('--shell_template', default=shell_template_str, type=str)
parser.add_argument('--mesh_p0', default=filename_p0, type=str)#for DEBUG
parser.add_argument('--mesh_pL', default=filename_pL, type=str)
parser.add_argument('--mesh_pH', default=filename_pH, type=str)
parser.add_argument('--mesh_pLsd', default=filename_pLsd, type=str)
parser.add_argument('--mesh_pHsd', default=filename_pHsd, type=str)
parser.add_argument('--pessureL', default=pessure_Low, type=float)
parser.add_argument('--pessureH', default=pessure_High, type=float)
parser.add_argument('--mat_model', default=mat_model, type=str)
parser.add_argument('--mat_init', default=mat_init, type=str)
parser.add_argument('--beta', default=0, type=float)#force_loss
parser.add_argument('--alpha', default=1, type=float)#stress_loss
parser.add_argument('--nFa', default=1, type=int)# 1 or 8
parser.add_argument('--use_Fb', default='False', type=str) # not used in the paper
parser.add_argument('--use_w_loss', default='False', type=str) # not used in the paper
parser.add_argument('--g_w_loss', default=0.01, type=float)
parser.add_argument('--g_divisor', default=1, type=int)#used only if g_w_loss=0
parser.add_argument('--g_threshold', default=10, type=float) # used only if use_w_loss='True'
parser.add_argument('--use_aera_in_w_loss', default='False', type=str) # not used in the paper
parser.add_argument('--use_von_mises', default='False', type=str) # not used in the paper
parser.add_argument('--use_S_true', default='False', type=str)#for DEBUG
parser.add_argument('--use_V_true', default='False', type=str)#for DEBUG
parser.add_argument('--use_Vmean_true', default='False', type=str)#for DEBUG, set nFa=1
parser.add_argument('--use_F_true', default='False', type=str)#for DEBUG
parser.add_argument('--use_Mat_true', default='False', type=str)#for DEBUG, set mat_init = mat_true
parser.add_argument('--max_iter', default=100000, type=int)#100000 is a good number
arg = parser.parse_args()
if arg.mat_init=='none':
    arg.mat_init=''
[print(i) for i in vars(arg).items()];
#%%
device=torch.device("cuda:"+str(arg.cuda))
#device=torch.device("cpu")
if arg.dtype == "float64":
    dtype=torch.float64
elif arg.dtype == "float32":
    dtype=torch.float32
else:
    raise ValueError("unkown dtype:"+arg.dtype)
#%%
filename_shell=arg.shell_template+".pt"
(boundary0, boundary1, Element_surface_pressure, Element_surface_free)=get_solid_mesh_cfg(filename_shell, 1)
#%%
mesh_pL=PolyhedronMesh()
mesh_pL.load_from_torch(arg.mesh_pL+".pt")
mesh_pH=PolyhedronMesh()
mesh_pH.load_from_torch(arg.mesh_pH+".pt")
#------DEBUG------
mesh_p0=PolyhedronMesh()
mesh_p0.load_from_torch(arg.mesh_p0+".pt")
NodeP0=mesh_p0.node.to(dtype).to(device)
S_pL_true=mesh_pL.mesh_data['S']
S_pH_true=mesh_pH.mesh_data['S']
S_pL_true=S_pL_true.to(dtype).to(device)
S_pH_true=S_pH_true.to(dtype).to(device)
#%%
mesh_pLsd=PolyhedronMesh()
mesh_pLsd.load_from_torch(arg.mesh_pLsd+".pt")
mesh_pHsd=PolyhedronMesh()
mesh_pHsd.load_from_torch(arg.mesh_pHsd+".pt")
S_pL_sd=mesh_pLsd.mesh_data['S']
S_pH_sd=mesh_pHsd.mesh_data['S']
S_pL_sd=S_pL_sd.to(dtype).to(device)
S_pH_sd=S_pH_sd.to(dtype).to(device)
#%%
NodepL=mesh_pL.node.to(dtype).to(device)
NodepH=mesh_pH.node.to(dtype).to(device)
Element=mesh_pL.element.to(device)
Element_surface_pressure=Element_surface_pressure.to(device)
n_nodes=NodepL.shape[0]
#%%
element_area_pL=QuadMesh.cal_element_area(mesh_pL.node, Element_surface_free)
element_area_pH=QuadMesh.cal_element_area(mesh_pH.node, Element_surface_free)
element_area_pL=element_area_pL.view(-1).to(dtype).to(device)
element_area_pH=element_area_pH.view(-1).to(dtype).to(device)
#%%
w_loss_S=torch.ones(len(Element), dtype=dtype, device=device, requires_grad=False)
if arg.use_w_loss == 'True':
    S_diff=0.5*(mesh_pLsd.element_data['S_diff']+mesh_pHsd.element_data['S_diff'])
    #S_diff=0.5*(mesh_pLsd.element_data['S_diff_true']+mesh_pHsd.element_data['S_diff_true'])
    S_diff=S_diff.view(-1).to(dtype).to(device)
    g=arg.g_w_loss
    if g == 0:
        from statsmodels.robust.scale import huber
        g_mean, g_std=huber(S_diff.cpu().numpy())
        g=g_mean/arg.g_divisor
        arg.g_w_loss=g
        print("g", g)
    w_loss_S=torch.exp(-S_diff**2/(2*g**2))
    w_loss_S[S_diff>g*arg.g_threshold]=0
#remove boundary elements
w_loss_S[:50*4]=0
w_loss_S[-50*4:]=0
w_loss_S_pL=w_loss_S
w_loss_S_pH=w_loss_S
if arg.use_aera_in_w_loss == 'True':
    w_loss_S_pL=w_loss_S_pL*element_area_pL/element_area_pL.max()
    w_loss_S_pH=w_loss_S_pH*element_area_pH/element_area_pH.max()
#%%
if arg.nFa == 8:
    w_loss_S_pL=w_loss_S_pL.view(-1,1,1,1)
    w_loss_S_pH=w_loss_S_pH.view(-1,1,1,1)
elif arg.nFa == 1:
    w_loss_S_pL=w_loss_S_pL.view(-1,1,1)
    w_loss_S_pH=w_loss_S_pH.view(-1,1,1)
else:
    raise ValueError()
#%%
pL_node_error_mean=((mesh_pLsd.node-mesh_pL.node)**2).sum(dim=1).sqrt().mean().item()
pL_node_error_max=((mesh_pLsd.node-mesh_pL.node)**2).sum(dim=1).sqrt().max().item()
pH_node_error_mean=((mesh_pHsd.node-mesh_pH.node)**2).sum(dim=1).sqrt().mean().item()
pH_node_error_max=((mesh_pHsd.node-mesh_pH.node)**2).sum(dim=1).sqrt().max().item()
print("pL_node_error: mean", pL_node_error_mean, "max", pL_node_error_max)
print("pH_node_error: mean", pH_node_error_mean, "max", pH_node_error_max)
#%% DEBUG
S_pL_error_mean=(S_pL_true-S_pL_sd).abs().mean().item()/S_pL_true.abs().mean().item()
S_pL_error_max=(S_pL_true-S_pL_sd).abs().max().item()/S_pL_true.abs().mean().item()
S_pH_error_mean=(S_pH_true-S_pH_sd).abs().mean().item()/S_pH_true.abs().mean().item()
S_pH_error_max=(S_pH_true-S_pH_sd).abs().max().item()/S_pH_true.abs().mean().item()
print("S_pL_error: mean", S_pL_error_mean, "max", S_pL_error_max)
print("S_pH_error: mean", S_pH_error_mean, "max", S_pH_error_max)
#%% DEBUG
S_avg_pL_error_mean=(S_pL_true.mean(dim=1)-S_pL_sd.mean(dim=1)).abs().mean().item()/S_pL_true.abs().mean().item()
S_avg_pL_error_max=(S_pL_true.mean(dim=1)-S_pL_sd.mean(dim=1)).abs().max().item()/S_pL_true.abs().mean().item()
S_avg_pH_error_mean=(S_pH_true.mean(dim=1)-S_pH_sd.mean(dim=1)).abs().mean().item()/S_pH_true.abs().mean().item()
S_avg_pH_error_max=(S_pH_true.mean(dim=1)-S_pH_sd.mean(dim=1)).abs().max().item()/S_pH_true.abs().mean().item()
print("S_avg_pL_error: mean", S_avg_pL_error_mean, "max", S_avg_pL_error_max)
print("S_avg_pH_error: mean", S_avg_pH_error_mean, "max", S_avg_pH_error_max)
#%% DEBUG
S_pL_true_vm=cal_von_mises_stress(S_pL_true)
S_pH_true_vm=cal_von_mises_stress(S_pH_true)
S_pL_sd_vm=cal_von_mises_stress(S_pL_sd)
S_pH_sd_vm=cal_von_mises_stress(S_pH_sd)
S_pL_vm_error_mean=(S_pL_true_vm-S_pL_sd_vm).abs().mean().item()/S_pL_true_vm.abs().mean().item()
S_pL_vm_error_max=(S_pL_true_vm-S_pL_sd_vm).abs().max().item()/S_pL_true_vm.abs().mean().item()
S_pH_vm_error_mean=(S_pH_true_vm-S_pH_sd_vm).abs().mean().item()/S_pH_true_vm.abs().mean().item()
S_pH_vm_error_max=(S_pH_true_vm-S_pH_sd_vm).abs().max().item()/S_pH_true_vm.abs().mean().item()
print("S_pL_vm_error: mean", S_pL_vm_error_mean, "max", S_pL_vm_error_max)
print("S_pH_vm_error: mean", S_pH_vm_error_mean, "max", S_pH_vm_error_max)
#%% DEBUG
S_avg_pL_true_vm=S_pL_true_vm.mean(dim=1)
S_avg_pH_true_vm=S_pH_true_vm.mean(dim=1)
S_avg_pL_sd_vm=S_pL_sd_vm.mean(dim=1)
S_avg_pH_sd_vm=S_pH_sd_vm.mean(dim=1)
S_avg_pL_vm_error_mean=(S_avg_pL_true_vm-S_avg_pL_sd_vm).abs().mean().item()/S_pL_true_vm.abs().mean().item()
S_avg_pL_vm_error_max=(S_avg_pL_true_vm-S_avg_pL_sd_vm).abs().max().item()/S_pL_true_vm.abs().mean().item()
S_avg_pH_vm_error_mean=(S_avg_pH_true_vm-S_avg_pH_sd_vm).abs().mean().item()/S_pH_true_vm.abs().mean().item()
S_avg_pH_vm_error_max=(S_avg_pH_true_vm-S_avg_pH_sd_vm).abs().max().item()/S_pH_true_vm.abs().mean().item()
print("S_avg_pL_vm_error: mean", S_avg_pL_vm_error_mean, "max", S_avg_pL_vm_error_max)
print("S_avg_pH_vm_error: mean", S_avg_pH_vm_error_mean, "max", S_avg_pH_vm_error_max)
#%%
data_error_pL={'pL_node_error_mean':pL_node_error_mean,   'pL_node_error_max':pL_node_error_max,
               'S_pL_error_mean':S_pL_error_mean,         'S_pL_error_max':S_pL_error_max,
               'S_avg_pL_error_mean':S_avg_pL_error_mean, 'S_avg_pL_error_max':S_avg_pL_error_max}
data_error_pH={'pH_node_error_mean':pH_node_error_mean,   'pH_node_error_max':pH_node_error_max,
               'S_pH_error_mean':S_pH_error_mean,         'S_pH_error_max':S_pH_error_max,
               'S_avg_pH_error_mean':S_avg_pH_error_mean, 'S_avg_pH_error_max':S_avg_pH_error_max}
#sys.exit()
#%% DEBUG with optimal weight
'''
w_loss_S_pL=torch.ones(len(Element), dtype=dtype, device=device, requires_grad=False)
w_loss_S_pH=torch.ones(len(Element), dtype=dtype, device=device, requires_grad=False)
if arg.use_w_loss == 'True':
    w_loss_S_pL=(S_pL_true.mean(dim=1)-S_pL_sd.mean(dim=1)).abs()/S_pL_true.abs().mean(dim=(1,2,3)).view(-1,1,1)
    w_loss_S_pL=w_loss_S_pL.mean(dim=(1,2))
    w_loss_S_pH=(S_pH_true.mean(dim=1)-S_pH_sd.mean(dim=1)).abs()/S_pH_true.abs().mean(dim=(1,2,3)).view(-1,1,1)
    w_loss_S_pH=w_loss_S_pH.mean(dim=(1,2))
    g=arg.g_w_loss #0.02 good for V0N and V1N
    w_loss_S_pL=torch.exp(-w_loss_S_pL**2/(2*g**2))
    w_loss_S_pH=torch.exp(-w_loss_S_pH**2/(2*g**2))
if arg.use_aera_in_w_loss == 'True':
    w_loss_S_pL=w_loss_S_pL*element_area_pL/element_area_pL.max()
    w_loss_S_pH=w_loss_S_pH*element_area_pH/element_area_pH.max()
w_loss_S=(w_loss_S_pL+w_loss_S_pH)/2
w_loss_S=w_loss_S.to(dtype).to(device)
w_loss_S_pL=w_loss_S
w_loss_S_pH=w_loss_S
#----------------------------
if arg.nFa == 8:
    w_loss_S_pL=w_loss_S_pL.view(-1,1,1,1)
    w_loss_S_pH=w_loss_S_pH.view(-1,1,1,1)
elif arg.nFa == 1:
    w_loss_S_pL=w_loss_S_pL.view(-1,1,1)
    w_loss_S_pH=w_loss_S_pH.view(-1,1,1)
else:
    raise ValueError()
#'''
#%%
w_loss_force_pL=torch.ones((n_nodes,1), dtype=dtype, device=device, requires_grad=False)
w_loss_force_pH=torch.ones((n_nodes,1), dtype=dtype, device=device, requires_grad=False)
if arg.use_w_loss == 'True':
    w_loss_force_pL=cal_attribute_on_node(mesh_pL.node.shape[0], Element, w_loss_S_pL.view(-1,1))
    w_loss_force_pH=cal_attribute_on_node(mesh_pH.node.shape[0], Element, w_loss_S_pH.view(-1,1))
w_loss_force_pL[boundary0]=0
w_loss_force_pL[boundary1]=0
w_loss_force_pH[boundary0]=0
w_loss_force_pH[boundary1]=0
w_loss_force=(w_loss_force_pL+w_loss_force_pH)/2
w_loss_force_pL=w_loss_force
w_loss_force_pH=w_loss_force
#%%
with torch.no_grad():
    OripL=cal_element_orientation(NodepL, Element)
    #OripH=cal_element_orientation(NodepH, Element)
    d_sf_dx_8i_pL, dx_dr_8i_pL, det_dx_dr_8i_pL=cal_d_sf_dx_and_dx_dr_8i(NodepL, Element)
    d_sf_dx_1i_pL, dx_dr_1i_pL, det_dx_dr_1i_pL=cal_d_sf_dx_and_dx_dr_1i(NodepL, Element)
    d_sf_dx_8i_pH, dx_dr_8i_pH, det_dx_dr_8i_pH=cal_d_sf_dx_and_dx_dr_8i(NodepH, Element)
    d_sf_dx_1i_pH, dx_dr_1i_pH, det_dx_dr_1i_pH=cal_d_sf_dx_and_dx_dr_1i(NodepH, Element)
    F_8i_pL_pH=cal_F_tensor_8i(NodepH, Element, NodepL)
    F_1i_pL_pH=cal_F_tensor_1i(NodepH, Element, NodepL)
    det_F_1i_pL_pH=det(F_1i_pL_pH).view(-1,1,1,1)
    force_ext_pL=cal_pressure_force(arg.pessureL, NodepL, Element_surface_pressure)
    force_ext_pH=cal_pressure_force(arg.pessureH, NodepH, Element_surface_pressure)
    #----DEBUG--------
    OriP0=cal_element_orientation(NodeP0, Element)
    F_8i_pL_true=cal_F_tensor_8i(NodepL, Element, NodeP0)
    V_8i_pL_true, R_8i_pL_true, U_8i_pL_true=polar_decomposition(F_8i_pL_true)
    F_1i_pL_true=cal_F_tensor_1i(NodepL, Element, NodeP0)
    V_1i_pL_true, R_1i_pL_true, U_1i_pL_true=polar_decomposition(F_1i_pL_true)
#%%
if "GOH" in arg.mat_model:
    from Mat_GOH_SRI import cal_cauchy_stress
    M0_min=0;    M0_max=120  #c0
    M1_min=0;    M1_max=6000 #k1
    M2_min=0.1;  M2_max=60   #k2
    #M12_min=0;   M12_max=10000
    M3_min=0;    M3_max=1/3
    M4_min=0;    M4_max=np.pi/2
    def get_Mat(m_variable):
        m0=m_variable[0]
        m1=m_variable[1]
        m2=m_variable[2]
        m3=m_variable[3]
        m4=m_variable[4]
        Mat=torch.zeros((1,6),dtype=dtype, device=device)
        if 1:
            Mat[0,0]=M0_min+(M0_max-M0_min)*torch.sigmoid(m0)
            Mat[0,1]=M1_min+(M1_max-M1_min)*torch.sigmoid(m1)
            Mat[0,2]=M2_min+(M2_max-M2_min)*torch.sigmoid(m2)
            Mat[0,3]=M3_min+(M3_max-M3_min)*torch.sigmoid(m3)
            Mat[0,4]=M4_min+(M4_max-M4_min)*torch.sigmoid(m4)
            Mat[0,5]=1e5
        '''
        if 0:
            k1_k2=M12_min+(M12_max-M12_min)*torch.sigmoid(m1)
            k2=M2_min+(M2_max-M2_min)*torch.sigmoid(m2)
            Mat[0,0]=M0_min+(M0_max-M0_min)*torch.sigmoid(m0)
            Mat[0,1]=k1_k2*k2
            Mat[0,2]=k2
            Mat[0,3]=M3_min+(M3_max-M3_min)*torch.sigmoid(m3)
            Mat[0,4]=M4_min+(M4_max-M4_min)*torch.sigmoid(m4)
            Mat[0,5]=1e5
        if 0: #bad: M3=1/3 always using true F
            Mat[0,0]=M0_min+(0.7213*(M0_max-M0_min))*nnF.softplus(m0)
            Mat[0,1]=M1_min+(0.7213*(M1_max-M1_min))*nnF.softplus(m1)
            Mat[0,2]=M2_min+(0.7213*(M2_max-M2_min))*nnF.softplus(m2)
            Mat[0,3]=torch.clamp(M3_min+(0.7213*(M3_max-M3_min))*nnF.softplus(m3), max=M3_max)
            Mat[0,4]=torch.clamp(M4_min+(0.7213*(M4_max-M4_min))*nnF.softplus(m4), max=M4_max)
            Mat[0,5]=1e5
        if 0: #bad Mat[0,0] is m0_max always
            Mat[0,0]=torch.clamp(M0_min+(0.7213*(M0_max-M0_min))*nnF.softplus(m0), max=M0_max)
            Mat[0,1]=torch.clamp(M1_min+(0.7213*(M1_max-M1_min))*nnF.softplus(m1), max=M1_max)
            Mat[0,2]=torch.clamp(M2_min+(0.7213*(M2_max-M2_min))*nnF.softplus(m2), max=M2_max)
            Mat[0,3]=torch.clamp(M3_min+(0.7213*(M3_max-M3_min))*nnF.softplus(m3), max=M3_max)
            Mat[0,4]=torch.clamp(M4_min+(0.7213*(M4_max-M4_min))*nnF.softplus(m4), max=M4_max)
            Mat[0,5]=1e5
        if 0: #bad Mat[0,0] is m0_max always
            Mat[0,0]=torch.clamp(M0_min+(M0_max-M0_min)*(m0+0.5), min=M0_min, max=M0_max)
            Mat[0,1]=torch.clamp(M1_min+(M1_max-M1_min)*(m1+0.5), min=M1_min, max=M1_max)
            Mat[0,2]=torch.clamp(M2_min+(M2_max-M2_min)*(m2+0.5), min=M2_min, max=M2_max)
            Mat[0,3]=torch.clamp(M3_min+(M3_max-M3_min)*(m3+0.5), min=M3_min, max=M3_max)
            Mat[0,4]=torch.clamp(M4_min+(M4_max-M4_min)*(m4+0.5), min=M4_min, max=M4_max)
            Mat[0,5]=1e5
        '''
        return Mat
#%%
if arg.mat_model=='MooneyRivlin_SRI':
    from Mat_MooneyRivlin_SRI import cal_cauchy_stress_ori as cal_cauchy_stress
    M0_min=0;    M0_max=1000
    M1_min=0;    M1_max=1000
    def get_Mat(m_variable):
        m0=m_variable[0]
        m1=m_variable[1]
        Mat=torch.zeros((1,3),dtype=dtype, device=device)
        Mat[0,0]=M0_min+(M0_max-M0_min)*torch.sigmoid(m0)
        Mat[0,1]=M1_min+(M1_max-M1_min)*torch.sigmoid(m1)
        Mat[0,2]=1e5
        return Mat
#%%
def get_m_init():
    m_init=torch.zeros(5, dtype=dtype, device=device, requires_grad=True)
    return m_init
m_variable=get_m_init()
#%%
if arg.nFa==8:
    Fa=torch.zeros((Element.shape[0],8,3,3), dtype=dtype, device=device, requires_grad=True)
elif arg.nFa==1:
    Fa=torch.zeros((Element.shape[0],1,3,3), dtype=dtype, device=device, requires_grad=True)
else:
    raise ValueError("nFa is not 1, not 8")
Fa.data[:,:,0,0]=1
Fa.data[:,:,1,1]=1
Fa.data[:,:,2,2]=1
Fb=torch.ones((Element.shape[0],1,1,1), dtype=dtype, device=device, requires_grad=True)
I=torch.eye(3, dtype=dtype, device=device).view(1,1,3,3).expand(Element.shape[0],1,3,3).contiguous()
#%%
def loss_function(A, B, w, reduction):
    Res=A-B
    Res2=Res**2
    wRes=w*Res
    wRes2=w*Res2
    if reduction == "SSE":
        loss=wRes2.sum()
    if reduction == "MSE":
        loss=wRes2.sum()/w.sum()
    elif reduction == "RMSE":
        loss=(wRes2.sum()/w.sum()).sqrt()
    elif reduction == "SLSE":
        loss=log(1+wRes2).sum()
    elif reduction == "MLSE":
        loss=log(1+wRes2).sum()/w.sum()
    elif reduction == "SAE":
        loss=wRes.abs().sum()
    elif reduction == "MAE":
        loss=wRes.abs().sum()/w.sum()
    return loss
#%%
#----DEBUG----------
#init with true F
#Fa.data=V_8i_pL_true.mean(dim=1, keepdim=True)
#Fb.data=torch.pow(det(V_1i_pL_true).view(-1,1,1,1), 1/3)*I
#%%
def cal_loss(Mat, Fa, Fb, reduction1="SSE", reduction2="SSE"):
    #--------------
    F_8i_pL=0.5*(Fa+Fa.transpose(2,3)).expand(-1,8,3,3)

    #F_bar method
    F_1i_pL=torch.pow(det(F_8i_pL).mean(dim=1).view(-1,1,1,1), 1/3)*I

    if arg.use_Fb == 'True':
        F_1i_pL=Fb.abs()*I

    #----DEBUG------------------
    if arg.use_V_true == 'True':
        F_8i_pL=V_8i_pL_true
        F_1i_pL=V_1i_pL_true
    if arg.use_Vmean_true == 'True':
        F_8i_pL=V_8i_pL_true.mean(dim=1, keepdim=True)
        F_1i_pL=torch.pow(det(V_1i_pL_true).view(-1,1,1,1), 1/3)*I
    #---------------------------

    Ori=OripL # use V_p0_to_pL

    #----DEBUG------------------
    if arg.use_F_true == 'True':
        Ori=OriP0 # use true F
        F_8i_pL=F_8i_pL_true
        F_1i_pL=F_1i_pL_true
    #---------------------------

    force_int_pL, Sd_pL, Sv_pL, Wd_pL, Wv_pL=cal_cauchy_stress_force_inverse(d_sf_dx_8i_pL, d_sf_dx_1i_pL,
                                                                             det_dx_dr_8i_pL, det_dx_dr_1i_pL,
                                                                             n_nodes, Element, F_8i_pL, F_1i_pL,
                                                                             Mat, Ori, cal_cauchy_stress,
                                                                             return_S_W=True)
    S_pL=Sd_pL+Sv_pL

    F_8i_pH=torch.matmul(F_8i_pL_pH, F_8i_pL)
    F_1i_pH=torch.matmul(F_1i_pL_pH, F_1i_pL)
    force_int_pH, Sd_pH, Sv_pH, Wd_pH, Wv_pH=cal_cauchy_stress_force_inverse(d_sf_dx_8i_pH, d_sf_dx_1i_pH,
                                                                             det_dx_dr_8i_pH, det_dx_dr_1i_pH,
                                                                             n_nodes, Element, F_8i_pH, F_1i_pH,
                                                                             Mat, Ori, cal_cauchy_stress,
                                                                             return_S_W=True)
    S_pH=Sd_pH+Sv_pH

    loss_force_pL=loss_function(force_int_pL, force_ext_pL, w_loss_force_pL, reduction1)
    loss_force_pH=loss_function(force_int_pH, force_ext_pH, w_loss_force_pH, reduction1)

    S_pL_target=S_pL_sd
    S_pH_target=S_pH_sd

    #----DEBUG-----------------
    if arg.use_S_true =='True':
        S_pL_target=S_pL_true
        S_pH_target=S_pH_true
    #--------------------------

    if arg.use_von_mises == 'True':
        S_pL_vm=cal_von_mises_stress(S_pL, apply_sqrt=False)# True => loss nan
        S_pH_vm=cal_von_mises_stress(S_pH, apply_sqrt=False)
        S_pL_target_vm=cal_von_mises_stress(S_pL_target, apply_sqrt=False)
        S_pH_target_vm=cal_von_mises_stress(S_pH_target, apply_sqrt=False)

        #print(S_pL_vm.shape, S_pH_vm.shape, S_pL_target_vm.shape, S_pH_target_vm.shape)

        if Fa.shape[1]==8:
            loss_S_pL=loss_function(S_pL_vm, S_pL_target_vm, w_loss_S_pL, reduction2)
            loss_S_pH=loss_function(S_pH_vm, S_pH_target_vm, w_loss_S_pH, reduction2)
        else:
            loss_S_pL=loss_function(S_pL_vm.mean(dim=1), S_pL_target_vm.mean(dim=1), w_loss_S_pL, reduction2)
            loss_S_pH=loss_function(S_pH_vm.mean(dim=1), S_pH_target_vm.mean(dim=1), w_loss_S_pH, reduction2)
    else:
        if Fa.shape[1]==8:
            loss_S_pL=loss_function(S_pL, S_pL_target, w_loss_S_pL, reduction2)
            loss_S_pH=loss_function(S_pH, S_pH_target, w_loss_S_pH, reduction2)
        else:
            loss_S_pL=loss_function(S_pL.mean(dim=1), S_pL_target.mean(dim=1), w_loss_S_pL, reduction2)
            loss_S_pH=loss_function(S_pH.mean(dim=1), S_pH_target.mean(dim=1), w_loss_S_pH, reduction2)

    loss_Fa=Fa.abs().max()
    loss_Fb=(det(F_1i_pL)-1).abs().max()

    return loss_force_pL, loss_force_pH, loss_S_pL, loss_S_pH, loss_Fa, loss_Fb, S_pL, S_pH
#%%
if len(arg.mat_init) == 0:
    Mat_init=get_Mat(m_variable)
else:
    if "GOH" in arg.mat_model:
        Mat_init=[float(m) for m in arg.mat_init.split(",")]
        Mat_init[4]=np.pi*(Mat_init[4]/180)
        Mat_init=torch.tensor([Mat_init], dtype=dtype, device=device)
    elif "MooneyRivlin" in arg.mat_model:
        Mat_init=[float(m) for m in arg.mat_init.split(",")]
        Mat_init=torch.tensor([Mat_init], dtype=dtype, device=device)
    else:
        raise ValueError("unkown mat_model:"+arg.mat_model)
    #----------------------------------
    optimizer_mat_init = LBFGS([m_variable], lr=1, line_search_fn="strong_wolfe", history_size=100, max_iter=1)
    def closure1():
        Mat=get_Mat(m_variable)
        loss=((Mat-Mat_init)**2).sum()
        if loss.requires_grad==True:
            optimizer_mat_init.zero_grad()
            loss.backward()
        return loss
    for iter1 in range(0, 1000):
        opt_cond=optimizer_mat_init.step(closure1)
        if iter1%100==0:
            Mat=get_Mat(m_variable)
            print("init Mat:", iter1, "Mat", Mat.detach().cpu().numpy().tolist())
        if opt_cond == True:
            break
    Mat_init=get_Mat(m_variable)
#----------------------------------
if len(arg.mat_init) > 0:
    optimizer_F_init = LBFGS([Fa], lr=1, line_search_fn="strong_wolfe", history_size =100, max_iter=1)
    if arg.use_Fb == 'True':
        optimizer_F_init = LBFGS([Fa, Fb], lr=1, line_search_fn="strong_wolfe", history_size =100, max_iter=1)
    def closure2(lfn='SSE'):
        Mat=get_Mat(m_variable)
        loss_force_pL, loss_force_pH, loss_S_pL, loss_S_pH, loss_Fa, loss_Fb, S_pL, S_pH=cal_loss(Mat, Fa, Fb, lfn, lfn)
        loss=(loss_force_pL+loss_force_pH)*arg.beta+(loss_S_pL+loss_S_pH)*arg.alpha
        if loss.requires_grad==True:
            optimizer_F_init.zero_grad()
            loss.backward()
        return loss
    for iter1 in range(0, arg.max_iter//10):
        opt_cond=optimizer_F_init.step(closure2)
        if iter1%100 == 0:
            print("init F: iter1", iter1, "loss", closure2(lfn='RMSE').item())
        if opt_cond == True:
            break
#%%
Mat_init=get_Mat(m_variable)
Mat_list=[Mat_init.detach().cpu().numpy().tolist()]
print("Mat_Init", Mat_list[-1][0])
loss_list=[]
loss_force_pL_list=[]
loss_force_pH_list=[]
loss_S_pL_list=[]
loss_S_pH_list=[]
time_list=[]
t0=time.time()
#%%
#optimizer1 = LBFGS([Fa], lr=1, line_search_fn="strong_wolfe", history_size =100, max_iter=1)
#optimizer2 = LBFGS([Fb], lr=1, line_search_fn="strong_wolfe", history_size =100, max_iter=1)
#optimizer1.set_backtracking(t_list=[0.5, 0.1, 0.05, 0.01, 1e-3, 1e-4, 1e-5, 1e-6], c=0.0001, verbose=False)
#optimizer2.set_backtracking(t_list=[0.5, 0.1, 0.05, 0.01, 1e-3, 1e-4, 1e-5, 1e-6], c=0.0001, verbose=False)
#%%
if arg.use_Mat_true != 'True':
    if arg.use_Fb == 'True':
        optimizer_mat = LBFGS([m_variable, Fa, Fb], lr=1, line_search_fn="strong_wolfe", history_size=100, max_iter=1)
    else:
        optimizer_mat = LBFGS([m_variable, Fa], lr=1, line_search_fn="strong_wolfe", history_size=100, max_iter=1)
else:#DEBUG
    if arg.use_Fb == 'True':
        optimizer_mat = LBFGS([Fa, Fb], lr=1, line_search_fn="strong_wolfe", history_size=100, max_iter=1)
    else:
        optimizer_mat = LBFGS([Fa], lr=1, line_search_fn="strong_wolfe", history_size=100, max_iter=1)
#------- DEBUG--------#
if arg.use_V_true == 'True' or arg.use_Vmean_true == 'True' or arg.use_F_true == 'True':
    if arg.use_Mat_true != 'True':
        optimizer_mat = LBFGS([m_variable], lr=1, line_search_fn="strong_wolfe", history_size=100, max_iter=1)
    else:
        temp_var=torch.tensor(0, dtype=dtype,devcie=device)
        optimizer_mat = LBFGS([temp_var], lr=1, line_search_fn="strong_wolfe", history_size=100, max_iter=1)
#optimizer_mat.set_backtracking(t_list=[0.5, 0.1, 0.05, 0.01, 1e-3, 1e-4, 1e-5, 1e-6], c=0.0001, verbose=False)
#%%
#optimizer1.reset_state()
#optimizer2.reset_state()
optimizer_mat.reset_state()
m_variable_good=m_variable.clone().detach()
opt_cond1=False
opt_cond2=False
opt_cond3=False
alpha=arg.alpha
beta=arg.beta
#%%
for iter1 in range(0, arg.max_iter):
    def closure():
        Mat=get_Mat(m_variable)
        loss_force_pL, loss_force_pH, loss_S_pL, loss_S_pH, loss_Fa, loss_Fb, S_pL, S_pH=cal_loss(Mat, Fa, Fb)
        loss=(loss_force_pL+loss_force_pH)*beta+(loss_S_pL+loss_S_pH)*alpha
        if loss.requires_grad==True:
            #optimizer1.zero_grad()
            #optimizer2.zero_grad()
            optimizer_mat.zero_grad()
            loss.backward()
        return loss
    #optimizer1.reset_state()
    #for n in range(0, 100):
    #    opt_cond1=optimizer1.step(closure)
    #opt_cond2=optimizer2.step(closure)
    #optimizer_mat.reset_state()
    #for n in range(0, 10):
    opt_cond3=optimizer_mat.step(closure)

    '''
    if opt_cond1 == "nan" or opt_cond2 == "nan" or opt_cond3 == "nan":
        print("loss is nan, reset optimizer ~~~")
        optimizer1.reset_state()
        optimizer2.reset_state()
        optimizer_mat.reset_state()
        m_variable.data=m_variable_good.data.clone()
        Fa.data=Fa_good.data.clone()
        Fb.data=Fb_good.data.clone()
    else:
        m_variable_good=m_variable.clone().detach()
        Fa_good=Fa.clone().detach()
        Fb_good=Fb.clone().detach()
    '''
    Mat=get_Mat(m_variable)
    loss_force_pL, loss_force_pH, loss_S_pL, loss_S_pH, loss_Fa, loss_Fb, S_pL, S_pH=cal_loss(Mat,Fa,Fb,'RMSE','RMSE')
    loss=(loss_force_pL+loss_force_pH)*beta+(loss_S_pL+loss_S_pH)*alpha
    loss=float(loss)/(2*beta+2*alpha)
    loss_force_pL=float(loss_force_pL)
    loss_force_pH=float(loss_force_pH)
    loss_S_pL=float(loss_S_pL)
    loss_S_pH=float(loss_S_pH)
    loss_Fa=float(loss_Fa)
    loss_Fb=float(loss_Fb)
    loss_list.append(loss)
    loss_force_pL_list.append(loss_force_pL)
    loss_force_pH_list.append(loss_force_pH)
    loss_S_pL_list.append(loss_S_pL)
    loss_S_pH_list.append(loss_S_pH)
    t1=time.time()
    time_list.append(t1-t0)

    Mat=Mat.detach().cpu().numpy()
    if np.isnan(Mat).sum()>0:
        print("nan, break iter1 =", iter1)
        break
    Mat_list.append(Mat)

    opt_flag=0
    if arg.use_Mat_true != 'True':
        if len(Mat_list) > 1:
            Mat1=np.array(Mat_list[-1])
            Mat2=np.array(Mat_list[-2])
            if np.abs(Mat1-Mat2).sum() == 0:
                opt_flag=1

    if iter1==0 or (iter1+1)%100 == 0 or iter1 == arg.max_iter-1 or opt_flag == 1:
        print(iter1, t1-t0, loss_Fa, loss_Fb)
        print(loss, loss_force_pL, loss_force_pH, loss_S_pL, loss_S_pH)
        print("Mat:", Mat_list[-1][0].tolist())
        display.clear_output(wait=False)
        fig, ax = plt.subplots()
        ax.plot(loss_list, 'g')
        ax.plot(loss_force_pL_list, 'r')
        ax.plot(loss_force_pH_list, 'm')
        ax.plot(loss_S_pL_list, 'b')
        ax.plot(loss_S_pH_list, 'c')
        ax.set_ylim(0, 10)
        ax.grid(True)
        display.display(fig)
        plt.close(fig)

    if opt_flag == 1:
        print("Mat does not change any more, break iter1 =", iter1)
        break
#%%
if arg.use_V_true == 'True' or arg.use_Vmean_true == 'True' or arg.use_F_true == 'True':
    print("DEBUG mode, do not save result files")
    sys.exit()
#%%
S_element_pL=S_pL.mean(dim=1)
VM_element_pL=cal_von_mises_stress(S_element_pL)
S_node_pL=cal_attribute_on_node(mesh_pL.node.shape[0], Element, S_element_pL)
VM_node_pL=cal_von_mises_stress(S_node_pL)
mesh_pL_mat=PolyhedronMesh()
mesh_pL_mat.node=mesh_pL.node
mesh_pL_mat.element=mesh_pL.element
mesh_pL_mat.element_data["S"]=S_element_pL.view(-1,9)
mesh_pL_mat.element_data['VM']=VM_element_pL.view(-1,1)
mesh_pL_mat.element_data['area']=element_area_pL.view(-1,1)
mesh_pL_mat.element_data['w_loss_S']=w_loss_S_pL.view(-1,1)
mesh_pL_mat.node_data["S"]=S_node_pL.view(-1,9)
mesh_pL_mat.node_data['VM']=VM_node_pL.view(-1,1)
mesh_pL_mat.mesh_data['arg']=arg
mesh_pL_mat.mesh_data['Mat']=Mat_list
mesh_pL_mat.mesh_data['loss']=loss_list
mesh_pL_mat.mesh_data['loss_force_pL']=loss_force_pL_list
mesh_pL_mat.mesh_data['loss_force_pH']=loss_force_pH_list
mesh_pL_mat.mesh_data['loss_S_pL']=loss_S_pL_list
mesh_pL_mat.mesh_data['loss_S_pH']=loss_S_pH_list
mesh_pL_mat.mesh_data['time']=time_list
mesh_pL_mat.mesh_data['data_error']=data_error_pL
mesh_pL_mat.mesh_data['solution']={'m_variable':m_variable, 'Fa':Fa, 'Fb':Fb}
filename_pL_mat=arg.mesh_pLsd+"_invivo_mat_sd_b"+str(beta)+"a"+str(alpha)+"n"+str(arg.nFa)
if arg.use_Fb == 'True':
    filename_pL_mat+='Fb'
if arg.use_w_loss == 'True' and arg.use_aera_in_w_loss == 'True':
    filename_pL_mat+='Lwa'+str(arg.g_w_loss)+'GT'+str(arg.g_threshold)
if arg.use_w_loss == 'True' and arg.use_aera_in_w_loss != 'True':
    filename_pL_mat+='Lw'+str(arg.g_w_loss)+'GT'+str(arg.g_threshold)
if arg.use_w_loss != 'True' and arg.use_aera_in_w_loss == 'True':
    filename_pL_mat+='La'
if arg.use_Mat_true == 'True':
    filename_pL_mat+='MatTrue'
else:
    if len(arg.mat_init) > 0:
        filename_pL_mat+='MatInit'
if arg.use_S_true =='True':
    filename_pL_mat+='TS'
mesh_pL_mat.save_by_vtk(filename_pL_mat+".vtk")
mesh_pL_mat.save_by_torch(filename_pL_mat+".pt")
print("save", filename_pL_mat)
#---------------------------------------------------
S_element_pH=S_pH.mean(dim=1)
VM_element_pH=cal_von_mises_stress(S_element_pH)
S_node_pH=cal_attribute_on_node(mesh_pH.node.shape[0], Element, S_element_pH)
VM_node_pH=cal_von_mises_stress(S_node_pH)
mesh_pH_mat=PolyhedronMesh()
mesh_pH_mat.node=mesh_pH.node
mesh_pH_mat.element=mesh_pH.element
mesh_pH_mat.element_data["S"]=S_element_pH.view(-1,9)
mesh_pH_mat.element_data['VM']=VM_element_pH.view(-1,1)
mesh_pH_mat.element_data['area']=element_area_pH.view(-1,1)
mesh_pH_mat.element_data['w_loss_S']=w_loss_S_pH.view(-1,1)
mesh_pH_mat.node_data["S"]=S_node_pH.view(-1,9)
mesh_pH_mat.node_data['VM']=VM_node_pH.view(-1,1)
mesh_pH_mat.mesh_data['arg']=arg
mesh_pH_mat.mesh_data['Mat']=Mat_list
mesh_pL_mat.mesh_data['loss']=loss_list
mesh_pL_mat.mesh_data['loss_force_pL']=loss_force_pL_list
mesh_pL_mat.mesh_data['loss_force_pH']=loss_force_pH_list
mesh_pL_mat.mesh_data['loss_S_pL']=loss_S_pL_list
mesh_pL_mat.mesh_data['loss_S_pH']=loss_S_pH_list
mesh_pH_mat.mesh_data['time']=time_list
mesh_pH_mat.mesh_data['data_error']=data_error_pH
mesh_pH_mat.mesh_data['solution']={'m_variable':m_variable, 'Fa':Fa, 'Fb':Fb}
filename_pH_mat=arg.mesh_pHsd+"_invivo_mat_sd_b"+str(beta)+"a"+str(alpha)+"n"+str(arg.nFa)
if arg.use_Fb == 'True':
    filename_pH_mat+='Fb'
if arg.use_w_loss == 'True' and arg.use_aera_in_w_loss == 'True':
    filename_pH_mat+='Lwa'+str(arg.g_w_loss)
if arg.use_w_loss == 'True' and arg.use_aera_in_w_loss != 'True':
    filename_pH_mat+='Lw'+str(arg.g_w_loss)
if arg.use_w_loss != 'True' and arg.use_aera_in_w_loss == 'True':
    filename_pH_mat+='La'
if arg.use_Mat_true == 'True':
    filename_pH_mat+='MatTrue'
else:
    if len(arg.mat_init) > 0:
        filename_pH_mat+='MatInit'
if arg.use_S_true =='True':
    filename_pH_mat+='TS'
mesh_pH_mat.save_by_vtk(filename_pH_mat+".vtk")
mesh_pH_mat.save_by_torch(filename_pH_mat+".pt")
print("save", filename_pH_mat)
[print(i) for i in vars(arg).items()];
