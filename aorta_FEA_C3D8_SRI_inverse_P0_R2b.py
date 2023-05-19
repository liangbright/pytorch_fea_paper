# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 19:39:51 2021

@author: liang
"""

import sys
sys.path.append("../../../MLFEA/code/c3d8")
sys.path.append("../../../MLFEA/code/mesh")
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
import torch
from torch.linalg import det
from AortaFEModel_C3D8_SRI import AortaFEModel, cal_potential_energy, cal_element_orientation
from FEModel_C3D8_SRI_fiber import cal_attribute_on_node
from FEModel_C3D8 import cal_F_tensor_8i, cal_F_tensor_1i
from von_mises_stress import cal_von_mises_stress
from PolyhedronMesh import PolyhedronMesh
from aorta_mesh import get_solid_mesh_cfg
import time
#%%
matA="200, 0, 1, 0.3333, 0, 1e5"
matB="50, 1000, 10, 0.3333, 0, 1e5"
matC="50, 1000, 10, 0.1, 60, 1e5"
mat_soft="10, 0, 1, 0.3333, 0, 1e5"
mat_sd="1e5, 0, 1, 0.3333, 0, 1e5"
all_mat=torch.load('../../../MLFEA/TAA/125mat.pt')['mat_str']
mat95=all_mat[95]
mat10=all_mat[10]
mat24=all_mat[24]
mat64=all_mat[64]
matMean=torch.load('../../../MLFEA/TAA/125mat.pt')['mean_mat_str']
#%%
path='D:/MLFEA/TAA/'
folder_data=path+'data/343c1.5_refine/'
folder_result=path+'data/343c1.5_refine/inverse/'
shell_template_str=folder_data+'bav17_AortaModel_P0_best'
mesh_px_str=folder_data+'matMean/p0_24_solid_matMean_p20_i10'
mesh_p0_init_str=''
#mesh_p0_init_str=folder_result+'p0_24_solid_matMean_p20_i10_inverse_p0'
mesh_p0_pred_str=folder_result+'p0_24_solid_matMean_p20_i10_inverse_p0'
mat_str=matMean
#%%
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=1, type=int)
parser.add_argument('--dtype', default="float64", type=str)
parser.add_argument('--n_layers', default=1, type=int)
parser.add_argument('--shell_template', default=shell_template_str, type=str)
parser.add_argument('--mesh_px', default=mesh_px_str, type=str)
parser.add_argument('--mesh_p0_pred', default=mesh_p0_pred_str, type=str)
parser.add_argument('--mesh_p0_init', default=mesh_p0_init_str, type=str)
parser.add_argument('--mat', default=mat_str, type=str)
parser.add_argument('--pressure', default=10, type=float)
parser.add_argument('--max_iter1', default=100, type=int)
parser.add_argument('--max_iter2', default=10001, type=int)
parser.add_argument('--use_stiffness', default=False, type=bool)
parser.add_argument('--reform_stiffness_interval', default=10, type=int)
parser.add_argument('--warm_up_T0', default=1000, type=int)
parser.add_argument('--loss1', default=0.1, type=float)#0.1 for 100 steps
parser.add_argument('--Rmax', default=0.005, type=float)
parser.add_argument('--save_by_vtk', default='True', type=str)#True: save *.vtk file in addition to *.pt file
parser.add_argument('--plot', default='False', type=str)
arg = parser.parse_args()
arg.save_by_vtk = True if arg.save_by_vtk == 'True' else False
arg.plot = True if arg.plot == 'True' else False
print(arg)
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
t0=time.time()
(boundary0, boundary1,
 Element_surface_pressure, Element_surface_free)=get_solid_mesh_cfg(arg.shell_template+'.pt', arg.n_layers)
#%%
Mesh_x=PolyhedronMesh()
Mesh_x.load_from_torch(arg.mesh_px+".pt")
Node_x=Mesh_x.node.to(dtype).to(device)
Element=Mesh_x.element.to(device)
#%%
from Mat_GOH_SRI import cal_1pk_stress, cal_cauchy_stress
#from InvivoMat_GOH_SRI import cal_1pk_stress, cal_cauchy_stress
Mat=[float(m) for m in arg.mat.split(",")]
Mat[4]=np.pi*(Mat[4]/180)
Mat=torch.tensor([Mat], dtype=dtype, device=device)
aorta_model=AortaFEModel(Node_x, Element, None, boundary0, boundary1, Element_surface_pressure,
                         Mat, cal_1pk_stress, cal_cauchy_stress, dtype, device, mode='inverse_p0')
aorta_model.element_orientation_on_deformed_mesh=False
pressure=arg.pressure
#%%
def loss1a_function(force_int, force_ext):
    loss=((force_int-force_ext)**2).sum(dim=1).sqrt().mean()
    return loss
#%%
def loss1b_function(force_int, force_ext):
    loss=((force_int-force_ext)**2).sum()
    return loss
#%%
print('len(arg.mesh_p0_init)', len(arg.mesh_p0_init), 'arg.mesh_p0_init', arg.mesh_p0_init)
if len(arg.mesh_p0_init) > 0:
    Mesh_X_init=PolyhedronMesh()
    Mesh_X_init.load_from_torch(arg.mesh_p0_init+".pt")
    Node_X_init=Mesh_X_init.node.to(dtype).to(device)
    u_field_init=Node_x-Node_X_init
    u_field_init=u_field_init*aorta_model.state['mask']
    print("initilize u_field with u_field_init")
    aorta_model.state['u_field'].data=u_field_init[aorta_model.state['free_node']].to(dtype).to(device)
    out=aorta_model.cal_energy_and_force(pressure)
    TPE1=out['TPE1']; TPE2=out['TPE2']; SE=out['SE']
    force_int=out['force_int']; force_ext=out['force_ext']
    force_int_of_element=out['force_int_of_element']
    F=out['F']; u_field=out['u_field']
    loss1=loss1a_function(force_int, force_ext)
    loss1=loss1.item()
    loss2=cal_potential_energy(force_int-force_ext, u_field)
    loss2=loss2.item()
    force_avg=(force_int_of_element**2).sum(dim=2).sqrt().mean()
    force_res=((force_int-force_ext)**2).sum(dim=1).sqrt()
    R=force_res/(force_avg+1e-10)
    Rmean=R.mean().item()
    Rmax=R.max().item()
    J=det(F)
    print("init: pressure", pressure)
    print("init: TPE", float(TPE1), "loss1", loss1, "Rmax", Rmax, "max|J-1|", float((J-1).abs().max()))
    sum_force_res=force_res.sum().item()
    print("init: sum(force_res)", force_res.sum().item(), "1/sum(force_res)", 1/sum_force_res)
    #1/sum_force_res should be similar to t_default for the first step
    if Rmax < arg.Rmax: # and loss1 < arg.loss1:
        opt_cond=True
        arg.max_iter1=0
        print("mesh_p0_init is optimal, set arg.max_iter1=0, set opt_cond=True")
        if arg.mesh_p0_init == arg.mesh_p0_pred:
            #no need to do anything
            print('mesh_p0_init is mesh_p0_pred, exit')
            sys.exit()
#%%
    del Mesh_X_init, Node_X_init, u_field_init
    del out, TPE2, SE, force_int, force_ext, J, F, u_field, R
    del sum_force_res, force_res, force_avg, force_int_of_element
    loss1_init=loss1
    Rmax_init=Rmax
    TPE_init=TPE1
    loss1_init=loss1
    loss2_init=loss2
    del loss1, loss2, Rmax, TPE1
#%%
t1=time.time()
print('init: time cost', t1-t0)
#%%
'''
from FE_lbfgs_ori import LBFGS
closure_opt={"output":"loss1", "loss1_fn":loss1b_function}
aorta_model.state['u_field'].requires_grad=True
optimizer = LBFGS([aorta_model.state['u_field']], lr=0.5, line_search_fn="strong_wolfe",
                  tolerance_grad =1e-10, tolerance_change=1e-20, history_size =100, max_iter=1)
#'''
#%%
from FE_lbfgs import LBFGS
#'''
closure_opt={"output":"TPE", "loss1_fn":loss1a_function}
aorta_model.state['u_field'].requires_grad=True
optimizer = LBFGS([aorta_model.state['u_field']], lr=1, line_search_fn="backtracking",
                  tolerance_grad =1e-10, tolerance_change=1e-20, history_size =100, max_iter=1)
optimizer.set_backtracking(c=0.5, verbose=False)
optimizer.set_backtracking(t_list=np.logspace(np.log10(0.5), np.log10(1e-5), 10).tolist())
optimizer.set_backtracking(t_default=0.5, t_default_init='auto')
#t_default_init for the first step must be very small to prevent nan
#'''
#%%
def reform_stiffness(pressure_t, iter2):
    ta=time.time()
    Output=aorta_model.cal_energy_and_force(pressure_t, return_stiffness="sparse")
    H=Output['H']
    H=H.astype("float64")
    optimizer.reset_state()
    optimizer.set_H0(H)
    tb=time.time()
    print("reform stiffness done:", tb-ta, "iter2", iter2)
    return tb-ta
#%%
loss1_list=[]
loss2_list=[]
Rmax_list=[]
TPE_list=[]
time_list=[]
flag_list=[]
t0=time.time()
#%%
Ugood=aorta_model.state['u_field'].clone().detach()
Ubest=Ugood
loss1_best=None
Rmax_best=None
TPE_best=None
max_iter1=int(arg.max_iter1)
max_iter2=int(arg.max_iter2)
fatal_error_flag=False
minor_error_flag=False
using_stiffness=False
#%%
for iter1 in range(0, max_iter1):
    pressure_iter=((iter1+1)/max_iter1)*pressure
    print("pressure_iter", pressure_iter, 't', (iter1+1)/max_iter1)

    iter2_error_idx=-1
    enable_stiffness=False
    if max_iter1 > 1:
        optimizer.reset_state()
        using_stiffness=False

    #if iter1 < 90:
    #    max_iter2=1000
    #else:
    #    max_iter2=10000

    for iter2 in range(0, max_iter2):
        #-------------------------------------------------------
        if arg.use_stiffness == True and iter2 >= arg.warm_up_T0 and closure_opt['output']=='TPE':
            if fatal_error_flag == False:
                enable_stiffness=True
        if minor_error_flag == True and using_stiffness == False:
            enable_stiffness=False
        #-------------------------------------------------------
        if enable_stiffness == True:
            if iter2%arg.reform_stiffness_interval == 0:
                try:
                    reform_stiffness(pressure_iter, iter2)
                    using_stiffness=True
                except:
                    optimizer.reset_state()
                    using_stiffness=False
        #------------------------------------------------
        def closure(output=closure_opt["output"], loss1_fn=closure_opt["loss1_fn"]):
            out=aorta_model.cal_energy_and_force_for_inverse_p0(pressure_iter, detach_X=True)
            TPE1=out['TPE1']; TPE2=out['TPE2']; SE=out['SE']
            force_int=out['force_int']; force_ext=out['force_ext']
            F=out['F']; u_field=out['u_field']
            loss1=loss1_fn(force_int, force_ext)
            force_res=force_int-force_ext
            loss2=cal_potential_energy(force_res, u_field)
            if output == "TPE":
                loss=loss2
            elif output == "loss1":
                loss=loss1
            elif output == "TPE+loss1":
                loss=loss2+loss1
            elif output == "loss2":
                return loss2
            else:
                loss=loss2
            if loss.requires_grad==True:
                optimizer.zero_grad()
                loss.backward()
            if output == "TPE":
                return TPE1
            elif output == "loss1":
                return loss1
            elif output == "TPE+loss1":
                return TPE1+loss1
            elif output == "loss2":
                return loss2
            else:
                loss1=float(loss1)
                loss2=float(loss2)
                TPE1=float(TPE1)
                F=F.detach()
                force_int=force_int.detach()
                force_ext=force_ext.detach()
                force_int_of_element=out['force_int_of_element'].detach()
                return loss1, loss2, TPE1, F, force_int, force_ext, force_int_of_element
        #------------------------------------------------
        try:
            opt_cond=optimizer.step(closure)
            flag_linesearch=optimizer.get_linesearch_flag()
            t_linesearch=optimizer.get_linesearch_t()
            output=torch.no_grad()(closure)(output='all', loss1_fn=loss1a_function)
            loss1, loss2, TPE, F, force_int, force_ext, force_int_of_element=output
            J=det(F)
        except:
            opt_cond=False
            fatal_error_flag=True
            aorta_model.state['u_field'].data=Ugood.clone().detach()
            optimizer.reset_state()
            using_stiffness=False
            if iter2==0:
                print("except at iter2 = 0, break")
                break
            else:
                print("except at iter2 > 0, continue")
                continue
        #------------------------------------------------
        #check error
        fatal_error_flag=False
        minor_error_flag=False
        #anythin is nan or inf?
        if (np.isnan(TPE) == True or np.isinf(TPE) == True
            or np.isnan(loss1) == True or np.isinf(loss1) == True
            or np.isnan(loss2) == True or np.isinf(loss2) == True):
            opt_cond=False
            fatal_error_flag=True
            print("iter1", iter1, "iter2", iter2, "errorA, using_stiffness is",using_stiffness)
        #It is possible that TPE > 0, not as bad as TPE=nan or +/- inf
        if TPE > 0:
             opt_cond=False
             minor_error_flag=True
             print("iter1", iter1, "iter2", iter2, "errorB, using_stiffness is",using_stiffness)
        #some elements may deform too much
        J_min=J.min().item()
        J_max=J.max().item()
        J_error=(J-1).abs().max().item()
        if J_error > 0.5:
            opt_cond=False
            fatal_error_flag=True
            print("iter1", iter1, "iter2", iter2, "errorC, using_stiffness is",using_stiffness)
        if 0.1 < J_error < 0.5:
            opt_cond=False
            minor_error_flag=True
            print("iter1", iter1, "iter2", iter2, "errorD, using_stiffness is",using_stiffness)
        #------------------------------------------------
        if fatal_error_flag==True or minor_error_flag==True:
            print("  TPE", TPE, "loss1", loss1, "loss2", loss2)
            print("  flag",flag_linesearch,"t_linesearch",t_linesearch,"min(J)",J_min,"max(J)",J_max)
        #------------------------------------------------
        if fatal_error_flag==True:
            aorta_model.state['u_field'].data=Ugood.clone().detach()
            optimizer.reset_state()
            using_stiffness=False
            #bad
            #if enable_stiffness == True:
            #    reform_stiffness(pressure_iter, iter2)
            iter2_error_idx=iter2
            if iter2==0:
                print("error at iter2 = 0, break")
                break
            else:
                continue
        #skip the code below if fatal_error_flag is True
        #------------------------------------------------
        flag_list.append(flag_linesearch)

        t1=time.time()
        time_list.append(t1-t0)

        TPE_list.append(TPE)
        loss1_list.append(loss1)
        loss2_list.append(loss2)

        force_avg=(force_int_of_element**2).sum(dim=2).sqrt().mean()
        force_res=((force_int-force_ext)**2).sum(dim=1).sqrt()
        R=force_res/(force_avg+1e-10)
        Rmean=R.mean().item()
        Rmax=R.max().item()
        Rmax_list.append(Rmax)
        #-------------------------------------------------------
        if minor_error_flag == False and len(Rmax_list) > 1:
            Ugood=aorta_model.state['u_field'].clone().detach()
            if Rmax < min(Rmax_list[:-1]) or max_iter1 > 1:
                Ubest=Ugood
                loss1_best=loss1
                Rmax_best=Rmax
                TPE_best=TPE
        #------------------------------------------
        #check convergance
        opt_cond=False
        if Rmax < arg.Rmax or loss1 < arg.loss1:
            Ubest=Ugood=aorta_model.state['u_field'].clone().detach()
            loss1_best=loss1
            Rmax_best=Rmax
            TPE_best=TPE
            opt_cond=True
        #------------------------------------------
        flag_print=False
        if iter2==0 or opt_cond == True:
            flag_print=True
        if enable_stiffness==False and iter2%100 == 0:
           flag_print=True
        if enable_stiffness==True:
            if iter2%arg.reform_stiffness_interval==0 or (iter2+1)%(arg.reform_stiffness_interval//2)==0:
                flag_print=True
        if flag_print==True:
            print("iter1", iter1, "iter2", iter2, "pressure_iter", pressure_iter, "time", t1-t0)
            print("  flag", flag_linesearch, "t_linesearch", t_linesearch, "max|J-1|", J_error)
            print('  Rmax', Rmax, 'Rmean', Rmean, "loss1", loss1, "loss2", loss2, "TPE", TPE)
            if arg.plot == True:
                display.clear_output(wait=False)
                fig, ax = plt.subplots()
                ax.plot(np.array(loss1_list)/max(loss1_list), 'r')
                ax.plot(loss1_list, 'm')
                if len(TPE_list) > 10:
                    negTPE=-np.array(TPE_list)
                    Vmax=np.max(negTPE[(np.isinf(negTPE)==False)&(negTPE>0)])
                    ax.plot(negTPE/Vmax, 'b')
                ax.plot(-np.array(flag_list), 'g.')
                ax.set_ylim(0, 1)
                ax.grid(True)
                display.display(fig)
                plt.close(fig)
        #-------------------------------------
        if opt_cond == True:
            print("opt_cond is True, break iter2 =", iter2)
            break
#%%
aorta_model.state['u_field'].data=Ubest.clone().detach()
u_field=aorta_model.get_u_field()
Node_X=Node_x-u_field
Sd, Sv=aorta_model.cal_stress('cauchy')
S=Sd+Sv
S_element=S.mean(dim=1)
VM_element=cal_von_mises_stress(S_element)
S_node=cal_attribute_on_node(Node_x.shape[0], Element, S_element)
VM_node=cal_von_mises_stress(S_node)
Mesh_X=PolyhedronMesh()
Mesh_X.element=Element.detach().cpu()
Mesh_X.node=Node_X.detach().cpu()
Mesh_X.element_data['S']=S_element.view(-1,9)
Mesh_X.element_data['VM']=VM_element.view(-1,1)
Mesh_X.node_data['S']=S_node.view(-1,9)
Mesh_X.node_data['VM']=VM_node.view(-1,1)
Mesh_X.mesh_data['arg']=arg
Mesh_X.mesh_data['TPE']=TPE_best
Mesh_X.mesh_data['loss1']=loss1_best
Mesh_X.mesh_data['Rmax']=Rmax_best
Mesh_X.mesh_data['time']=time_list[-1]
Mesh_X.mesh_data['opt_cond']=opt_cond
mesh_p0_pred=arg.mesh_p0_pred
if opt_cond == False:
    mesh_p0_pred=arg.mesh_p0_pred+'_error'
Mesh_X.save_by_torch(mesh_p0_pred+".pt")
if arg.save_by_vtk == True:
    Mesh_X.save_by_vtk(mesh_p0_pred+".vtk")
print("saved", mesh_p0_pred)
