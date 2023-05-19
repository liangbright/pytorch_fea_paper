#%%
import sys
sys.path.append("c3d8")
sys.path.append("mesh")
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
import torch
from torch.linalg import det
from FEModel_C3D8 import cal_F_tensor_8i, cal_F_tensor_1i
from von_mises_stress import cal_von_mises_stress
from PolyhedronMesh import PolyhedronMesh
from aorta_mesh import get_solid_mesh_cfg
#from aorta_mat_distribution import generate_mat_distribution
import time
#%%
matMean=torch.load('./app3/data/343c1.5/125mat.pt')['mean_mat_str']
#%%
def get_must_points(delta):
    #pressure=20, delta=0.05: 20 must points
    t=0
    n=0
    t_str="0"
    while t<1:
        n=n+1
        t=delta*n
        ts='{:.2f}'.format(t)
        t_str=t_str+","+ts
    return t_str
must_points=get_must_points(0.01)
must_points=''
#%%
folder='./app3/'
#%%
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=1, type=int)
parser.add_argument('--dtype', default="float64", type=str)
parser.add_argument('--n_layers', default=1, type=int)
parser.add_argument('--mesh_px', default=folder+'matMean/p0_174_solid_matMean_p20_i10', type=str)
parser.add_argument('--mesh_p0', default=folder+'inverse/p0_174_solid_matMean_p20_i10_inverse_p0', type=str)
parser.add_argument('--mesh_p0_init', default='', type=str)
parser.add_argument('--t_start_init', default=-1, type=float)
parser.add_argument('--shell_template', default=folder+'bav17_AortaModel_P0_best', type=str)
parser.add_argument('--mat', default=matMean, type=str)
parser.add_argument('--mat_model', default="GOH_SRI", type=str)
parser.add_argument('--pressure', default=10, type=float)
parser.add_argument('--init_step_size', default=0.0001, type=float)
parser.add_argument('--min_step_size', default=1e-10, type=float)
parser.add_argument('--max_step_size', default=0.001, type=float)
parser.add_argument('--max_t_linesearch', default=0.1, type=float)
parser.add_argument('--min_t_linesearch', default=1e-10, type=float)
parser.add_argument('--default_t_linesearch', default=1e-3, type=float)
parser.add_argument('--reform_stiffness_interval', default=10, type=int)
parser.add_argument('--max_iter1', default=1e10, type=int)
parser.add_argument('--max_iter2', default=1000, type=int)
parser.add_argument('--U_ratio', default=1e-2, type=float)
parser.add_argument('--TPE_ratio', default=1, type=float)
parser.add_argument('--E_ratio', default=1, type=float)
parser.add_argument('--f_ratio', default=1, type=float)
parser.add_argument('--loss1', default=0.5, type=float)
parser.add_argument('--Rmax', default=0.005, type=float)
parser.add_argument('--max_retry', default=100, type=int)
parser.add_argument('--random_seed', default=0, type=int)
parser.add_argument('--must_points', default=must_points, type=str)
parser.add_argument('--save_all_t', default='False', type=str) #True: save all (converged) time points
parser.add_argument('--save_by_vtk', default='True', type=str)#True: save *.vtk file in addition to *.pt file
parser.add_argument('--plot', default='False', type=str)#plot loss1, TPE, etc vs sim_t
arg = parser.parse_args()
arg.save_all_t = True if arg.save_all_t == 'True' else False
arg.save_by_vtk = True if arg.save_by_vtk == 'True' else False
arg.plot = True if arg.plot == 'True' else False
print(arg)
#%%
if arg.cuda >=0:
    device=torch.device("cuda:"+str(arg.cuda))
else:
    device=torch.device("cpu")
if arg.dtype == "float64":
    dtype=torch.float64
elif arg.dtype == "float32":
    dtype=torch.float32
else:
    raise ValueError("unkown dtype:"+arg.dtype)
#%%
filename_shell=arg.shell_template+".pt"
(boundary0, boundary1, Element_surface_pressure, Element_surface_free)=get_solid_mesh_cfg(filename_shell, arg.n_layers)
#%%
Mesh_x=PolyhedronMesh()
Mesh_x.load_from_torch(arg.mesh_px+".pt")
Node_x=Mesh_x.node.to(dtype).to(device)
Element=Mesh_x.element.to(device)
#%%
if "SRI" in arg.mat_model:
    from AortaFEModel_C3D8_SRI import AortaFEModel, cal_potential_energy, cal_element_orientation
    from FEModel_C3D8_SRI_fiber import cal_attribute_on_node
else:
    from AortaFEModel_C3D8 import AortaFEModel, cal_potential_energy, cal_element_orientation
    from FEModel_C3D8_fiber import cal_attribute_on_node
#------------------
if arg.mat_model == "GOH_SRI":
    from Mat_GOH_SRI import cal_1pk_stress, cal_cauchy_stress
    if 'distribution' in arg.mat:
        Mat=eval(arg.mat) # run generate_mat_distribution(?)
        Mat=Mat.clone()
        Mat[:,4]=np.pi*(Mat[:,4]/180)
        Mat=Mat.to(dtype=dtype, device=device)
    else:
        Mat=[float(m) for m in arg.mat.split(",")]
        Mat[4]=np.pi*(Mat[4]/180)
        Mat=torch.tensor([Mat], dtype=dtype, device=device)
elif arg.mat_model == "GOH":
    from Mat_GOH import cal_1pk_stress, cal_cauchy_stress
    Mat=[float(m) for m in arg.mat.split(",")]
    Mat[4]=np.pi*(Mat[4]/180)
    Mat=torch.tensor([Mat], dtype=dtype, device=device)
elif arg.mat_model == "GOH_Fbar":
    from Mat_GOH_Fbar import cal_1pk_stress, cal_cauchy_stress
    Mat=[float(m) for m in arg.mat.split(",")]
    Mat[4]=np.pi*(Mat[4]/180)
    Mat=torch.tensor([Mat], dtype=dtype, device=device)
elif arg.mat_model == "GOH_3Field":
    from Mat_GOH_3Field import cal_1pk_stress, cal_cauchy_stress
    Mat=[float(m) for m in arg.mat.split(",")]
    Mat[4]=np.pi*(Mat[4]/180)
    Mat=torch.tensor([Mat], dtype=dtype, device=device)
elif arg.mat_model == "MooneyRivlin_SRI":
    from Mat_MooneyRivlin_SRI import cal_1pk_stress_ori as cal_1pk_stress
    from Mat_MooneyRivlin_SRI import cal_cauchy_stress_ori as cal_cauchy_stress
    Mat=[float(m) for m in arg.mat.split(",")]
    Mat=torch.tensor([Mat], dtype=dtype, device=device)
elif arg.mat_model == 'Invivo_GOH_SRI':
    from InvivoMat_GOH_SRI import cal_1pk_stress, cal_cauchy_stress
    Mat=[float(m) for m in arg.mat.split(",")]
    Mat[4]=np.pi*(Mat[4]/180)
    Mat=torch.tensor([Mat], dtype=dtype, device=device)
else:
    raise ValueError("unkown mat_model:"+arg.mat_model)
#------------------
aorta_model=AortaFEModel(Node_x, Element, None, boundary0, boundary1, Element_surface_pressure,
                         Mat, cal_1pk_stress, cal_cauchy_stress, dtype, device, mode='inverse_p0')
aorta_model.element_orientation_on_deformed_mesh=False
if "Invivo" in arg.mat_model:
    aorta_model.element_orientation_on_deformed_mesh=True
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
t_start=0
if len(arg.mesh_p0_init) > 0:
    t0=time.time()
    Mesh_X_init=PolyhedronMesh()
    Mesh_X_init.load_from_torch(arg.mesh_p0_init+".pt")
    Node_X_init=Mesh_X_init.node.to(dtype).to(device)
    u_field_init=Node_x-Node_X_init
    u_field_init=u_field_init[aorta_model.state['free_node']]
    print("initialize u_field with u_field_init")
    aorta_model.state['u_field'].data=u_field_init.to(dtype).to(device)
    out=aorta_model.cal_energy_and_force(pressure)
    TPE1=out['TPE1']; TPE2=out['TPE2']; SE=out['SE']
    force_int=out['force_int']; force_ext=out['force_ext']
    F=out['F']; u_field=out['u_field']
    loss1=loss1a_function(force_int, force_ext)
    print("Target TPE", float(TPE2), "loss1", loss1, "at pressure", pressure)
    del out, loss1, TPE1, TPE2, SE, force_int, force_ext, F, u_field
    del Mesh_X_init, Node_X_init, u_field_init
    if 0 <= arg.t_start_init <= 1:
        t_start=arg.t_start_init
    else:
        print("search for the optimal t_start of AutoStepper")
        t_start_list=np.linspace(0, 1, int(1/arg.max_step_size))
        loss_list=[]
        for τ in t_start_list:
            out=aorta_model.cal_energy_and_force(pressure*τ)
            TPE1=out['TPE1']; TPE2=out['TPE2']; SE=out['SE']
            force_int=out['force_int']; force_ext=out['force_ext']
            F=out['F']; u_field=out['u_field']
            loss=loss1a_function(force_int, force_ext)
            loss_list.append(float(loss))
        idx_min=np.argmin(loss_list)
        loss_min=loss_list[idx_min]
        t_start=t_start_list[idx_min]
        t_start=t_start-arg.init_step_size
        t_start=max(min(t_start, 1), 0)
        t1=time.time()
        print("t_start", t_start, "loss", loss_min, ", time cost", t1-t0)
        del out, TPE1, TPE2, SE, force_int, force_ext, F, u_field
        del t_start_list, loss, loss_list, t1, t0
#%%
Rmax_list=[]
loss1_list=[]
loss2_list=[]
TPE_list=[]
sim_t_list=[]
time_list=[]
flag_list=[]
rst_list=[]
t0=time.time()
#%%
state={}
state['idx_save']=-1
state['t_save']=[]
def save(is_final_result=False):
    if is_final_result == False:
        state['idx_save']+=1
        state['t_save'].append(t_good)
    u_field=aorta_model.get_u_field()
    Node_X=Node_x-u_field
    Mesh_X=PolyhedronMesh()
    Mesh_X.element=Element.detach().cpu()
    Mesh_X.node=Node_X.detach().cpu()
    Mesh_X.mesh_data['arg']=arg
    Mesh_X.mesh_data['t_good']=t_good
    Mesh_X.mesh_data['TPE']=TPE_list
    Mesh_X.mesh_data['loss1']=loss1_list
    Mesh_X.mesh_data['Rmax']=Rmax_list
    Mesh_X.mesh_data['rst']=rst_list
    Mesh_X.mesh_data['t']=sim_t_list
    Mesh_X.mesh_data['time']=time_list
    Mesh_X.mesh_data['t_save']=state['t_save']
    filename_save=arg.mesh_p0
    if is_final_result == False:
        filename_save+="_i"+str(state['idx_save'])
    if is_final_result == True and opt_cond == False:
        filename_save+="_error"
    if arg.save_by_vtk == True:
        Mesh_X.save_by_vtk(filename_save+".vtk")
    Mesh_X.save_by_torch(filename_save+".pt")
    print("save: t", t_good, "name", filename_save)
#%%
from FE_lbfgs import LBFGS
optimizer = LBFGS([aorta_model.state['u_field']], lr=1, line_search_fn="backtracking",
                  tolerance_grad =1e-10, tolerance_change=0, history_size=10+arg.reform_stiffness_interval, max_iter=1)
optimizer.set_backtracking(t_list=np.logspace(np.log10(arg.max_t_linesearch), np.log10(arg.min_t_linesearch), 10).tolist())
optimizer.set_backtracking(t_default=arg.default_t_linesearch, t_default_init='auto')
optimizer.set_backtracking(c=0.5, verbose=False)
#%%
from AutoStepper import AutoStepper
must_points=[]
if len(arg.must_points) > 0:
    must_points=[float(m) for m in arg.must_points.split(",")]
auto_stepper=AutoStepper(t_start=t_start, t_end=1,
                         Δt_init=arg.init_step_size,
                         Δt_min=arg.min_step_size,
                         Δt_max=arg.max_step_size,
                         alpha=2, beta=0.5,
                         max_retry=arg.max_retry,
                         random_seed=arg.random_seed,
                         t_list=must_points)
#%%
def reform_stiffness(pressure_t):
    ta=time.time()
    out=aorta_model.cal_energy_and_force(pressure_t, return_stiffness="sparse")
    H=out['H']
    H=H.astype("float64")
    #print("type(H)", type(H))
    #print('H (mean, max)', np.mean(np.abs(H.data)), np.max(np.abs(H.data)), 'nnz', H.nnz)
    optimizer.reset_state()
    optimizer.set_H0(H)
    tb=time.time()
    #print("reform stiffness done:", tb-ta)
    return tb-ta
#%%
def plot_result():
    display.clear_output(wait=False)
    fig, ax = plt.subplots()
    ax.plot(sim_t_list, np.array(loss1_list)/max(loss1_list), 'm', linewidth=0.5)
    ax.plot(sim_t_list, loss1_list, 'r-', linewidth=0.5)
    TPE_=-np.array(TPE_list)
    ax.plot(sim_t_list, TPE_/TPE_.max(),  'g')
    ax.plot(sim_t_list, -np.array(flag_list), 'b.')
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.set_title("t "+ str(auto_stepper.t)+" iter2 "+str(iter2)+" rsi " +str(reform_stiffness_interval)
                 +" time "+str(int(time_list[-1])))
    display.display(fig)
    plt.close(fig)
#%%
t_good=t_start
Ugood=aorta_model.state['u_field'].clone().detach()
#save mesh at t=0
if arg.save_all_t == True or len(must_points) !=0:
    save(False)
#%%
closure_opt={"output":"TPE", "loss1_fn":loss1a_function}
#closure_opt={"output":"loss1", "loss1_fn":loss1b_function}
max_iter1=arg.max_iter1
max_iter2=arg.max_iter2
reform_stiffness_interval=arg.reform_stiffness_interval
tab=0
tab_total=0
t_R_total=0
retry_counter=0
iter1=0
#%%
while iter1 <= max_iter1:
    iter1+=1
    auto_stepper.step()
    pressure_t=auto_stepper.t*pressure
    print("t", auto_stepper.t, ", Δt", auto_stepper.Δt, ", rsi", reform_stiffness_interval)
    opt_cond=False
    reform_stiffness_counter=0
    reform_stiffness_flag=0
    reform_stiffness_iter2=0
    fatal_error_iter2=None
    minor_error_iter2=None
    iter2=0
    while iter2 <= max_iter2:
        iter2+=1
        if iter2==1 or (iter2-1)%reform_stiffness_interval==0 or reform_stiffness_flag == 1:
            try:
                rs_time_cost=reform_stiffness(pressure_t)
            except:
                opt_cond=False
                print("break1: except at iter2 =", iter2)
                break
            tab=0.5*(tab+rs_time_cost)
            tab_total+=rs_time_cost
            rst_list.append(rs_time_cost)
            reform_stiffness_counter+=1
            reform_stiffness_flag=0
            reform_stiffness_iter2=iter2
            #print("reform_stiffness done")
        #-------------------------------------------------
        def closure(output=closure_opt["output"], loss1_fn=closure_opt["loss1_fn"]):
            out=aorta_model.cal_energy_and_force(pressure_t)
            TPE1=out['TPE1']; TPE2=out['TPE2']; SE=out['SE']
            force_int=out['force_int']; force_ext=out['force_ext']
            F=out['F']; u_field=out['u_field']
            loss1=loss1_fn(force_int, force_ext)
            loss2=cal_potential_energy(force_int-force_ext, u_field)
            if output == "TPE":
                loss=loss2
            elif output == "loss1":
                loss=loss1
            else: #output is "loss2" or "all"
                loss=loss2
            if loss.requires_grad==True:
                optimizer.zero_grad()
                loss.backward()
            if output == "TPE":
                return TPE1
            elif output == "loss1":
                return loss1
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
        #-------------------------------------------------
        try:
            U1=aorta_model.state['u_field'].clone().detach()
            opt_cond=optimizer.step(closure)
            U2=aorta_model.state['u_field'].clone().detach()
            t_linesearch=optimizer.get_linesearch_t()
            flag_linesearch=optimizer.get_linesearch_flag()
            output=torch.no_grad()(closure)(output='all', loss1_fn=loss1a_function)
            loss1, loss2, TPE, F, force_int, force_ext, force_int_of_element=output
            J=det(F)
            force_avg=(force_int_of_element**2).sum(dim=2).sqrt().mean()
            force_res=((force_int-force_ext)**2).sum(dim=1).sqrt()
            R=force_res/(force_avg+1e-10)
            Rmean=R.mean().item()
            Rmax=R.max().item()
        except:
            opt_cond=False
            print("break2: except at iter2 =", iter2)
            break
        #--------------------------------------------------------
        #check for error
        minor_error_flag=0
        fatal_error_flag=0
        if opt_cond == "nan" or opt_cond == "inf":
            opt_cond=False
            fatal_error_flag=1
            print("iter2", iter2, "error: loss is nan or inf")

        if (np.isnan(TPE) == True or np.isinf(TPE) == True # or TPE > 0
            or np.isnan(loss1) == True or np.isinf(loss1) == True
            or np.isnan(loss2) == True or np.isinf(loss2) == True):
            opt_cond=False
            fatal_error_flag=1
            print("iter2", iter2, "errorA")

        J_min=J.min().item()
        J_max=J.max().item()
        J_error=(J-1).abs().max().item()

        if J_error > 0.5:
            opt_cond=False
            fatal_error_flag=1
            print("iter2", iter2, "errorB")

        if 0.1 <= J_error <= 0.5:
            opt_cond=False
            minor_error_flag=1
            print("iter2", iter2, "errorC")

        #'''
        if TPE > 0:
            opt_cond=False
            minor_error_flag=1
            print("iter2", iter2, "errorD")
        #'''
        if fatal_error_flag == 1 or minor_error_flag == 1:
            print("  TPE", TPE, "Rmax", Rmax, "loss1", loss1, "loss2", loss2)
            print("  J_min", J_min, "J_max", J_max, "flag_linesearch", flag_linesearch, "t_linesearch", t_linesearch)

        #'''
        if fatal_error_flag == 1:
            print("break3: error at iter2 =", iter2)
            break
        else:
            pass
            #if minor_error_flag == 1:
            #    reform_stiffness_flag=1
            #t_default_init is set to 'auto'
            # set reform_stiffness_flag=1 then t in optimizer may be always t_default_init too small
            #the code below may be better
        '''
        if fatal_error_flag == 1:
            if iter2 == 1:
                print("break3: error at iter2 =", iter2)
                break
            if fatal_error_iter2 is None:
                fatal_error_iter2=iter2
                reform_stiffness_flag=1
                aorta_model.state['u_field'].data=U1
                continue
            else:
                fatal_error_iter2_old=fatal_error_iter2
                fatal_error_iter2=iter2
                if (fatal_error_iter2 - fatal_error_iter2_old) >  reform_stiffness_interval//2:
                    reform_stiffness_flag=1
                    aorta_model.state['u_field'].data=U1
                    continue
                else:
                    print("break4: error at iter2 =", iter2)
                    break
        #'''
        if minor_error_flag == 1 and fatal_error_flag == 0:
            if minor_error_iter2 is None:
                minor_error_iter2=iter2
                reform_stiffness_flag=1
            else:
                minor_error_iter2_old=minor_error_iter2
                minor_error_iter2=iter2
                if (minor_error_iter2 - minor_error_iter2_old) >  reform_stiffness_interval//2:
                    reform_stiffness_flag=1
        #'''
        #--------------------------------------------------------
        flag_list.append(flag_linesearch)
        Rmax_list.append(Rmax)
        loss1_list.append(loss1)
        loss2_list.append(loss2)
        TPE_list.append(TPE)
        t1=time.time()
        time_list.append(t1-t0)
        sim_t_list.append(auto_stepper.t)
        #--------------------------------------------------------
        #check for convergence
        if iter2==1:
            loss1_ref=loss1
            loss2_ref=abs(loss2)
        f_ratio=loss1_list[-1]/(1e-10+loss1_ref)
        E_ratio=abs(loss2_list[-1])/(1e-10+loss2_ref)

        if len(TPE_list) == 1:
            TPE_ratio=1
        else:
            TPE_ratio=abs(TPE_list[-1]-TPE_list[-2])/(1e-10+abs(TPE_list[-1]))

        U_ratio=((U2-U1)**2).sum().sqrt()/(1e-10+(U2**2).sum().sqrt())
        U_ratio=float(U_ratio)

        opt_cond=False #opt_cond may be set to True by optimizer, so set it False
        #if (U_ratio < arg.U_ratio and E_ratio < arg.E_ratio and f_ratio < arg.f_ratio
        #    and loss1 < arg.loss1 and Rmax < arg.Rmax):
        if Rmax < arg.Rmax or loss1 < arg.loss1:
            opt_cond=True

        if iter2==1 or iter2%100==0 or opt_cond is True:
            print('iter2', iter2, 'Rmax', Rmax, 'Rmean', Rmean, "force_avg", float(force_avg))
            print("  U_ratio", U_ratio, "E_ratio", E_ratio, "TPE_ratio", TPE_ratio, "f_ratio", f_ratio)
            print("  loss1", loss1, "loss2", loss2, "TPE", TPE, "max|J-1|", J_error)
            print("  flag_linesearch", flag_linesearch, "t_linesearch", t_linesearch)

        if opt_cond is True:
            print("opt_cond", opt_cond)
            print("  t", auto_stepper.t, "iter2", iter2, "pressure_t", pressure_t, "time", time_list[-1])
            print("  reform_stiffness_counter", reform_stiffness_counter, "tab", tab, "tab_total", tab_total)
            break

        #flag_linesearch < 0 very often
        #it is possible that TPE increases and loss1 (Rmax) decreases, so flag < 0
        #if flag_linesearch < 0:
        #    reform_stiffness_flag=1
    #--------------------------------------------------------
    #end of while True (iter2)
    if arg.plot == True:
        plot_result()

    if opt_cond is True:
        t_good=auto_stepper.t
        Ugood=aorta_model.state['u_field'].clone().detach()
        if retry_counter == 0:
            auto_stepper.increase_Δt()
        retry_counter=0
        if arg.save_all_t == True or t_good in must_points:
            save(False)
    else:
        print("opt_cond is False, try to adjust Δt, retry_counter", retry_counter)
        if retry_counter==0:
            auto_stepper.initialize_retry()
        retry_counter+=1
        if retry_counter > arg.max_retry:
            print("abort: retry_counter > max_retry")
            break
        print("go back to the previous t", t_good, "Δt", auto_stepper.Δt)
        auto_stepper.goback(t_good, "rand")
        aorta_model.state['u_field'].data=Ugood.clone().detach()

    if auto_stepper.t == auto_stepper.t_end and opt_cond == True:
        print("t is t_end and opt_cond is True, break")
        break
#%%
#save the final result
save(True)