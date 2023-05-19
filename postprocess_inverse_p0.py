import os
import torch
#%%
p0_list=['p0_24_solid_matMean_p20_i10_BD_BD0A20B0.5a_p0_t20',
         'p0_24_solid_matMean_p20_i10_inverse_p0']
#%%
p0_list=['p0_150_solid_matMean_p20_i10_BD_BD0A20B0.5a_p0_t20',
         'p0_150_solid_matMean_p20_i10_inverse_p0'
         ]
#%%
p0_list=['p0_168_solid_matMean_p20_i10_BD_BD0A20B0.5a_p0_t20',
         'p0_168_solid_matMean_p20_i10_inverse_p0'
         ]
#%%
p0_list=['p0_171_solid_matMean_p20_i10_BD_BD0A20B0.5a_p0_t20',
         'p0_171_solid_matMean_p20_i10_inverse_p0'
         ]
#%%
p0_list=['p0_174_solid_matMean_p20_i10_BD_BD0A20B0.5a_p0_t20',
         'p0_174_solid_matMean_p20_i10_inverse_p0_refine'
         ]
#%%
p0_list=['p0_192_solid_matMean_p20_i10_BD_BD0A20B0.5a_p0_t20',
         'p0_192_solid_matMean_p20_i10_inverse_p0_refine'
         ]
#%%
p0_list=['p0_318_solid_matMean_p20_i10_BD_BD0A20B0.5a_p0_t20',
         'p0_318_solid_matMean_p20_i10_inverse_p0_refine'
         ]
#%%
pressure=10
folder_result='./app3/result/' # folder_result += 'BD' or 'PyFEA_P0'
shell_template="./app3/data/bav17_AortaModel_P0_best"
mat=torch.load('./app3/data/343c1.5/125mat.pt')['mean_mat_str']
for p0_name in p0_list:
    mesh_p0=folder_result+p0_name
    mesh_px=mesh_p0+'_to_p'+str(pressure)
    cmd=("python aorta_FEA_C3D8_SRI_quasi_newton_R1.py"
         +" --cuda 0"
         +" --pressure "+str(pressure)
         +' --mat ' + '"' + mat +'"'
         +" --mesh_p0 "+mesh_p0
         +" --mesh_px "+mesh_px
         +' --random_seed 0'
         +' --save_by_vtk True'
         +' --shell_template '+shell_template
         +' --n_layers 1'
         +' --plot False'
         +' --loss1 0.01'
         +' --Rmax 0.005'
         )
    print(cmd)
    os.system(cmd)
    #break
#