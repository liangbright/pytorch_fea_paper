import os
import torch
#%% #['24', '150', '168', '171', '174', '192', '318']
p0_list=['p0_24_solid_matMean_p20_i10_inverse_p0_NN',
         'p0_150_solid_matMean_p20_i10_inverse_p0_NN',
         'p0_168_solid_matMean_p20_i10_inverse_p0_NN',
         'p0_171_solid_matMean_p20_i10_inverse_p0_NN',
         'p0_174_solid_matMean_p20_i10_inverse_p0_NN',
         'p0_192_solid_matMean_p20_i10_inverse_p0_NN',
         'p0_318_solid_matMean_p20_i10_inverse_p0_NN'
         ]
#%%
pressure=10
folder_result='../../../MLFEA/TAA/result/inverse_p0_autoencoder_disp/Encoder(128,16,1)_Decoder(128,16,1)/0.8/'
shell_template="../../../MLFEA/TAA/data/bav17_AortaModel_P0_best"
mat=torch.load('../../../MLFEA/TAA/data/343c1.5_fast/125mat.pt')['mean_mat_str']
for p0_name in p0_list:
    mesh_p0=folder_result+p0_name
    mesh_px=mesh_p0+'_to_p'+str(pressure)
    cmd=("python aorta_FEA_C3D8_SRI_quasi_newton_R1.py"
         +" --cuda 1"
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