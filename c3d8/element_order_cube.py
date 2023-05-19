# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 21:43:46 2021

@author: liang
"""
#%%
import torch
a=torch.zeros((3, 4, 5, 6))
id=-1
for z in range(0,3):
    for y in range(0, 4):
        for x in range(0, 5):
            for c in range(0, 6):
                id+=1
                a[z,y,x,c]=id
print(a.reshape(-1))
#%%
Nx=3
Ny=4
Nz=5
grid=torch.zeros((Nx*Ny*Nz, 3), dtype=torch.float32)
element=torch.zeros(((Nx-1)*(Ny-1)*(Nz-1), 8), dtype=torch.int64)
map=torch.zeros((Nz, Ny, Nx), dtype=torch.int64)
id=-1
for z in range(0,Nz):
    for y in range(0, Ny):
        for x in range(0, Nx):
            id+=1
            map[z,y,x]=id
            grid[id,0]=x
            grid[id,1]=y
            grid[id,2]=z
print('grid done')
id=-1
for z in range(0,Nz-1):  
    for y in range(0, Ny-1):
        for x in range(0, Nx-1):
            id+=1
            element[id,0]=map[z,y,x]
            element[id,1]=map[z,y,x+1]
            element[id,2]=map[z,y+1,x+1]
            element[id,3]=map[z,y+1,x]
            element[id,4]=map[z+1,y,x]
            element[id,5]=map[z+1,y,x+1]
            element[id,6]=map[z+1,y+1,x+1]
            element[id,7]=map[z+1,y+1,x]
print('element done')
grid_image=grid.view(Nz, Ny, Nx, 3)
element_image=element.view(Nz-1, Ny-1, Nx-1, 8)
#%%
element_x=grid[element.view(-1), :]
element_x=element_x.view(-1,8,3)
#%%

