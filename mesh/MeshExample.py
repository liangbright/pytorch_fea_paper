import torch
import numpy as np
from QuadMesh import QuadMesh
from HexahedronMesh import HexahedronMesh as HexMesh
#%%
def create_quad_cylinder_mesh(n_rings, n_points_per_ring, dtype=torch.float32, device=torch.device("cpu")):
    theta=2*np.pi/n_points_per_ring
    node=np.zeros((n_rings*n_points_per_ring, 3))
    k=-1
    for n in range(0, n_rings):
        for m in range(0, n_points_per_ring):
            x=np.cos(theta*m)
            y=np.sin(theta*m)
            z=n/n_rings
            k=k+1
            node[k,0]=x
            node[k,1]=y
            node[k,2]=z
    element=[]
    for n in range(1, n_rings):
        idxA=np.arange((n-1)*n_points_per_ring, n*n_points_per_ring)
        idxB=np.arange(n*n_points_per_ring, (n+1)*n_points_per_ring)
        for i in range(0, n_points_per_ring-1):
            element.append([idxA[i], idxA[i+1], idxB[i+1], idxB[i]])
        element.append([idxA[n_points_per_ring-1], idxA[0], idxB[0], idxB[n_points_per_ring-1]])
    cylinder=QuadMesh()
    cylinder.node=torch.tensor(node, dtype=dtype, device=device)
    cylinder.element=torch.tensor(element, dtype=torch.int64, device=device)
    return cylinder
#%%
def create_quad_grid_mesh(Nx, Ny, dtype=torch.float32, device=torch.device("cpu")):
    element=torch.zeros(((Nx-1)*(Ny-1), 4), dtype=torch.int64)
    grid=torch.zeros((Nx*Ny, 3), dtype=dtype)
    map=torch.zeros((Ny, Nx), dtype=torch.int64)
    id=-1
    boundary=[]
    for y in range(0, Ny):
        for x in range(0, Nx):
            id+=1
            map[y,x]=id
            grid[id,0]=x
            grid[id,1]=y
            if y==0 or y==Ny-1 or x==0 or x==Nx-1:
                boundary.append(id)
    boundary=torch.tensor(boundary, dtype=torch.int64)
    id=-1
    for y in range(0, Ny-1):
        for x in range(0, Nx-1):
            id+=1
            element[id,0]=map[y,x]
            element[id,1]=map[y,x+1]
            element[id,2]=map[y+1,x+1]
            element[id,3]=map[y+1,x]
    grid_mesh=QuadMesh()
    grid_mesh.node=grid.to(device)
    grid_mesh.element=element.to(device)
    grid_mesh.node_set['boundary']=boundary.to(device)
    return grid_mesh
#%%
def create_hex_grid_mesh(Nx, Ny, Nz, dtype=torch.float32, device=torch.device("cpu")):
    element=torch.zeros(((Nx-1)*(Ny-1)*(Nz-1), 8), dtype=torch.int64)
    grid=torch.zeros((Nx*Ny*Nz, 3), dtype=dtype)
    map=torch.zeros((Nz, Ny, Nx), dtype=torch.int64)
    id=-1
    boundary=[]
    for z in range(0,Nz):
        for y in range(0, Ny):
            for x in range(0, Nx):
                id+=1
                map[z,y,x]=id
                grid[id,0]=x
                grid[id,1]=y
                grid[id,2]=z
                if z==0 or z==Nz-1 or y==0 or y==Ny-1 or x==0 or x==Nx-1:
                    boundary.append(id)
    boundary=torch.tensor(boundary, dtype=torch.int64)
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
    grid_mesh=HexMesh()
    grid_mesh.node=grid.to(device)
    grid_mesh.element=element.to(device)
    grid_mesh.node_set['boundary']=boundary.to(device)
    return grid_mesh
#%%
if __name__ == '__main__':
    mesh0=create_quad_grid_mesh(10,20)
    mesh0.save_by_vtk("D:/MLFEA/TAA/mesh/quad_grid_mesh_x10y20.vtk")

    mesh1=create_hex_grid_mesh(10,20,2)
    mesh1.save_by_vtk("D:/MLFEA/TAA/mesh/hex_grid_mesh_x10y20z2.vtk")