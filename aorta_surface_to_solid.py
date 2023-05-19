# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 13:10:29 2022

@author: liang
"""

import sys
sys.path.append("mesh")
import numpy as np
import torch
from torch_scatter import scatter
from PolygonMesh import PolygonMesh
from TriangleMesh import TriangleMesh
from PolyhedronMesh import PolyhedronMesh
from copy import deepcopy
#%%
@torch.no_grad()
def offset_surface(surface, offset, n_iter, n_smooth):
    #surface is TriangleMesh
    #offset.shape (1, )  or (n_nodes, )
    surface.build_adj_node_link()
    adj_node_link=surface.adj_node_link
    offset=offset/n_iter
    for k in range(0, n_iter):
        surface.update_node_normal()
        node_normal=surface.node_normal # (n_nodes,3)
        normal=node_normal
        X0=surface.node[adj_node_link[:,0]]
        X1=surface.node[adj_node_link[:,1]]
        distance=((X0-X1)**2).sum(dim=1, keepdim=True).sqrt()+1e-8
        weight=1/distance
        for n in range(0, n_smooth):
            normal=normal[adj_node_link[:,0]]
            normal=scatter(weight*normal, adj_node_link[:,1], dim=0, dim_size=node_normal.shape[0], reduce="mean")
            normal=normal/torch.norm(normal, p=2, dim=1, keepdim=True)
        #print(normal.shape, offset.shape)
        surface.node+=normal*offset
#%%
@torch.no_grad()
def offset_surface_list(surface_init, thickness, n_iter, n_smooth):
    #surface is TriangleMesh
    #thickness.shape (1, n_layers)  or (n_nodes, n_layers)
    n_layers=thickness.shape[1]
    surface_list=[surface_init]
    element=surface_init.element
    for n in range(1, n_layers+1):
        #print('n', n)
        surface=TriangleMesh()
        surface.node=surface_list[n-1].node.clone()
        surface.element=element.clone()
        offset=thickness[:,(n-1):n]
        offset_surface(surface, offset, n_iter, n_smooth)
        surface_list.append(surface)
    return surface_list
#%%
@torch.no_grad()
def make_solid(surface_list, element_ref):
    if isinstance(element_ref, torch.Tensor):
        element_ref=element_ref.detach().cpu().numpy()
    node=[]
    for n in range(0, len(surface_list)):
        node.append(surface_list[n].node)
    node=torch.cat(node, dim=0)
    element=[]
    for n in range(1, len(surface_list)):
        for m in range(0, len(element_ref)):
            elm_a=list(element_ref[m])
            elm_b=list(element_ref[m])
            N=len(surface_list[0].node)
            a=(n-1)*N
            b=n*N
            for k in range(0, len(elm_a)):
                elm_a[k]+=a
            for k in range(0, len(elm_b)):
                elm_b[k]+=b
            elm=elm_a+elm_b
            element.append(elm)
    solid=PolyhedronMesh()
    solid.node=node
    solid.element=element
    return solid
#%%
@torch.no_grad()
def aorta_surface_to_solid(surface, direction, thickness, n_layers, n_iter=1, n_smooth=1):
    surface_init=TriangleMesh()
    surface_init.node=surface.node.clone()
    surface_init.element=deepcopy(surface.element)
    element_ref=deepcopy(surface.element)
    surface_init.quad_to_tri()
    thickness=direction*thickness*torch.ones((surface.node.shape[0], n_layers))/n_layers
    surface_list=offset_surface_list(surface_init, thickness, n_iter, n_smooth)
    if direction < 0:
        surface_list=surface_list[-1::-1]
    #surface_inner=surface_list[0]
    #surface_outer=surface_list[-1]
    solid=make_solid(surface_list, element_ref)
    for n in range(0, len(surface_list)):
        surface_list[n].element=element_ref
    return solid, surface_list
#%%
