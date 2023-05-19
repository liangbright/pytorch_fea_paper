# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 22:29:00 2023

@author: liang
"""
import numpy as np

def read_abaqus_inp(filename, re_label_node=True):
    node_id=[]
    node=[]
    element_id=[]
    element=[]
    with open(filename, 'r') as file:
        inp=file.readlines()
    #-----------
    lineindex=-1
    for k in range(0, len(inp)):
        if "*NODE" in inp[k] or "*Node" in inp[k]:
            lineindex=k
            break
    if lineindex < 0:
        print('NODE/Node keyword is not found')
        return node_id, node, element_id, element
    k=lineindex
    while True:
        k=k+1
        if "*" in inp[k]:
            lineindex=k
            break
        temp=inp[k].replace(" ", "")
        temp=temp.split(",")
        node_id.append(int(temp[0]))
        node.append([float(temp[1]), float(temp[2]), float(temp[3])])
    #-----------
    lineindexlist=[]
    for k in range(0, len(inp)):
        if "*ELEMENT" in inp[k] or "*Element" in inp[k]:
            lineindexlist.append(k)
    if len(lineindexlist) == 0:
        print('ELEMENT/Element keyword is not found')
        return node_id, node, element_id, element
    for lineindex in lineindexlist:
        k=lineindex
        while True:
            k=k+1
            if "*" in inp[k]:
                break
            temp=inp[k].replace(" ", "")
            temp=temp.split(",")
            temp=[int(a) for a in temp]
            element_id.append(int(temp[0]))
            element.append(temp[1:])
    node_id=np.array(node_id)
    node=np.array(node)
    element_id=np.array(element_id)
    element=np.array(element, dtype='object')
    if re_label_node == True:
        #sort node by id
        idx_sorted=np.argsort(node_id)
        node_id=node_id[idx_sorted]
        node=node[idx_sorted]
        #sort element by id
        idx_sorted=np.argsort(element_id)
        element_id=node_id[idx_sorted]
        element=element[idx_sorted]
        #map old id to new id
        map={}
        for n in range(0, len(node_id)):
            map[node_id[n]]=n
        for m in range(0, len(element_id)):
            for n in range(0, len(element[m])):
                element[m][n]=map[element[m][n]]
    return node, element, node_id, element_id