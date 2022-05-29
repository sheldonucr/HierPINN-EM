#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 06 23:24:15 2022

@author: wentian
"""

import random
import numpy as np
import scipy.io as sio


# Filenames (Code will generate end-start number of random designs)
start = 0
end = 200

# Destination directory (designs will be stored here)
destDir = 'data/'

# Design range
J_range = [-5e9, -2e9, -1e9, -5e8, 0, 5e8, 1e9, 2e9, 5e9]
W_range = [1]
L_range = [10, 20, 30, 40]

# Canvas size 256x256
boundry = 256

# Max wire numbers
# nWiresMax_end = 105
nWiresMax_start = 10
nWiresMax = nWiresMax_start

# Simulation parameters
plots = 0
T = 353.0
tmax = 1e6      # 10 years
tstep = 1e4     # 1 year


def genDesign(filename:int):
    # Start at index (0,0)
    # terminalsQueue = [(0,0,1,1)]
    # Start at index (w/2,w/2)
    terminalsQueue = [(max(W_range)/2,max(W_range)/2,1,1)]
    nWires = 0
    
    # Conflict prevents wire overlap
    conflict = np.array([[0 for _ in range(0,2*boundry)] for _ in range(0, 2*boundry)])
    
    # Branches to enfore 2 branch max per vertex
    branches = {}
    
    # Add header to the geo file
    design = addHeader()
    while(terminalsQueue != [] and nWires < nWiresMax):            
        design, conflict, branches, newTerminals, nWires = addWire(design,nWires,branches,terminalsQueue.pop(0),conflict)
        terminalsQueue += newTerminals
    
    saveDesign(design,nWires,filename)
    return design

def addHeader():
    design = []
    design.append('//+')
    design.append('SetFactory("OpenCASCADE");')
    return design

def addWire(design,nWires,branches,vertex,conflict):
    # Initialize new terminals, if new wire is not added then return empty list
    newTerminals = []
    
    # Randomly choose if the new wire shoud be vertical or horizontal
    vertical = random.choice([True,False])
    if(vertical):
        # Vertical
        geom = (vertex[2]*random.choice(W_range),vertex[3]*random.choice(L_range))
        start_point = (vertex[0]-geom[0]/2, vertex[1])
    else:
        # Horizontal
        geom = (vertex[2]*random.choice(L_range),vertex[3]*random.choice(W_range))
        start_point = (vertex[0], vertex[1]-geom[1]/2)
    
    
    # Bounds check all terminals of the new wire
    if(start_point[0]+geom[0] < 0 or start_point[0]+geom[0] > boundry or start_point[1]+geom[1] < 0 or start_point[1]+geom[1] > boundry):
        # Bounds breached, dont add wire
        pass
    else:
        # Conflict check
        conflictLcl = np.copy(conflict);
        # Overlap
        conflictLcl[int(2*min(start_point[0],start_point[0]+geom[0])):int(2*max(start_point[0],start_point[0]+geom[0])),int(2*min(start_point[1],start_point[1]+geom[1])):int(2*max(start_point[1],start_point[1]+geom[1]))] += 1
        # the area around the vertex point is allowed to have overlap
        conflictLcl[int(2*(vertex[0]-max(W_range)/2)):int(2*(vertex[0]+max(W_range)/2)), int(2*(vertex[1]-max(W_range)/2)):int(2*(vertex[1]+max(W_range)/2))] = \
            np.clip(conflictLcl[int(2*(vertex[0]-max(W_range)/2)):int(2*(vertex[0]+max(W_range)/2)), int(2*(vertex[1]-max(W_range)/2)):int(2*(vertex[1]+max(W_range)/2))], 0, 1)
       
        if(np.amax(conflictLcl) > 1):
            # Overlap detected, dont add wire
            pass
        elif((vertex[0],vertex[1]) in branches and branches[(vertex[0],vertex[1])] > 1):
            # Edge contact detected, dont add wire
            pass
        else:
            # No overlap, okay to add the new wire
            nWires += 1
            if((vertex[0],vertex[1]) in branches):
                branches[(vertex[0],vertex[1])] += 1
            else:
                branches[(vertex[0],vertex[1])] = 1
            
            design.append('Rectangle({0}) = {{{1}, {2}, 0, {3}, {4}, 0}};'.format(nWires,start_point[0],start_point[1],geom[0],geom[1]))
            design.append('//+')
            # Add the four vertices of the new wire to newTerminals
            #newTerminals += [vertex,(vertex[0],vertex[1]+geom[1]),(vertex[0]+geom[0],vertex[1]),(vertex[0]+geom[0],vertex[1]+geom[1])]
            if(vertical):
                newTerminals += [(vertex[0],vertex[1],vertex[2],vertex[3]*-1),(vertex[0],vertex[1]+geom[1],vertex[2],vertex[3])]
            else:
                newTerminals += [(vertex[0],vertex[1],vertex[2]*-1,vertex[3]),(vertex[0]+geom[0],vertex[1],vertex[2],vertex[3])]
            # Update the conflict matrix
            conflict = np.copy(conflictLcl)
    
    return design, conflict, branches, newTerminals, nWires


def saveDesign(design,nWires,filename):
    # Generate J vector
    J = [random.choice(J_range) for _ in range(0,nWires)]
    # Save
    np.savetxt(destDir + str(filename) + '.geo', design, delimiter = '',fmt='%s')
    sio.savemat(destDir + str(filename) + '.mat', {'J':J, 'T':T, 'tmax':tmax, 'tstep':tstep, 'plots':plots})

if __name__ == "__main__":

    for filename in range(start,end+1):
        if(nWiresMax < 105):
            nWiresMax = nWiresMax_start + filename//2
        design = genDesign(filename)
        #np.savetxt(destDir + str(filename) + '.geo', design, delimiter = '')
        # print(design[-2])    
