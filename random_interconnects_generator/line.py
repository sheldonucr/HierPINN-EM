#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:29:15 2019

@author: ssadiqbatcha
"""

import random
import numpy as np
import scipy.io as sio


# Filenames (Code will generate end-start number of random designs)
start = 25
end = 25

# Destination directory (designs will be stored here)
destDir = 'data/'

# Design range
J_range = [-5e9, -2e9, -1e9, -5e8, 0, 5e8, 1e9, 2e9, 5e9]
L_range = [10, 20, 30, 40]
Seg_range = [ 10]

width = 1

# Simulation parameters
plots = 0
T = 353.0
tmax = 1e6     # 10 years
tstep = 1e4     # 1 year


def genDesign(filename:int):
    
    # Add header to the geo file
    design = addHeader()
    nWires = random.choice(Seg_range)
    
    vertex = [0, 0]
    for i in range(nWires):
        geo = [random.choice(L_range), width]
        design.append('Rectangle({0}) = {{{1}, {2}, 0, {3}, {4}, 0}};'.format(i+1,vertex[0],vertex[1],geo[0],geo[1]))
        design.append('//+')
        vertex = [vertex[0]+geo[0],0]

    saveDesign(design,nWires,filename)
    return design

def addHeader():
    design = []
    design.append('//+')
    design.append('SetFactory("OpenCASCADE");')
    return design

def saveDesign(design,nWires,filename):
    # Generate J vector
    J = [random.choice(J_range) for _ in range(0,nWires)]
    # Save
    np.savetxt(destDir + str(filename) + '.geo', design, delimiter = '',fmt='%s')
    sio.savemat(destDir + str(filename) + '.mat', {'J':J, 'T':T, 'tmax':tmax, 'tstep':tstep, 'plots':plots})

if __name__ == "__main__":
    for filename in range(start,end+1):
        design = genDesign(filename)
        #np.savetxt(destDir + str(filename) + '.geo', design, delimiter = '')    
