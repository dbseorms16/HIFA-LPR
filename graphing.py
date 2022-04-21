# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:59:00 2020

@author: user
"""

import os
import PSNR
import numpy as np
import matplotlib.pyplot as plt


def graph(X=[], Y=[], savefolder ='./', typeof='PSNR', Xlab='epoch'):
    plt.figure()
    
    plt.title('graph')
    
    coloring = ''
    if(typeof=='PSNR'):
        coloring = 'blue'
    else:
        coloring = 'red'
   
    plt.xlabel(Xlab)
    plt.ylabel(typeof)
    plt.plot(Y,label='current '+typeof, color='black', linestyle='dashed', marker='o', 
             markersize =3, markerfacecolor='black')
    
    plt.plot(X, label='previous '+typeof, color=coloring, linestyle='dashed', marker='o', 
             markersize =3, markerfacecolor=coloring)
     
    plt.savefig(savefolder+typeof+'.jpg')
    plt.close()
