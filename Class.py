"""
Created on Sun Apr 12 17:16:29 2020

@author: Francesco Cambria & Mattia Cattaneo, Politecnico di Milano
"""



import numpy as np
from skimage import io
from skimage.external import tifffile as tif



# CREATION OF THE CLASS
'''
Create a class that allows to open, visualize and save a stack
'''
class Stack(object):
    '''
    The parameter 'mode' tells if the parameter 'file' is a file path where
    open the stack or if it is already an imported stack
    '''
    def __init__(self, mode, file):
        if mode == 'file':
            self.stck = np.float32(io.imread(file))
        elif mode == 'stack':
            self.stck = file
        else:
            raise TypeError('Wrong mode selected: please type file or stack in mode')
            
    # it generates the middle position of a stack        
    def middleplane(self):
        st=self.stck
        middle=int((st.shape[0])/2)
        return middle

    # it takes a specific image of the stack
    def imagemode(self, depth):
        st = self.stck
        if (depth) > (np.shape(st))[0]:
            raise TypeError('There are not enough images! Select a number from zero to ' + str((np.shape(st))[0]))
        im = st[depth, :, :]
        return im

    # it takes a specific section of the stack
    def sectionmode(self, depth):
        st = self.stck
        if depth > (np.shape(st))[2]:
            raise TypeError('There are not enough sections! Select a number from 0 to ' + str((np.shape(st))[2]))
        se = st[:, :, depth]
        return se

    # it save the passed element in uint16 or float32 data type
    def savestack(self, filename, bits):
        if bits == 16:
            data = np.uint16(self.stck)
        elif bits == 32:
            data = np.float32(self.stck)
        else:
            raise TypeError('Only uint16 or float 32 dataType are allowed. Parameter must be 16 or 32 respectively')
        tif.imsave(filename + '.tif', data)
        return











