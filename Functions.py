"""
Created on Sat Apr  4 17:20:13 2020

@author: Francesco Cambria & Mattia Cattaneo, Politecnico di Milano
"""



import numpy as np
from numpy.fft import ifftshift, fftn, ifftn



# BACKGROUND REMOVAL FUNCTION
'''
This function take as input a n-dimensional array and set to zero all the
elements smaller than a fixed threshold

Parameters:
    array: n-dimensional array
    threshold: value under which an element is put to zero

Return:
    back: background-free array
'''
def backfree (array, threshold):
    back = (array > threshold) * array
    # the parenthesis return 1 or 0 if array > threshold or viceversa, respectively
    return back



# NORMALIZATION FUNCTION
'''
This function take as input an n-dimensional array and normalize it to 1

Parameter:
    array: n-dimensional array to normalize
Return:
    normal: normalized array
'''
def norm (array):
    normal = array / np.sum(array)
    return normal



# ZERO-PADDING FUNCTION
'''
This function takes as input two n-dimensional arrays, one with smaller
dimesnions wrt the other. The aim is to bring the smaller array to the same
dimensions of the bigger one adding zeros. The result will be a bigger array,
with the desired dimensions of reference, where the original array is placed
in the middle of the zero-padded array (zeros are placed all around the
original array to pad)

Parameters:
    array: n-dimensional array to pad
    reference: n-dimensional array with the desired dimensions in which put array
Return:
    result: padded array
'''
def pad (array, reference):
    if array.shape == reference.shape:    # if same dimensions, do nothing
        return array
    
    result = np.zeros(reference.shape, dtype=np.float32)
    offset = np.zeros(reference.ndim, dtype=np.uint8)
    
    # definition of the offsets -> position of the void matrix (result) in which put our array
    for i in range (reference.ndim):
        offset[i] = int((reference.shape[i] - array.shape[i]) / 2)
        
    # create a list of slices from offset to offset + shape in each dimension
    insert = [slice(offset[dim], offset[dim] + array.shape[dim]) for dim in range (reference.ndim)]
    
    # insert array in result at the specified position given by offsets
    result[insert] = array
    return result



# ZERO-PADDING FUNCTION USING NUMPY.PAD (slower than the other function...)
'''
This function takes as input two 2D/3D arrays, one with smaller
dimesnions wrt the other. The aim is to bring the smaller array to the same
dimensions of the bigger one adding zeros. The result will be a bigger array,
with the desired dimensions of reference, where the original array is placed
in the middle of the zero-padded array (zeros are placed all around the
original array to pad)

Parameters:
    array: 2 or 3-dimensional array to pad
    reference: 2 or 3-dimensional array with the desired dimensions
Return:
    result: padded array
'''
def padv2 (array, reference):
    if array.shape == reference.shape:    # if same dimensions, do nothing
        return array
    
    z = reference.shape[0] - array.shape[0]
    y = reference.shape[1] - array.shape[1]
    
    '''
    If dimensions along axis have the same parity we will have a perfectly
    centered result, otherwise we will have one more element on one side.
    
    The following piece of code creates the correct tuples to pass to np.pad
    function, according to the parity of dimensions, as explained before
    '''
    if z%2 == 0:
        zoff = (int(z/2), int(z/2))
    else:
        zoff = (int(z/2)+1, int(z/2))
        
    if y%2 == 0:
        yoff = (int(y/2), int(y/2))
    else:
        yoff = (int(y/2)+1, int(y/2))
        
    if array.ndim == reference.ndim == 3:    # add the third coordinate if we have a 3D array
        x = reference.shape[2] - array.shape[2]
        
        if x%2 == 0:
            xoff = (int(x/2), int(x/2))
        else:
            xoff=(int(x/2)+1, int(x/2))
            
        result = np.pad(array, (zoff, yoff, xoff), 'constant')    # result for 3D arrays
        return result
        
    result = np.pad(array, (zoff, yoff), 'constant')    # result for 2D arrays
    return result



# FFT CONVOLUTION
'''
This function convolve two n-dimensional arrays using FFT. The two inputs must
have the same dimensions/shapes

Parameters:
    array1: first input (can be an image)
    array2: second input (can be a PSF)
Return:
    result: convoluted array
    ft1: FFT of the array1
    ft2: FFT of the array2
'''
def conv(array1, array2):
    # first step: do the Fourier transform of the two arrays
    # fftshift: it shifts the zero-frequency component to the center of the spectrum
    # ifftshift: it is used to remove phase errors
    # we can also choose if put fftshift/ifftshift or not
    ft1 = fftn(array1)
    ft2 = fftn(array2)

    # second step: do the product in the Fourier domain and the inverse transform
    product = ifftshift(ifftn(ft1 * ft2))
    result = np.abs(product)    # coming back to the image
    return result #, ft1, ft2





































