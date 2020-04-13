"""
Created on Sun Apr 12 17:16:05 2020

@author: Francesco Cambria & Mattia Cattaneo, Politecnico di Milano
"""



from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import signal

from Functions import backfree, norm, pad, padv2, conv
from Class import Stack



# Parameters definition
imname = 'membrane_substack.tif'    # name of the image stack to import
PSFname = 'psf_rescaled.tif'    # name of the PSF stack to import
mode = 'stack'    # select 'stack' if you want to pass to the class a stack;
                  # select 'file' if you want to pass to the class a path where open a stack
THRESHOLD=12000    # threshold for the background removal function



# Import image and PSF stacks from files and convert them into 3D arrays
im = io.imread(imname)
imarray = np.array(im, dtype=np.float32)
print("\nDimension of the image stack: ", imarray.shape)
print("Maximum value of the image stack: ", np.amax(imarray))
PSF = io.imread(PSFname)
PSFarray = np.array(PSF, dtype=np.float32)
print("\nDimension of the PSF stack: ", PSFarray.shape)
print("Maximum value of the PSF stack: ", np.amax(PSFarray))



# Define the middle plane of the stacks
IMG_PLANE = Stack(mode, imarray).middleplane()
PSF_PLANE = Stack(mode, PSFarray).middleplane()
# Parameters for the final visualization -> select the image to visualize
imdepth = IMG_PLANE    # change this parameter if you want to visualize an image that is not the middle one
PSFdepth = PSF_PLANE    # change this parameter if you want to visualize an image that is not the middle one



# Background removal
# all the elements smaller than THRESHOLD are setted to zero
backPSF = backfree(PSFarray, THRESHOLD)



# Normalization of the PSF
normPSF=norm(backPSF)    # volume normalized to 1
#print("\nMaximum value of the normalized PSF stack: ", np.amax(normPSF))



# Zero-padding of the PSF stack
start = time.time()
padPSF = pad(normPSF, imarray)
end = time.time()
print("\nTime padding: ", (end - start))

start = time.time()
padv2PSF = padv2(normPSF, imarray)
end = time.time()
print("Time padding with numpy.pad function: ", (end - start))

print("\nDimension of the zero padded normalized PSF stack: ", padPSF.shape)
print("Maximum value of the zero padded normalized PSF stack: ", np.amax(padPSF))
#print("Index position of the maximum value: ", np.unravel_index(np.argmax(padPSF), padPSF.shape), "\n")
print("\nDimension of the zero padded normalized PSF stack with numpy.pad: ", padv2PSF.shape)
print("Maximum value of the zero padded normalized PSF stack with numpy.pad: ", np.amax(padv2PSF))
#print("Index position of the maximum value: ", np.unravel_index(np.argmax(padPSF), padPSF.shape), "\n")



# CONVOLUTION
start = time.time()    # start time; we can put it also at the beginning of the code
myconv = conv(imarray, padPSF)
# the function returns the convoluted image and the Fourier transforms of the two arrays in input
end = time.time()    # end time

print("\nTime required for the convolution operation: ", (end - start), "seconds\n")	
print("Difference between sum elements image array and output array: ", (np.sum(imarray) - np.sum(myconv)))
print("Maximum value of the output stack: ", np.amax(myconv))
#print("Index position of the maximum value: ", np.unravel_index(np.argmax(im_back), im_back.shape))



# scipy.signal.fftconvolve
start = time.time()    # start time
sciconv = signal.fftconvolve(imarray, padPSF, mode='same')
# it works similarly putting normPSF instead of padPSF: it runs faster even if there are small differences
end = time.time()    # end time

print("\nTime required for the scipy convolution operation: ", (end - start), "seconds\n")	
print("Difference between sum image array and sum scipy array: ", (np.sum(imarray) - np.sum(sciconv)))
#print("Difference between sum output array and scipy array: ", (np.sum(im_back) - np.sum(out)))
print("Maximum value of the scipy stack: ", np.amax(sciconv))
#print("Index position of the maximum value: ", np.unravel_index(np.argmax(out), out.shape))



# Difference between my convolution and scipy convolution
diff = myconv - sciconv

print("\nMaximum value of the difference stack: ", np.amax(diff))
print("Index position of the maximum value: ", np.unravel_index(np.argmax(diff), diff.shape))
print("Minimum value of the difference stack: ", np.amin(diff))
print("Index position of the minimum value: ", np.unravel_index(np.argmin(diff), diff.shape))



# Save stacks
# the second passed parameter is used to choose if save in uint16 or float32 data type
'''
filename = 'background-free_PSF'
Stack(mode, backPSF).savestack(filename, 16)
filename = 'normalized_PSF'
Stack(mode, normPSF).savestack(filename, 32)
filename = 'zero_padded_normalized_PSF'
Stack(mode, padPSF).savestack(filename, 32)
'''
filename = 'my_fftconvolve'
Stack(mode, myconv).savestack(filename, 16)
filename = 'scipy.signal.fftconvolve'
Stack(mode, sciconv).savestack(filename, 16)
filename = 'conv_difference'
Stack(mode, diff).savestack(filename, 32)



# Plot the figures
A = Stack(mode, imarray).imagemode(imdepth)
B = Stack(mode, PSFarray).imagemode(PSFdepth)
C = Stack(mode, normPSF).imagemode(PSFdepth)
D = Stack(mode, padPSF).imagemode(imdepth)
E = Stack(mode, myconv).imagemode(imdepth)
F = Stack(mode, sciconv).imagemode(imdepth)

fig=plt.figure(figsize=(11, 7))
plt.subplot(231), plt.imshow(np.abs(A), "gray"), plt.title("Original image")
plt.subplot(232), plt.imshow(np.abs(B), "gray"), plt.title("Original PSF")
plt.subplot(233), plt.imshow(np.abs(C), "gray"), plt.title("Normalized background-free PSF")
plt.subplot(234), plt.imshow(np.abs(D), 'gray'), plt.title("Zero-padded normalized PSF")
plt.subplot(235), plt.imshow(np.abs(E), 'gray'), plt.title("Convolved image")
plt.subplot(236), plt.imshow(np.abs(F), 'gray'), plt.title("scipy.signal.fftconvolve")
plt.show()

















