# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 14:18:50 2018
Uses more variables so can see image progression
Uses minimum_distance method.
@author: ctod
"""
from scipy import misc, signal as sig, ndimage as ndi
from skimage import (exposure as exp, morphology as mp,
                     filters as fil, feature as feat)
import matplotlib.pyplot as plt
import numpy as np
import math

def overlay(im1, im2):
    plt.figure()
    plt.imshow(im1, cmap='gray', interpolation='none')
    plt.imshow(im2, cmap='jet', alpha=0.3, interpolation='none')
    # plot image1 and overlay with image2
    plt.show()

def plot(im):
    plt.figure()
    plt.imshow(im)

smlst = 0
m_d = 1
d = 25
#
i = np.load('shot_psf_5.npy')
# i = misc.imread(image, 'True')
i = i.astype(int)
adapt = exp.equalize_adapthist(i)
# apply adaptive histogram to improve contrast
adapt = adapt*255  # increase scale from 0-1 to 0-255
fltr = sig.wiener(adapt)  # apply wiener filter to remove noise
fltr = fltr.astype(int)  # array converted to float, must convert back
thresh = fil.threshold_otsu(fltr)
# use Otsu's method to find suitable threshhold value
binary = fltr > thresh  # create a binary image using said threshold
fil = ndi.binary_fill_holes(binary)  # fills holes in image
ope = ndi.binary_opening(fil)  # erosion followed by dialation
aro = mp.remove_small_objects(ope, min_size=smlst)
# remove small connected objects
#
maxs = feat.peak_local_max(i, indices=False,  # footprint=np.ones((win, win))
                           min_distance=m_d, labels=aro)
# finds local maxima according to given arguements
maxs = ndi.label(maxs)[0]   # labels maxs
# w = mp.watershed(-i, maxs, mask=aro)
# perform watershed algorithm on inverse of original, filling from maxs,
# filling only where aropen1 is non-zero
n = maxs.max()  # number of spots = max value or number of labels
# plt.imshow(w)
# print(n)
r0 = (d/2)*(math.sqrt(1/n))  # Fried parameter calculation
# print(r0)
# ns = 'N = ' + str(n)
# r0s = '/ r0 = ' + str(r0)
print('n = ' + str(n) + '; r0 = ' + str(r0))
