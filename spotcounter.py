# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 14:18:50 2018

@author: ctod
"""


def spotcounter(image, smlst_size=2, pk_win=10, d=25):
    """for counting the number of speckles in an image,
    to assist in the calculation of the Fried parameter. 'image' is the image
    to be operated on, 'smlst_size' defines threshold at which objects of this
    size and smaller will be removed, 'pk_win' defines the size of the window
    within which peaks will be searched for, and 'd' defines size of aperature
    of camera/telescope"""
    # necessary imports
    from scipy import misc, signal as sig, ndimage as ndi
    from skimage import (exposure as exp, morphology as mp,
                         filters as fil, feature as feat)
    # import matplotlib.pyplot as plt
    import numpy as np
    import math
    #
    i = image
    # i = misc.imread(image, 'True')
    i = i.astype(int)
    i = exp.equalize_adapthist(i)
    # apply adaptive histogram to improve contrast
    i = i*255  # increase scale from 0-1 to 0-255
    i = sig.wiener(i)  # apply wiener filter to remove noise
    i = i.astype(int)  # array converted to float, must convert back
    t = fil.threshold_otsu(i)
    # use Otsu's method to find suitable threshhold value
    b = i > t  # create a binary image using said threshold
    b = ndi.binary_fill_holes(b)  # fills holes in image
    b = ndi.binary_opening(b)  # erosion followed by dialation
    b = mp.remove_small_objects(b, min_size=smlst_size)
    # remove small connected objects
    #
    m = feat.peak_local_max(i, indices=False,
                            footprint=np.ones((pk_win, pk_win)), labels=b)
    # finds local maxima according to given arguements
    m = ndi.label(m)[0]   # labels maxs
    w = mp.watershed(-i, m, mask=b)
    # perform watershed algorithm on inverse of original, filling from maxs,
    # filling only where aropen1 is non-zero
    n = w.max()  # number of spots = max value or number of labels
    # plt.imshow(w)
    # print(n)
    r0 = (d/2)*(math.sqrt(1/n))  # Fried parameter calculation
    # print(r0)
    # ns = 'N = ' + str(n)
    # r0s = '/ r0 = ' + str(r0)
    return(w, n, r0)
