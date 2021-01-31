""" 
Label pixels from NMF representations
"""
# Author: Ping-Chen Tsai, Aug 2020 
import NMF_Sparse
import numpy as np
from PIL import Image
import scipy.io
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt 
from math import sqrt

def normalizeColumns(matrix):
    '''
    normalizes the input matrix to have the L1 norm of each column = 1
    
    Parameters
    ----------
    
    matrix: nd array, matrix to be normalized
    
    '''
#     output = np.zeros((matrix.shape))
    for i in range(matrix.shape[1]):
#         output[:,i] = np.divide(matrix[:,i], sum(abs(matrix[:,i])))
        matrix[:,i] = np.divide(matrix[:,i], sum(abs(matrix[:,i])))
    return matrix #output

def get_likelihoodMatrix(W_stk, H_stk, NumOfClasses, k_clusters, numImages, width, length):
    '''
    Get the likelihood matrix of the nmf data matrix. 
    Each pixel is a seismic structure in W. 
    The likelihood matrix Y indicates the likelihood of each pixel being that structure
    
    Parameters
    ----------
    
    W_stk: nd array, updated W with stacked result. shape = (single image dimension, components number)
    
    H_stk: nd array, updated H with stacked result. shape = (components number, number of total dataset samples)

    Returns
    ----------
    
    Y: nd-array, likelihood matrix
    '''
    Qr = np.kron(np.eye(NumOfClasses, NumOfClasses), np.ones((k_clusters,1)))
    Qr = normalizeColumns(Qr)
    Y = np.zeros((numImages, width*length, NumOfClasses))
    N_class = np.ones((1,NumOfClasses))

    for img in range(0, numImages):
        Hi = np.reshape(H_stk[:,img],(len(H_stk[:,img]),1))
        # map the coefficients of each image into the seismic structures that make up that image
        # Each Y shows the likelihood of each seismic structure for each pixel in the image.
        H_l = np.dot(Hi,N_class)
        WH_i = np.dot(W_stk, np.multiply(Qr, H_l))
    #     print('len(WH_product)',len(WH_product))
        Y[img,:, :] = WH_i
        
    return Y
def extract_coordination(Y, NumOfClasses, numImages, width, length, bkgrd_label,
                         gaussian_filtering = True, median_filtering = True, sigma = 0.45):
    '''
    Get the likelihood matrix of the nmf data matrix. 
    Each pixel is a seismic structure in W. 
    The likelihood matrix Y indicates the likelihood of each pixel being that structure
    
    Parameters
    ----------
    
    W_stk: nd array, updated W with stacked result. shape = single image dimension, components number
    
    H_stk: nd array, updated H with stacked result. shape = components number, number of total dataset samples

    Returns
    ----------
    
    classified_all: nd array, classification matrix. Shape = (total num of data, width, length)
    '''
    print('We are extracting coordination of pixel labels...')
    if gaussian_filtering:
    #     Filter each image
        for n in np.arange(numImages):
            Y_n = np.zeros((width,length))
            Y_n = Y[n, :, :]
            for label in range(NumOfClasses):
                temp = np.reshape(Y_n[: ,label].T,(width,length))
                temp = gaussian_filter(temp, sigma)
                Y[n, :,label] = np.reshape(temp,(length*width,))
            del Y_n

    # Get the maximun value of each row 
    vals= np.max(Y,axis = 2)
    #  Using vals to show which area has more confidence
    # devided by sum of each row
    conf = np.divide(vals, np.sum(Y,axis = 2)+1e-5)
    # Get the column index of maximum value. The index indicate which class the pixel falls in.
    classImg = np.argmax(Y,axis = 2)

    classified_all = np.zeros((numImages, width, length))
    if median_filtering:
        #     Filter each image
        for n in np.arange(numImages):
            img = np.reshape(classImg[n], (length, width))
            classified_all[n] = median_filter(img,size = (3,3), mode = 'reflect')
            del img

    # Turn all the low confidence pixels to bkgrd_label class
    std = np.std(conf, axis = 1)
    threshold = np.mean(conf, axis = 1)+std/2.7 # 3.1 can be tuned

    # Turn all the low confidence pixels to other class, class 5
    # Reshape confidence values into data matrix's dimension
    conf = np.reshape(conf, (numImages, length, width))
    
    # loop through all the images to get the classification mask
    for n in np.arange(numImages):
        classified_all[n][conf[n]<threshold[n]] = bkgrd_label 
        
    del classImg, threshold, conf, vals
    print('==== Finished ===')
    return classified_all

def split_labelResult(classified_all, NumOfClasses, bkgrd_label = 5, background = True):
    ''' 
    To split the classification result into individuals.

    Parameters
    ----------
    
    classified_all: nd arrays. Contains pixel labels coordinations
    
    bkgrd_label: int, labels for background. 
    
    background: bool, background flag. If the flag is False, then no background color when plotting the prediction

    Returns
    ----------
    
    classN: list of arrays. Contain class of labels coordinations
    '''
    classN = []
    for i in np.arange(NumOfClasses):
        classN.append(np.where(classified_all==i))
    if background:
        classN.append(np.where(classified_all== bkgrd_label))
    return classN

def createBinaryMask(classL, classN, numImages, width, length, numImagesPerClass, plotMask = True):
    '''
    Create binary mask for each input class

    Returns
    ----------
    
    class_mask: list of nd-arrays, Each array is a class of predicted masks
    '''
    print('We are creating binary masks for each class...')
    # create an empty mask with size to total number of image
    msk_N = np.zeros((numImages, width, length)) 
    # The saved mask will exclude any labels from other class. 
    for c in classL:
    # c = [0, 1, 2, 3] for LANDMASS
        strtrange = sum(numImagesPerClass[:c])
        endRange = sum(numImagesPerClass[:c])+ numImagesPerClass[c]
#         Uncomment following line to check the range of each class
#         print('Class %i range: '% c,strtrange, endRange)

        # Exlude any image index that is not within the range of class index, 
        # for example the class index of fault is 1000-1499, if index 10 has a fault lable, 
        # exclude coordination at index 10 from putput.
        trueIdx = np.where(np.logical_and(classN[c][0]>=strtrange, classN[c][0]<endRange))

        img_index = classN[c][0][trueIdx[0]]
        row_index = classN[c][1][trueIdx[0]]
        col_index = classN[c][2][trueIdx[0]]

        msk_N[img_index, row_index,col_index] = 1 # assign 1 to the predicted coordination
        del trueIdx, img_index, row_index, col_index # clean memory
        
    if plotMask:
        NMF_Sparse.plt_boxData(classL, numImagesPerClass, numImages, width, length, array_data = msk_N, WH_product = None)
        
    #     Split class of mask
    class_mask = []
    for c in classL:
        strtrange = sum(numImagesPerClass[:c])
        endRange = sum(numImagesPerClass[:c])+ numImagesPerClass[c]
        class_mask.append(msk_N[strtrange:endRange])
    print('=== Finished ====')
    return class_mask, msk_N

def save_toPath(maskPath, class_mask, classL, numImagesPerClass):
    '''
    Save each class masks to the paths defined by user.
    Parameters
    ----------
    
    class_mask: list of nd-array: contain classes of masks
    
    maskPath: list of nd-array. Paths defined by user.contains n paths, where n is the number of classes input to data matrx X.
    '''
    # class_mask[class_mask == 1] = 255.0 # change true label to 255
    print('Saving binary masks...')
    for n, path_ in zip(classL, maskPath):

    #     take the number of image in each class, save the mask through path
        for m in range(numImagesPerClass[n]):
    #       class_mask[0][n].shape[0]
            to_be_saved = class_mask[0][n][m]
            to_be_saved[to_be_saved == 1] = 255.0 # change true label to 255
            rescaled =to_be_saved.astype(np.uint8)
            # rescaled = (255.0 / to_be_saved.max() * (to_be_saved - to_be_saved.min())).astype(np.uint8)
            img_ = Image.fromarray(rescaled)
            img_.save(path_ + 'Mask_pred %i.png'%(m+1))
    print('=== Finished ===')

def plotLabels(classL, numImagesPerClass,numImages, width, length, classN, array_data, colorarr, plot = True):
    """
    Plot the predicted pixel-level labels
    
    Parameters
    ----------
    
    classN: list of arrays. Contain class of labels corrdinations
    
    classL: list of integers. Contain labels for each class. 
    
    array_data: nd array，original data array
    
    colorarr: list of lists. Each sublist is RGB color from 0-255
    
    plot: bool

    Returns
    ----------
    
    color_img: nd array, colorred mask of the images

    """
    temp1 = np.zeros((numImages, width, length, 3)) # 3 channels
    color_img = np.uint8(temp1)
    for i in np.arange(len(classN)):
        color_img[classN[i][0], classN[i][1], classN[i][2], :] = colorarr[i]

    if plot:
        # Plot original images with masks
        for i in classL:
            fig, axarr = plt.subplots(1, 5, figsize=(13, 13))
            for ax, n in zip(axarr.flatten(), np.arange(5)):
                if i!=0:
                    idx = sum(numImagesPerClass[:i])+n
                else:
                    idx = n
                ax.imshow(array_data[idx+5], cmap='gray')
                ax.imshow(color_img[idx+5], alpha=.4)
                ax.set_title("%i. " % (n+1) + "Class %i"  % (i+1), fontsize=13)
            plt.show()
    del fig, axarr
        
    return color_img