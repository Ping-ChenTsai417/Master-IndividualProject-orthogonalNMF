""" 
Validate Accuracy of mask predictions.
"""
# Author: Ping-Chen Tsai, Aug 2020 

import NMF_Sparse
import label_Pixel
import numpy as np
import imageio
from PIL import Image
from math import sqrt
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
import glob
import os
import re
from os import listdir
import matplotlib.pyplot as plt 
from skimage.color import rgb2gray


def importMask(width, length):
    '''Import mask png files. The ground truth mask should have the same dimension with the box image
    Parameters
    ----------
    width: int, single image number of row

    length: int, single image number of column
    
    Returns
    ----------
    
    mask_truth: nd array, contain all classes of mask. 
                mask_truth shape = (num of class, num of masks per class, mask row, mask length)
    
    maskNum: int, number of masks per class
    '''
    # Import mask
    # User should change the paths to ground truth
    chaotic_path ='C:/Users/Tsaip/OneDrive - Imperial College London/Ovation Data Internship 2020/Data/ContenSimilarityTest/ground truth/chaotic mask/mask_ch'
    fault_path ='C:/Users/Tsaip/OneDrive - Imperial College London/Ovation Data Internship 2020/Data/ContenSimilarityTest/ground truth/fault mask/mask_fa'
    salt_path ='C:/Users/Tsaip/OneDrive - Imperial College London/Ovation Data Internship 2020/Data/ContenSimilarityTest/ground truth/salt mask/mask_sa'
    path = [chaotic_path,fault_path,salt_path]

    files = []
    for p in path:
        # Assume each class has the same number of masks
        fileList = glob.glob(p + "/*.png")
        fileList.sort(key = lambda sortNum: int(re.sub(r'\D', '', sortNum))) # Sort the mask name in ascending order
        files.append(fileList)
    del fileList
    mask_truth = np.zeros((len(files),len(files[0]), width, length))
    
    n = 0
    for f in files:
        for path_m, k in zip(f, range(len(files[0]))):
            im = imageio.imread(path_m)
            mask_truth[n][k] = rgb2gray(im) # Turn RGB to gray scale
        n+= 1
    del path_m, k
    mask_truth[mask_truth>0] = 1
    maskNum = len(files[0])
    print('There are', maskNum , ' mask for each class.')
    del files
    return mask_truth, maskNum

def plotPredvsGT(mask_truth, numImagesPerClass, array_data, color_img, classL_acc, maskNum, imgIdx = 0):
    '''
    Plot prediction vs ground truth mask
    
    Parameters
    ----------
    
    mask_truth: nd array, contain all classes of mask. 
                mask_truth shape = (num of class, num of masks per class, mask row, mask length)
    
    numImagesPerClass: int, number of images per class for the original dataset 
    
    array_data: nd array, original dataset
    
    color_img: nd array, colorred mask of the images
    
    classL_acc： int, classes chosen to validate
    
    maskNum: int, total mask numbers per class
    
    imgIdx: int, index of image to show
    
    '''
    if imgIdx > maskNum :
        imgIdx = maskNum-1
    print('Showing the %i'%imgIdx,'image with predicted mask and ground truth.')
    for n, c in zip(range(maskNum), classL_acc):
        fig, ax = plt.subplots(1, 3, figsize=(7, 7))
        if n!=0:
            idx = sum(numImagesPerClass[:c])+imgIdx
        else:
            idx = n+imgIdx
        ax[0].imshow(array_data[idx], cmap='gray')
        ax[0].set_title("Class %i  " %(c+1), fontsize=14)

        ax[1].imshow(array_data[idx], cmap='gray')
        ax[1].imshow(color_img[idx], alpha=.4)
        ax[1].set_title("Predicted Class %i  " %(c+1), fontsize=14)
        ax[2].imshow(mask_truth[n][imgIdx], cmap='gray')
        ax[2].set_title("Ground truth ", fontsize=14)
        fig.savefig('PredvsTruth%i.jpg'%(c+1))
        del fig
    return

def get_Ypred_Ytrue(mask_truth, classL_acc, classified_all, maskNum, numImagesPerClass, bkgrd_label, background = True):
    '''
    Create prediction array and ground truth array.

    Parameters
    ----------
    
    mask_truth: nd array, contain all classes of mask. 
                mask_truth shape = (num of class, num of masks per class, mask row, mask length)
    
    classL_acc： int, classes chosen to validate
    
    classified_all: nd arrays. Contains pixel labels coordinations
    
    maskNum: int, total mask numbers per class
    
    numImagesPerClass: int, number of images per class for the original dataset 
    
    bkgrd_label: int, background label
    
    background: bool, flag for background

    Returns
    ----------
    
    y_truth: nd array, ground truths
    
    y_pred: nd array, pixel-level predictions
    '''
    # We don't validate unkown class's accuracy. Just class 1, 3, 4
    # Class1 label = 0, Class2 label = 1,Class3 label = 2, Class4 label = 3
    mask_pred  = []
    width = mask_truth.shape[2]
    length = mask_truth.shape[3]
    for c in classL_acc:
        
        pred = classified_all[sum(numImagesPerClass[:c]):sum(numImagesPerClass[:c])+ maskNum]
        pred[pred == c] = 1
        # Convert background label to zeros
        if background:
            pred[pred == bkgrd_label] = 0
        mask_pred.append(pred)
   
    y_truth = np.zeros((len(mask_truth), maskNum, width*length))
    y_pred = np.zeros((len(mask_truth), maskNum, width*length))
    for n in range(len(mask_truth)):
        for ii in range(maskNum):
            y_truth[n,ii,:] = mask_truth[n][ii].flatten()
            
            y_pred[n,ii,:] = mask_pred[n][ii].flatten()
    del mask_pred        
    return y_truth, y_pred

def getAccuracy(y_truth, y_pred, classL_acc):
    '''
    Compute the Jacccard Index and accuracy score for multiclass or binary class pixel label prediction.
    Jaccard Score is the intersection of binary images y_pred and y_true divided by the union of BW1 and BW2

    Parameters
    ---------
    
    y_truth: nd array, ground truth
    
    y_pred: nd array, pixel-level prediction
    
    classL_acc: int, classes chosen to validate

    Returns
    ----------
    
    jacc_score: 1d array, contains jaccard index for each predicted classes
    
    cf: 1d array, contains accuracy scores for each predicted classes

    '''
    jacc_score = np.zeros((len(classL_acc)))
    cf = np.zeros((len(classL_acc)))
    for n in range(len(classL_acc)):
        try:
            jacc_score[n] = jaccard_score(y_truth[n].flatten(), y_pred[n].flatten(), average = 'binary')
        except Exception:
            jacc_score[n] = jaccard_score(y_truth[n].flatten(), y_pred[n].flatten(), average = 'micro')
        cf[n] = accuracy_score(y_truth[n].flatten(), y_pred[n].flatten())
    print('Jaccard score:', jacc_score)
    print('Accuracy score: ',cf)
    return jacc_score, cf

def tune_sparsity_alpha(array_data, X, iteration, numImagesPerClass, classL, classL_acc, width, 
                        length, sparsity_opt, kmeans, NumOfClasses, k_clusters):
    ''' 
    Run several iterations to check the effect of sparsity on accuracy and result. 
    User can modify the functions to check how other parameters affect the accracy.

    This function will plot the jaccard indeies and accuracy scores for each predicted class
    '''
    mask_truth, maskNum = importMask(width, length)

    numImages = X.shape[1]
    jacc_score = np.zeros((iteration, len(classL_acc)))
    cf = np.zeros((iteration, len(classL_acc)))

    # sparsity_opt = np.linspace(0.4, 0.85, num = iteration)
    # alpha_opt = np.linspace(0.01, 0.2, num = iteration)
    # sigma_opt = np.linspace(0.2, 1, num = iteration)
    for itr in range(iteration):
        # Set Sparsity
        N_p = len(X)
        set_sparsity = sparsity_opt[itr]
        L1toL2 = sqrt(N_p) - sqrt(N_p-1)*set_sparsity # L1L2ratio
        # if testing alpha, then uncomment the following and modify input 
        # alpha = alpha[itr]
        alpha = 0.057

        #     RunNMF
        W_init, H_init = NMF_Sparse.initialise_WH(width, length, kmeans, NumOfClasses, k_clusters, numImages)
        
        #     Sparsify columns for W_init
        W_init = NMF_Sparse.sparcify_columns(W_init, set_sparsity)
        W_all, H_all = NMF_Sparse.nmf_allClass(X, NumOfClasses, k_clusters, W_init, H_init, alpha = alpha, l1_ratio = L1toL2)
        
        del W_init, H_init# save memory
        
        # Generate likelihood matrix 
        Y = label_Pixel.get_likelihoodMatrix(W_all, H_all, NumOfClasses, k_clusters, numImages, width, length)

        background = True # Choose if the label includes 'background', background indicates no traps
        if background:
            bkgrd_label = NumOfClasses+1 
        else:
            bkgrd_label = NumOfClasses-1
        

        classified_all = label_Pixel.extract_coordination(Y, NumOfClasses, numImages, width, length, bkgrd_label,
                                gaussian_filtering = True, median_filtering = True, sigma = 0.45)
        del Y
        classN = label_Pixel.split_labelResult(classified_all,NumOfClasses, bkgrd_label=bkgrd_label, background=True)

        # blue chaotic; light blue unkown; green fault ; red salt; grey background
        colorarr = np.array([[0,0,255],[100,255,255], [0,255,0], [255,0,0], [183,183,183]])
        color_img = label_Pixel.plotLabels(classL, numImagesPerClass,numImages, width, length, 
                            classN, array_data, colorarr, plot = False)
        del color_img, classN
        
        y_truth, y_pred = get_Ypred_Ytrue(mask_truth, classL_acc, classified_all, maskNum, numImagesPerClass, bkgrd_label)
        del classified_all

        for n in range(len(classL_acc)):
            try:
                jacc_score[itr, n] = jaccard_score(y_truth[n].flatten(), y_pred[n].flatten(), average = 'binary')
            except Exception:
                jacc_score[itr, n] = jaccard_score(y_truth[n].flatten(), y_pred[n].flatten(), average = 'micro')
            cf[itr, n] = accuracy_score(y_truth[n].flatten(), y_pred[n].flatten())

    del mask_truth   
    fig = plt.figure(figsize=(16,5))
    ax0 = fig.add_subplot(121)
    ax0.set_title('Jaccard Score for 3 Classes',fontsize = 17)
    # ax0.set_xlim(sparsity,1)
    ax0.set_xlabel('Sparsity', fontsize = 14)
    ax0.set_ylabel('score',fontsize = 14)
    ax0.grid()
    ax1 = fig.add_subplot(122)
    ax1.set_title('Accuracy Score for 3 Classes',fontsize = 17)
    # ax1.set_xlim(sparsity_opt[0],sparsity_opt[-1])
    ax1.set_xlabel('Sparsity', fontsize = 14)
    ax1.set_ylabel('score',fontsize = 14)
    ax1.grid()
    for n in range(len(y_truth)):
        ax0.plot(sparsity_opt, jacc_score[:, n], label="Class %i"%(classL_acc[n]+1))
        lgd = ax0.legend(loc='best',fontsize = 14)

        ax1.plot(sparsity_opt, cf[:, n], label="Class %i"%(classL_acc[n]+1))
        lgd1 = ax1.legend(loc='best',fontsize = 14)
