""" 
Run this module to tune the model. 
"""
# Author: Ping-Chen Tsai, Aug 2020 

import NMF_Sparse
import label_Pixel
import validate_Accuracy
import numpy as np
from math import sqrt
import scipy.io
from sklearn.cluster import KMeans
import imageio

# Import data
mypath = 'C:/Users/Tsaip//OneDrive - Imperial College London/Ovation Data Internship 2020/Data/ContenSimilarityTest/images_Landmass.mat'
list_data = scipy.io.loadmat(mypath)['images']
array_data = np.array(list_data)

# Define parameters
NumOfClasses = 4 # number of class input
classL = [0, 1, 2, 3] # Labels. should be starting from 0 !!! Cannot be random integers
classL_acc = classL_acc = [0, 2, 3] # Class labels for accuracy test. 
                                    # For landmass, We only class 0, 2, and 3 accuracy because we don't have ground truth of 'unknown' class

k_clusters = 250 # feature numbers selected by users
numImages = array_data.shape[0] # numImages is total num of images
numImagesPerClass = [500, 500, 500, 500] # Same number of images per class for LANDMASS.
set_sparsity = 0.5125 # target sparsity. Tune it 
alpha_ = 0.0575 # Regularization intensity for running NMF model. User can tune it

# Image shape:(width, length)
width = array_data[0].shape[0]
length = array_data[0].shape[1]
del list_data, mypath # Save memory
# Plot five images of each class, uncomment the following to to plot
# NMF_Sparse.plt_boxData(classL, numImagesPerClass, numImages, width, length, array_data = array_data)

# Create data matrix X. Reshape data matrix for NMF input
X = np.reshape(array_data,(numImages,width*length)).T
print('Data matrix X shape: ', X.shape)
# Check if there is negative element in the matrix. The dataset should be normalised to [0, 1]
if (X < 0).any():
    print('Matrix elements are not all positive!')

# split data matrix to classes for factors initialisation
X_class = NMF_Sparse.splitX(X, classL, numImagesPerClass)

#  Run Kmeans for initialisation
kmeans = []
print('Running Kmeans for initialisation...')
for i in np.arange(NumOfClasses):
    kmeans.append(KMeans(n_clusters=k_clusters, max_iter=1000).fit(X_class[i].T)) # Perform kmeans on each dataset features
print('=== Finished ===')
del X_class # save memory

# ===========================================================================================================
# Run multiple times the NMF model to see how sparsity affect the accuracy. User can also modify the function to see alpha effects on accuracy

iteration = 5 # loops of running model
sparsity_opt = np.linspace(0.4, 0.85, num = iteration)
# Modify tune_sparsity_alpha() to tune alpha_opt and sigma_opt
# alpha_opt = np.linspace(0.01, 0.2, num = iteration)
# sigma_opt = np.linspace(0.2, 1, num = iteration)

validate_Accuracy.tune_sparsity_alpha(array_data, X, iteration, numImagesPerClass, classL, classL_acc, width,
                                        length, sparsity_opt, kmeans, NumOfClasses, k_clusters)
# ===========================================================================================================