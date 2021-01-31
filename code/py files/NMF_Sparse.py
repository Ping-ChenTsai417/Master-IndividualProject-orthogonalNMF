""" 
Run Non-nagative matrx factorization with sparseness constraints
"""
# Author: Ping-Chen Tsai, Aug 2020 

import time
import numpy as np
import matplotlib.pyplot as plt 
from math import sqrt

from sklearn.decomposition import NMF

def plt_boxData(classL, numImagesPerClass, numImages, width, length, array_data = None, WH_product = None):
    '''
    Plot the first five sample images of the dataset in each class.
    It will give 25 plots in total.

    Parameters
    ----------
    classL: list, list of class labels. Starting from 0

    numImagesPerClass: list, relating to classL: image number per corresponding class

    numImages: int, total number of images in image dataset

    width: int, single image row dimension
    
    length: int, single image column dimension
    
    array_data: 3D array-like, stacked classes with shape = numImg, imgWidth, imgLength
    
    WH_product: nd array, W*H product, . Need to be reshped for plotting
        
    Return
    ----------
    
    none
    '''
    try:
        img_all = np.reshape(WH_product.T, (numImages,width,length))
    except Exception:
        img_all = array_data

    for i in classL:
        fig, axarr = plt.subplots(1, 5, figsize=(11, 11))
        for ax, n in zip(axarr.flatten(), np.arange(5)):
            if i!=0:
                idx = sum(numImagesPerClass[:i])+n
            else:
                idx = n
                # user can modify [idx+5] to choose the images display
            ax.imshow(img_all[idx+5], cmap='gray')
            ax.set_title("%i. " % (n+1) + "Class %i"  % (i+1), fontsize=13)
        plt.show()
        del fig, axarr
        
def splitX(X, classL, numImagesPerClass):
    '''
    Split data matrix X into classes matrix for performing Kmeans on each class

    Parameters
    ----------
    X: 2D array, shape = total number of images, image dimension
    
    classL: list, list of class labels. Starting from 0

    Returns
    ----------
    X_class: list of arrays, splitted classes of data
    '''
    X_class = []
    # split features based on classes
    for i in classL:
        if i!= 0:
            cur_idx = sum(numImagesPerClass[:i])
            next_idx = cur_idx + numImagesPerClass[i]
        else:
            cur_idx = 0
            next_idx = numImagesPerClass[i]
            
        X_class.append(X[:, cur_idx:next_idx])
        
    return X_class
def initialise_WH(width, length, kmeans, NumOfClasses, k_clusters, numImages):
    '''
    Initialisation of W and H

    Parameters
    ----------
    width: int, row of an image data
    
    length: int, column of an image data
    
    kmeans: list, kmeans classifiers for classes
    
    NumOfClasses: int
    
    k_clusters: int
    
    numImages: int

    Returns
    ----------
    
    W_init: nd array
    
    H_init: nd array
    '''
    # Extract kmeans centres of each class and stack them into W_init
    W_init = np.zeros((width*length, NumOfClasses*k_clusters))
    for i in np.arange(NumOfClasses):
        W_init[:, i*k_clusters:(i+1)*k_clusters] = kmeans[i].cluster_centers_.T
    
    np.random.seed(20)
    H_init = np.random.uniform(0.0, 1.0, size = (NumOfClasses*k_clusters, numImages))

    # Make sure all the centroids are nonnegative
    W_init[W_init<0.00001] = 0
    return W_init, H_init

def nmf_singleClass(NumOfClasses, X_class, k_clusters, alpha = 0, l1_ratio = 0):
    '''
    This is a experimental function. Leave this function
    Run Non-negative matrix factorization for each classes.
    Assign the updated W and H into a list.

    Parameters
    ----------
    
    NumOfClasses: int
    
    k_clusters: int, should be less than image dimension. Selected by elbow's method

    Returns
    ----------
    
    W_class: list of arrays, contains W for each class
    
    H_class: list of arrays, contains H for each class
    
    '''
    # Individual class nmf
    NMFmodel = []
    W_class = []
    H_class = []

    for i in np.arange(NumOfClasses):
        NMFmodel = NMF(n_components=k_clusters,init='nndsvd', solver='mu', 
                            beta_loss='frobenius', tol=0.0001, max_iter=500, 
                            random_state=None, alpha = alpha, l1_ratio = l1_ratio)


        W_class.append(NMFmodel.fit_transform(X_class[i]))
        H_class.append(NMFmodel.components_)
    del NMFmodel
    return W_class, H_class

def nmf_allClass(X, NumOfClasses, k_clusters, W_init, H_init, alpha=0, l1_ratio =0):
    '''
    Run Non-negative matrix factorization for each classes.
    Assign the updated W and H into a list.

    Parameters
    ----------
    
    NumOfClasses: int
    
    k_clusters: int, should be less than image dimension. Selected by elbow's method

    Returns
    ----------
    
    W_all: 2d array, updated W for all classes. shape = image dimension, num of components
    
    H_all: 2d array, updated H for all classes. shape = num of components, total num of sample
    '''
    # Used kmeans centroid as W and H initialisation. Use nmf package for updating W and H.
    n_components = NumOfClasses*k_clusters
    print('We are now running NMF with sparsity constraints ... ')
    modelTest1 = NMF(n_components= n_components, init='custom', solver='mu', 
                     beta_loss='frobenius', tol=0.0001, max_iter=500, 
                     random_state=None, alpha = alpha, l1_ratio = l1_ratio)

    W_all = modelTest1.fit_transform(X, W = W_init, H = H_init)
    H_all = modelTest1.components_
    del modelTest1
    print('=========== Finished ==============')
    return W_all, H_all

def stack_singleWH(W_class, H_class, NumOfClasses,numImagesPerClass, k_clusters, width, length, numImages):
    '''
    This is a experimental function. It is not used for generating masks. Leave this.
    Stack each class of updated W hozizontally and updated H diagonally for processing

    Parameters
    -----------
    
    W_class: list of arrays, contains W for each class
    
    H_class: list of arrays, contains H for each class

    Returns
    ----------
    
    W_conc: nd array, shape = single image dimension, components number
    
    H_conc: nd array, shape = components number, number of total dataset samples
    
    '''
    W_conc = np.zeros((width*length, k_clusters*NumOfClasses))
    H_conc = np.zeros((k_clusters*NumOfClasses, numImages))

    for i in np.arange(NumOfClasses):
        if i!=0:
            cur_idx = sum(numImagesPerClass[:i])
            next_idx = cur_idx + numImagesPerClass[i]
        
        else:# i==0
            cur_idx = 0
            next_idx = numImagesPerClass[i]
        W_conc[:, i*k_clusters:(i+1)*k_clusters] = W_class[i]
        H_conc[i*k_clusters:(i+1)*k_clusters, cur_idx:next_idx] = H_class[i]
    return W_conc, H_conc

def replace_Hall_with_Hclass(H_all, H_class, NumOfClasses, numImagesPerClass, k_clusters):
    ''' 
    This is a experimental function. It is not used for generating masks. Leave this.
    Stack single class of Hclass result onto all class result H_all
    Concatenate W and H to to facilitate stacked calculation
    
    Parameters
    ----------
    
    H_all: nd array, updated H for the whole dataset. shape = components number, total num of sample

    Returns
    ----------
    
    H_concat: nd array, shape = components number, number of total dataset samples
    '''
    H_concat = H_all.copy()
    for i in np.arange(NumOfClasses):
        if i!=0:
            cur_idx = sum(numImagesPerClass[:i])
            next_idx = cur_idx + numImagesPerClass[i]
        else:# i==0
            cur_idx = 0
            next_idx = numImagesPerClass[i]
        H_concat[i*k_clusters:(i+1)*k_clusters, cur_idx:next_idx] = H_class[i]
    return H_concat

def projection(N_p, s, L1, L2sqr, nonNeg, set_sparsity):
    '''
    Projection operator that enforces sparsity on columns.
    When a vector s is given, find a same-sized vector v that has the 
    desired L1 and L2 norm that has the closes Euclidian distance to s
    
    If the nonNeg flag is set, the output v is
    constrained to being non-negative (v>=0).
    
    Parameters
    ----------
    
    N_p: int, number of pixels
    
    s: 1d array, vector to be sparsified
    
    L1: L1 norm
    
    L2sqr: L2 square norm
    
    nonNeg: Bool, non negativity flag
    
    set_sparsity: double, target sparsity set by user

    Returns
    ----------
    
    v: 1d array, sparsified columns
    
    iterations: int, number of iteration to convergence
    '''
    N = len(s)

    try: 
        nonNeg
    except NameError: 
        isneg = s<0 # save column negativity state
        s = abs(s) # take absolute of the colume.

    # projecting the point to the sum constraint hyperplane
    v = s + (L1-sum(s))/N
    # Initialize an array for zero valued components
    zerocoeff = []
    j = 0
    while 1:
        midpoint = np.ones((N,1))*L1/(N-len(zerocoeff))#  projection operator by Hoyer(2004)
        midpoint[zerocoeff] = 0
        midpoint=np.reshape(midpoint,(len(midpoint),))
        w = v-midpoint
        a = sum(w**2)
        # b = 2*w.T*v
        b = 2*w@v
        c = round(sum(v**2)-L2sqr, 5) 
        alphap = (-b+np.real(sqrt(b**2-4*a*c)))/(2*a)
        v = alphap*w + v

        if all(vv>=0 for vv in v):
            # we found the solution!
#             print('All elements in v are non-negative')
            itrations = j+1
            break
        j += 1

        # Set negs to zero, subtract appropriate amount from rest
        zerocoeff = np.where(v<=0)
    #     print("Replace the negative values of the array with 0")
        v[v <= 0] = 0
        tempsum = sum(v)
        v = v + (L1-tempsum)/(N_p-len(zerocoeff)) #Calculate c := (sum(s)−L1)/(dim(x)−size(Z))
        v[v <= 0] = 0
        new_sparsity = sum(v==0)/len(v)
        if new_sparsity > set_sparsity:
            v[v <= 0] = 0
            itrations = j
#             print('ALERT: Does not converge during sparsifying process')
            break
    # sum(v.^2)=k2 which is closest to s in the euclidian sense
    try: 
        nonNeg
    except NameError: 
        print('Return v\'s original sign')
    #     (-2*isneg + 1) make the non-nagative element index  -1
        v = np.multiply((-2*isneg + 1), v) # Return original signs to solution
    if any(abs(v.imag))>1e-5:
        print('ERROR: you have imaginary values!')
    del zerocoeff, a, b, c, w, s, tempsum, alphap
    return v, itrations

def sparcify_columns(mat, set_sparsity):
    '''
    Sparsifying the matrix column.
    The algorithm takes the vector w_i as input and find a vector v that
    has target L1 and L2 norms that achieve the pre-defined sparseness ρ(w).

    Parameters
    ----------
    
    mat: nd array, matrix to be sparsified
    
    set_sparsity: double, target sparsity
    
    Returns
    ----------
    
    mat: nd array, sparsified matrix
    
    References
    ----------
    
    Patrik O. Hoyer, 2004
    '''
    print('Please wait. We are sparsifying columns, it takes about 7-10 minutes ... ')
    #     Number of pixel
    N_p = len(mat)
    #  desired L1 to L2 ratio to acheive sparsity level:
    L1toL2 = sqrt(N_p) - sqrt(N_p-1)*set_sparsity # L1L2ratio
    #L2 norms of columns of matrix
    L2W_norm = np.linalg.norm(mat, axis = 0) #L2W
    #     Apply sparseness constraints on W_init
    i = 1
    start = time.time() # claculate sparsifying time
    for i in range(0, mat.shape[1]):
        col = mat[:, i]
    #     set colume to achieve desired sparseness 
        L1W_norm = L1toL2*L2W_norm[i]
        scol, itr = projection(N_p, col, L1W_norm, L2W_norm[i]**2, True, set_sparsity)
    #     update:
        mat[:, i] = scol
        del scol, itr
#     sparse_matrix = mat
    end = time.time()
    print('=== Finished ===')
    print("Elapsed time for applying sparseness constraint:", 
                                             end - start, 'seconds.') 
    del L2W_norm, start, end, N_p
    return mat

def plotHistogram_W(W_, name_, NumOfClasses, k_clusters):
    '''
    Plot Histogram of basis matrix.
    The histogram indicates disdributions of each geophysical feature magnitude.
    
    Parameters
    ----------
    
    W_ : numpy array of basis matrix W

    name_: numpy array of string
    
    Examples
    ----------
    
    W_ = np.array([W1, W2])
    
    name_ = np.array([name1, name2])

    '''
    f, axs = plt.subplots(1,2, figsize=(15,3))
    im = np.arange(2) # Plot distribution of two W
    
    for n, W, name in zip(im, W_, name_):
        w = np.zeros((NumOfClasses, W.shape[0]*k_clusters))
        W_flat = np.reshape(W, (W.shape[0]*W.shape[1],))
        for i in np.arange(NumOfClasses):
#             Assign classes of w into w[i] for plotting
            w[i] = W_flat[W.shape[0]*k_clusters*i : W.shape[0]*k_clusters*(i+1)]

            # You can normalize it by setting 
            # density=True and stacked=True. By doing this the total area under each distribution becomes 1.
            # e.g: dict(density=True, stacked=True)
            kwargs = dict(alpha=0.35) # Set opacity

            axs[n].hist(w[i], **kwargs, label= 'Class %i'%(i+1) )
        del w, W_flat
        axs[n].set_title('Frequency Histogram of '+ name)
        axs[n].set_xlabel('Seismic Feature Magnitude',fontsize=14)
        axs[n].set_ylabel('Frequency',fontsize=14)
        axs[n].legend(prop={'size': 14})
    plt.show()
    print('The above histograms show distibution of features in two different basis matrices W')
    f.savefig('W_Histograms.jpg')
    del f
    return 