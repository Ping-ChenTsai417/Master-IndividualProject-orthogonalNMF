# %matplotlib inline
import numpy as np
import scipy.io
import scipy.misc
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
import glob
# import os
# from os import listdir
# from os.path import isfile, join
import matplotlib.pyplot as plt 
from matplotlib import style 
from math import sqrt

mypath = 'C:/Users/Tsaip/OneDrive - Imperial College London/Ovation Data Internship 2020/Data/ContenSimilarityTest/images_Landmass.mat'

a = 2
b = 3
c = a+b
