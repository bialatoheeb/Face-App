import cv2
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from scipy import ndimage as ndi
#https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_gabor.html

Sex = {'1' : 'Female', '0': 'Male'}
Race = {'0': 'White', '1': 'Black', '2': 'Asian', '3':'Indian', '4' : 'Others'}


def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

def window_mean_std(new_img):
    # Divide the (img_size, img_size) pixels into (10, 10) pixels each
    # Obtain the mean and standard deviation of each (10, 10) pixel
    # This results in a (400, 2) array to be used
    window = 10
    new_win = int(new_img.shape[0]/window)
    arr = sliding_window_view(new_img, window_shape = (window, window))[::window, ::window].reshape((new_win*new_win, window*window))
    edges_window_mean = np.mean(arr, axis=1)
    edges_window_std = np.std(arr, axis=1)

    return np.array([edges_window_mean, edges_window_std]).T



def process_img(name, img_size):

    # Read in image
    img = cv2.imread(name)

    # Sharpen image to avoid null va;ues for very blurry images
    kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
    img2 = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

    img2 = cv2.resize(img2, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img2 = cv2.Canny(img2, 100, 200, 3, L2gradient=True)
    img2 = (img2 - img2.mean()) / img2.std()
    img2 = window_mean_std(img2)



    img3 = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    # Gabor feature
    ksize = 15
    lamda = np.pi/4
    theta = np.pi/4
    sigma = 1.0
    phi = 1
    gamma = 0.5

    kernel = cv2.getGaborKernel((ksize, ksize),sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(img3, cv2.CV_8UC3, kernel)
    img3  = power(filtered_img, kernel)
    img3 = (img3 - img3.mean()) / img3.std()
    img3 = window_mean_std(img3)



    return  np.array([img2, img3]).flatten()
