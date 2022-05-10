import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
import cv2 as cv
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import img_as_float

def just_plot(image_mats, dim, size):
    x, y= dim[0], dim[1]
    plt.figure(figsize=size)
    for i in range(x):
        for j in range(y):
            plt.subplot(x, y, i*y+j+1)
            plt.imshow(image_mats[i], cmap='gray_r')
    plt.show()

def radon_transform(image):
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta)    
    return sinogram

def inv_radon(image, sinogram):
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    reconstruction_fbp = iradon(sinogram, theta=theta, filter_name='ramp')    
    return reconstruction_fbp

def cluster_radon(sinogram, ext=6):
    H, W = sinogram.shape[0], sinogram.shape[1]
    res = np.zeros((H, W))
    cumsum = np.cumsum(sinogram, 0)
    eps = 1e-6
    mid = W//2
    for i in range(0, H):
        idx = min(i+5, H-1)
        numer = abs((cumsum[idx, :mid]-cumsum[i, :mid])-(cumsum[idx, -mid:]-cumsum[i, -mid:]))
        deno = (cumsum[idx, :mid]-cumsum[i, :mid])+(cumsum[idx, -mid:]-cumsum[i, -mid:])
        res[i, :mid] = 1- (numer/(deno+eps))
        res[i, -mid:] = 1- (numer/(deno+eps))
    return res

def contrast_stretch(image):
    max_int = np.max(np.max(image, 0))
    min_int = np.min(np.min(image, 0))
    image = 255*(image-min_int)/(max_int-min_int)
    return np.uint8(image)

def line_selection(lines, min_dist=20):
    h, w = lines.shape[0], lines.shape[1]
    res = np.zeros((h, w))
    im_float = img_as_float(lines)
    coordinates = peak_local_max(im_float, min_distance=min_dist)
    res[coordinates[:, 0], coordinates[:, 1]]=255
    if np.sum(res==255) == 0:
        coordinates = peak_local_max(im_float, min_distance=0) 
        res[coordinates[:, 0], coordinates[:, 1]] = 255
    return res


def detect_lines(image, linedist=25, rspc=1, tspc=2, thresh=0.7, cluster=10):
    image_contrast = contrast_stretch(image)
    bil_filtered = cv.bilateralFilter(image_contrast, 15, 1000, 0.001)
    edges = cv.Canny(bil_filtered, 100, 200)
    
    sinogram = radon_transform(edges)
    clustered = cluster_radon(sinogram, cluster)
    sinogram_clust = sinogram*clustered
    max_val = np.max(np.max(sinogram_clust, 0))
    threshed = sinogram_clust*(sinogram_clust > thresh*max_val)
    select_lines = line_selection(threshed, 20)
    inv_sino = inv_radon(edges, select_lines)
    # just_plot([clustered, threshed], (1, 2), (4, 8))
    kernel = np.ones((2, 2), np.uint8)
    _, binthresh = cv.threshold(inv_sino, 0, 255, cv.THRESH_BINARY)
    image_dilation = cv.dilate(binthresh, kernel, iterations=1)
    coordis = []
    for i in range(image_dilation.shape[0]):
        for j in range(image_dilation.shape[1]):
            if image_dilation[i, j]==255:
                coordis.append((i, j))
    coordis = np.array(coordis)
    return coordis