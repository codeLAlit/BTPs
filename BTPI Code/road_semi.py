from tqdm import tqdm
from skimage.color import rgb2lab, lab2rgb
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

def train(imagedir, labeldir):
    imfiles = os.listdir(imagedir)
    anet = 0
    bnet = 0
    for imgname in imfiles:
        img = plt.imread(os.path.join(imagedir, imgname))
        label = plt.imread(os.path.join(labeldir, imgname.split('.')[0]+'.tif'))
        Lab = rgb2lab(img)
        a = Lab[:, :, 1]
        b = Lab[:, :, 2]
        al = a*label
        bl = b*label
        al = np.sum(np.sum(al))/np.sum(np.sum(label))
        bl = np.sum(np.sum(bl))/np.sum(np.sum(label))
        anet+=al
        bnet+=bl
    ast = anet/len(imfiles)
    bst = anet/len(imfiles)
    return ast, bst

def contrast_stretch(image):
    max_int = np.max(np.max(image, 0))
    min_int = np.min(np.min(image, 0))
    image = 255*(image-min_int)/(max_int-min_int)
    return np.uint8(image)

def algorithm(test, ast, bst):
    lab = rgb2lab(test)
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    h, w = a.shape
    am, bm = ast, bst

    ar = am*np.ones((h, w))
    br = bm*np.ones((h, w))

    ea = np.sqrt((ar-a)**2+(br-b)**2)
    ea_max = np.max(np.max(ea, 0))
    ea = ea/ea_max
    ea = contrast_stretch(ea)
    ret, thresh1 = cv.threshold(ea, 5, 255, cv.THRESH_BINARY_INV)
    edges = cv.Canny(thresh1, 1000, 1200)
    res = np.copy(test)

    for i in range(h):
        for j in range(w):
            if edges[i, j] == 255:
                res[i, j, :] = np.array([255, 0, 0])

    return res

train_imagedir = "Train2/image"
train_labeldir = "Train2/labels"
ast, bst = train(train_imagedir, train_labeldir)
print(ast, bst)
files = os.listdir("Test2_images")
for file in tqdm(files):
    test = plt.imread(os.path.join("Test2_images", file))
    res = algorithm(test, ast, bst)
    plt.imsave("Result2_images/"+file.split('.')[0]+'.png', res)
