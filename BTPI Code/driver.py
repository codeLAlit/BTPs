import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale
from cluster_radon import detect_lines
import os
from tqdm import tqdm

total = len(os.listdir("Test1_images"))
ent = 1
for img in tqdm(os.listdir("Test1_images")):
    original = cv.imread("Test1_images/"+img)
    original = cv.cvtColor(original, cv.COLOR_BGR2RGB)
    testimage = cv.imread("Test1_images/"+img, 0)
    mind = min(testimage.shape[0], testimage.shape[1])
    testimage = cv.resize(testimage, (mind, mind))
    original = cv.resize(original, (mind, mind))
    scale_fact = 0.5
    if mind < 400:
        scale_fact = 1
    testimage = rescale(testimage, scale=scale_fact, mode='reflect')
    original = rescale(original, scale=scale_fact, mode='reflect', multichannel=True)
    coordis = detect_lines(testimage)

    try:
        plt.figure()
        plt.imshow(original, cmap=plt.cm.gray)
        plt.autoscale(False)
        plt.plot(coordis[:, 1], coordis[:, 0], linestyle='None', marker='.', markerfacecolor='#97ecfc', markersize=1)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("Result1_images/result"+img, bbox_inches='tight',pad_inches = 0)
    except:
        plt.figure()
        plt.imshow(original, cmap=plt.cm.gray)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("Result1_images/result"+img, bbox_inches='tight',pad_inches = 0)

    ent+=1
