import cv2
from tkinter import *
from tkinter.filedialog import *
import numpy as np
from matplotlib import pyplot as plt

def traitement(img, imgQuery):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img,None)
    kp2, des2 = sift.detectAndCompute(imgQuery,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img,kp1,imgQuery,kp2,good,None,flags=2)

    return img3, good, matches

if __name__ == "__main__":
    allRef = {
        'img/templates/doge.jpg',
        'img/templates/feelsbadman.jpg',
        'img/templates/scumbagsteve.jpg'
    }

    filepath = askopenfilename(title="Ouvrir une image", filetypes=[('img files', '.png'), ('all files', '.*')])
    img = cv2.imread(filepath);

    bestResult = [None, [], []]

    for template in allRef:
        imgQuery = cv2.imread(template, 0)
        img3, good, matches = traitement(img, imgQuery)
        if bestResult[1].__len__() < good.__len__():
            bestResult[0] = img3
            bestResult[1] = good
            bestResult[2] = matches

    plt.imshow(bestResult[0])
    plt.show()