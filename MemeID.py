import cv2
import tkinter as tk
from tkinter.filedialog import *
import numpy as np
from matplotlib import pyplot as plt
import glob, os

from PIL import Image
from PIL import ImageTk as tki
from tkinter import PhotoImage

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

def chooseImage():
    filepath = askopenfilename(title="Ouvrir une image",
                               filetypes=[('img files', '.png .jpg .jpeg'), ('all files', '.*')])
    if filepath == '':
        return None
    return filepath

def findMeme(filepath):
    allRef = []

    os.chdir("./img/templates")
    for file in glob.glob("*.jpg"):
        allRef.append(os.path.abspath(file))

    if filepath is None:
        return None
    img = cv2.imread(filepath);

    bestResult = [None, [], []]

    for template in allRef:
        imgQuery = cv2.imread(template, 0)
        img3, good, matches = traitement(img, imgQuery)
        if bestResult[1].__len__() < good.__len__():
            bestResult[0] = img3
            bestResult[1] = good
            bestResult[2] = matches
    return bestResult

def work():
    filepath = chooseImage()
    result = findMeme(filepath)
    #Ici, on veut prendre l'image dans result[0] puis l'afficher dans le canvas
    #aide : https://stackoverflow.com/questions/28670461/read-an-image-with-opencv-and-display-it-with-tkinter
    canvas = Canvas(fenetre, width=350, height=200)
    
    canvas.pack()


if __name__ == "__main__":
    fenetre = tk.Tk()
    fenetre.title('MemeID')

    #Barre de menu
    menubar = Menu(fenetre)
    menu1 = Menu(menubar, tearoff=0)
    menu1.add_command(label="Ouvrir...", command=work)
    menu1.add_separator()
    menu1.add_command(label="Quitter", command=fenetre.quit)
    menubar.add_cascade(label="Fichier", menu=menu1)

    fenetre.config(menu=menubar)

    #Image
    photo = None

    canvas = Canvas(fenetre, width=350, height=200)
    canvas.create_image(0, 0, anchor=NW, image=photo)
    canvas.pack()

    # plt.imshow(bestResult[0])
    # plt.show()

    fenetre.mainloop()
