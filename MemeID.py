import cv2
import numpy as np
import glob, os
import webbrowser
import tkinter as tk

from matplotlib import pyplot as plt
from tkinter import PhotoImage
from tkinter.filedialog import *
from PIL import Image
from PIL import ImageTk as tki

class MainApp:

    def __init__(self, root):
        #Barre de menu
        self.menubar = Menu(root)
        self.menu1 = Menu(self.menubar, tearoff=0)
        self.menu1.add_command(label="Ouvrir...", command=self.work)
        self.menu1.add_separator()
        self.menu1.add_command(label="Quitter", command=root.quit)
        self.menubar.add_cascade(label="Fichier", menu=self.menu1)
        # un vrai bouton ça serait bien
        self.menu1.add_command(label="So dank", command=self.knowYourDank)
        self.labelImage = Label(root)
        self.labelImage.pack(fill=BOTH)

        root.config(menu=self.menubar)

    def traitement(self, img, imgQuery):
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

    def chooseImage(self):
        filepath = askopenfilename(title="Ouvrir une image",
                                   filetypes=[('img files', '.png .jpg .jpeg'), ('all files', '.*')])
        if filepath == '':
            return None
        return filepath

    def findMeme(self, filepath):
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
            img3, good, matches = self.traitement(img, imgQuery)
            if bestResult[1].__len__() < good.__len__():
                bestResult[0] = img3
                bestResult[1] = good
                bestResult[2] = matches
        return bestResult

    def work(self):
        filepath = self.chooseImage()
        result = self.findMeme(filepath)

        im = Image.fromarray(result[0])
        imgtk = tki.PhotoImage(image=im)

        # Put it in the display window
        self.labelImage.imgtk = imgtk;
        self.labelImage.config(image=imgtk)

    # Ouvrir la page KYM du meme identifié
    def knowYourDank(self, bestResult):
        webbrowser.open('www.knowyourmeme.com/memes/')


if __name__ == "__main__":
    fenetre = tk.Tk()
    fenetre.title('MemeID')

    MainApp(fenetre)

    # plt.imshow(bestResult[0])
    # plt.show()

    fenetre.mainloop()
