import cv2
import glob, os
import webbrowser
import tkinter as tk
from tkinter.filedialog import *
from PIL import Image
from PIL import ImageTk as tki


class MainApp:
    def __init__(self, root):
        root.geometry("800x600")
        # Barre de menu
        self.menubar = Menu(root)
        self.menu1 = Menu(self.menubar, tearoff=0)
        self.menu1.add_command(label="Ouvrir...", command=self.work)
        self.menu1.add_separator()
        self.menu1.add_command(label="Quitter", command=root.quit)
        self.menubar.add_cascade(label="Fichier", menu=self.menu1)
        # un vrai bouton ça serait bien
        self.menu1.add_command(label="So dank", command=self.knowYourDank)

        self.panel = PanedWindow(root, orient=HORIZONTAL)
        self.panel.pack(side=TOP, expand=Y, fill=BOTH, pady=2, padx=2)

        # Bouton KnowYourMeme
        self.button1 = Button(self.panel, text="So dank", command=self.knowYourDank)
        self.panel.add(self.button1)

        # Image
        self.imgStart = Image.open("img/start.png")
        self.tkimgStart = tki.PhotoImage(self.imgStart)
        self.labelImage = Label(self.panel, image=self.tkimgStart, anchor=CENTER)
        self.labelImage.bind("<Configure>", self.resize_image)
        self.panel.add(self.labelImage)

        self.panel.pack()

        root.config(menu=self.menubar)

        # Attributs
        self.result = None
        self.copy_of_image = None
        self.im = None
        self.imgtk = None

    def traitement(self, img, imgQuery):
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img, None)
        kp2, des2 = sift.detectAndCompute(imgQuery, None)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(img, kp1, imgQuery, kp2, good, None, flags=2)

        return img3, good, matches

    def chooseImage(self):
        filepath = askopenfilename(title="Ouvrir une image",
                                   filetypes=[('img files', '.png .jpg .jpeg'), ('all files', '.*')])
        if filepath == '':
            return None
        return filepath

    def findMeme(self, filepath):
        allRef = []

        try:
            os.chdir("img/templates")
        except:
            print("already in")

        for file in glob.glob("*.jpg"):
            allRef.append([os.path.abspath(file), os.path.splitext(file)[0]])

        print(allRef)
        if filepath is None:
            return None
        img = cv2.imread(filepath)

        bestResult = [None, [], [], None]

        for template in allRef:
            imgQuery = cv2.imread(template[0], 0)
            img3, good, matches = self.traitement(img, imgQuery)
            if bestResult[1].__len__() < good.__len__():
                bestResult[0] = img3
                bestResult[1] = good
                bestResult[2] = matches
                bestResult[3] = template[1]
        return bestResult

    def work(self):
        filepath = self.chooseImage()
        self.result = self.findMeme(filepath)

        self.im = Image.fromarray(self.result[0])
        self.imgtk = tki.PhotoImage(image=self.im)

        # Put it in the display window
        self.labelImage.imgtk = self.imgtk;
        self.labelImage.config(image=self.imgtk)

        self.copy_of_image = self.im.copy()

    # Ouvrir la page KYM du meme identifié
    def knowYourDank(self):
        try:
            webbrowser.open('www.knowyourmeme.com/memes/' + self.result[3])
        except:
            print("Pas de result")

    def resize_image(self, event):
        new_width = self.labelImage.winfo_width()
        new_height = self.labelImage.winfo_height()
        image = self.copy_of_image.resize((new_width, new_height))
        photo = tki.PhotoImage(image=image)
        self.labelImage.config(image=photo)
        self.labelImage.image = photo  # avoid garbage collection


if __name__ == "__main__":
    fenetre = tk.Tk()
    fenetre.title('MemeID')

    MainApp(fenetre)

    fenetre.mainloop()
