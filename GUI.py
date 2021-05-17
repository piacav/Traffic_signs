import tkinter as tk
from pathlib import Path
from tkinter import *
from tkinter import filedialog

import cv2
import numpy as np
import pandas as pd
from PIL import ImageTk, Image
from keras.models import load_model

from preprocess import preprocessing
from Test_det import postprocess

'''ENVIRONMENT VARIABLES'''
names = pd.read_csv("labels.csv")["Name"].values

'''Initialize GUI'''
top = tk.Tk()
top.geometry('1200x800')
top.title('Traffic sign classification')
top.configure(background='#CDCDCD')
sign_image = Label(top, background='#CDCDCD')

'''
Classifies the uploaded image and shows the prediction in the GUI. Callback of "Classify image" button
Parameters:
    file_path:  image path
'''


def classify(file_path):
    cap = cv.VideoCapture(file_path)
    hasFrame, frame = cap.read()
    # Create a 4D blob from a frame
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence and show the predicted image in the gui
    sign_image.image = postprocess(frame, outs)


'''
Shows the "Classify image" button when the image is loaded in the GUI
Parameters:
    file_path:  image path
'''


def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 20, 'bold'),
                         highlightbackground='#364156')
    classify_b.place(relx=0.79, rely=0.46)


'''
Open the filesystem dialog and lets the user choose the image to load. Callaback to "Upload image" button
'''


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded = uploaded.resize((400, 400), Image.ANTIALIAS)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im, width=400, height=400, background='#CDCDCD')
        sign_image.image = im
        show_classify_button(file_path)
    except:
        pass


'''CONFIGURE THE LAYOUT AND START THE GUI'''

upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 20, 'bold'), highlightbackground='#364156')
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Classify Traffic Sign", pady=20, font=('arial', 30, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()
