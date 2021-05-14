import tkinter as tk
from pathlib import Path
from tkinter import *
from tkinter import filedialog
from preprocess import preprocessing

import cv2
import numpy as np
import pandas as pd
from PIL import ImageTk, Image
from keras.models import load_model

'''ENVIRONMENT VARIABLES'''

model = load_model(str(Path('Models', 'class_cnn_220210503T2208')))
names = pd.read_csv("labels.csv")["Name"].values

'''Initialize GUI'''
top = tk.Tk()
top.geometry('1200x800')
top.title('Traffic sign classification')
top.configure(background='#CDCDCD')
label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top, background='#CDCDCD')

'''
Classifies the uploaded image and shows the prediction in the GUI. Callback of "Classify image" button
Parameters:
    file_path:  image path
'''


def classify(file_path):

    image = cv2.imread(file_path)
    image = np.asarray(image)
    image = cv2.resize(image, (48, 48))
    image = preprocessing(image)
    image = image.reshape(1, 48, 48, 1)

    prediction = model.predict(image)
    class_index = model.predict_classes(image)
    probability_value = np.amax(prediction)
    sign = names[class_index[0]]
    print(sign, probability_value)
    sign = sign + '\n' + str(round(probability_value * 100, 2)) + '%'
    if probability_value >= 0.9:
        label.configure(foreground='#19db07', text=sign, font=('arial', 40, 'bold'), background='#CDCDCD')
    else:
        label.configure(foreground='#db0707', text=sign, font=('arial', 40, 'bold'), background='#CDCDCD')


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
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


'''CONFIGURE THE LAYOUT AND START THE GUI'''

upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 20, 'bold'), highlightbackground='#364156')
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Classify Traffic Sign", pady=20, font=('arial', 30, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()
