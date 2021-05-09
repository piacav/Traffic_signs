import tkinter as tk
from tkinter import *
from tkinter import filedialog

import cv2
import numpy as np
import pandas as pd
from PIL import ImageTk, Image
from keras.models import load_model
from pathlib import Path

model = load_model(str(Path('Models', 'class_cnn_220210503T2208')))
names = pd.read_csv("labels.csv")["Name"].values

# initialise GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Traffic sign classification')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top, background='#CDCDCD')


def grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def equalize(image):
    image = cv2.equalizeHist(image)
    return image


def preprocessing(image):
    image = grayscale(image)
    image = equalize(image)
    image = image / 255
    return image


def classify(file_path):
    global label_packed
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


def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 20, 'bold'),
                         highlightbackground='#364156')
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded = uploaded.resize((200, 200), Image.ANTIALIAS)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im, width=200, height=200, background='#CDCDCD')
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 20, 'bold'), highlightbackground='#364156')

upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Know Your Traffic Sign", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()
