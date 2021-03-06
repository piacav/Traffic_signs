import os
import tkinter as tk
from pathlib import Path
from tkinter import *
from tkinter import filedialog

import pandas as pd
from PIL import ImageTk, Image

'''ENVIRONMENT VARIABLES'''
names = pd.read_csv("labels.csv")["Name"].values
inpWidth = 608  # Width of network's input image
inpHeight = 608  # Height of network's input image
output_path = str(Path('Output', 'recognition_image.jpg'))
# net = Test_det.net  # net of detection

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
    command = 'python3 Test_det.py --image={}'.format(file_path)
    os.system(command)
    pred = Image.open(output_path)
    pred.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
    im = ImageTk.PhotoImage(pred)
    sign_image.configure(image=im, width=1024, height=600, background='#CDCDCD')
    sign_image.image = im
    '''cap = cv.VideoCapture(file_path)
    hasFrame, frame = cap.read()
    # Create a 4D blob from a frame
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(Test_det.getOutputsNames(net))

    # Remove the bounding boxes with low confidence and show the predicted image in the gui
    Test_det.postprocess(frame, outs)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = Image.fromarray(np.uint8(frame))
    frame.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
    print(type(frame))
    im = ImageTk.PhotoImage(frame)
    sign_image.configure(image=im, width=1024, height=600, background='#CDCDCD')
    sign_image.image = im'''


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
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        print(type(uploaded))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im, width=1024, height=600, background='#CDCDCD')
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
