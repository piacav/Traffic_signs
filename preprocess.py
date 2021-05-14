import os
from pathlib import Path

import cv2
import skimage.draw
from PIL import Image
from sklearn.model_selection import train_test_split

'''
Splits the total training set into training and validation sets
Validation set is 10% of the total training set
Parameters:
    path    : path of the total training set
Returns:
    train   : list of training images
    val     : list of validation images
'''


def split_trainset(path):
    imgs = list(annotation_preprocess(path)[0].keys())
    train, val = train_test_split(imgs, train_size=0.9)
    for v in val:
        os.replace(Path('Dataset', 'GTSDB', 'Train', v), Path('Dataset', 'GTSDB', 'Validation', v))
    return train, val


'''
Loads the images and creates a list of names and a list of loaded images
Parameters:
    path    : path of the folder from which images have to be loaded
Returns  :
    imgs    : list of all loaded images
    names   : list of names of all images
'''


def image_load(path):
    # image names i.e. '00000.ppm'
    names = []
    # images loaded with Image.open i.e.<PIL.PpmImagePlugin.PpmImageFile image mode=RGB size=1360x800 at 0x7F968779DF40>
    imgs = []
    # images read with skimage i.e. [array([[[255, 255, 255], [255, 255, 255], [255, 255, 255], ...]
    ir = []

    for file in sorted(os.listdir(path)):
        if Path(path, file).is_file() and Path(path, file).suffix == '.ppm':
            imgs.append(Image.open(Path(path, file)))
            image = skimage.io.imread(Path(path, file))
            names.append(file)
            ir.append(image)
    return imgs, names, ir


'''
Builds images annotations for the training process from a semicolon-separated-values file.
Creates a dictionary with filenames as keys and values lists of tuples containing bounding box coordinates of signs
in the image.
Parameters:
    path        : annotation file path
Returns:
    annotations : final dictionary containing signs annotations
'''


def annotation_preprocess(path):
    annotations_dict = {}
    annotations_list = []
    with open(path) as file:
        for line in file.readlines():
            temp = line.strip('\n').split(';')
            annotations_list.append(temp)
            id = temp[0][:-4]
            if id not in annotations_dict.keys():
                annotations_dict[id] = [(int(temp[1]), int(temp[2]), int(temp[3]), int(temp[4]), int(temp[5]))]
            else:
                annotations_dict[id].append((int(temp[1]), int(temp[2]), int(temp[3]), int(temp[4]), int(temp[5])))
    return annotations_dict, annotations_list


'''
Converts a .ppm image and saves it in .jpg format
Parameters:
    path:   image path
    file:   image file name
'''


def ppm2jpg(path, file):

    im = Image.open(Path(path, file))
    print(file[:-4] + '.jpg')
    im.save(Path(path, 'lello.jpg'))


'''
Converts image in gray-scale format
Parameters:
    img:    image path
Returns:
    img:    gray-scaled image
'''


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


'''
Equalizes a gray-scaled image
Parameters:
    img:    image path
Returns:
    img:    equalized gray-scaled image
'''


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


'''
Applies preprocessing functions like grayscale and equalize, then normalizes the image
Parameters:
    img:    image path
Returns:
    img:    preprocessed image
'''


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)  # STANDARDIZE THE LIGHTING IN AN IMAGE
    img = img / 255  # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
    return img
