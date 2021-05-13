import os
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import skimage.draw

gt_train = Path('Dataset', 'GTSDB', 'Metadata', 'gt_train.txt')


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

path_full = Path('Dataset', 'GTSDB', 'Full')


def image_load(path):
    names = []  # Nomi delle immagini es. '00000.ppm'
    imgs = []  # Immagini aperte con Image.open es. es. <PIL.PpmImagePlugin.PpmImageFile image mode=RGB size=1360x800 at 0x7F968779DF40>
    ir = []  # Immagini lette con skimage es. [array([[[255, 255, 255], [255, 255, 255], [255, 255, 255], ...]

    for file in sorted(os.listdir(path)):
        if Path(path, file).is_file() and Path(path, file).suffix == '.ppm':
            imgs.append(Image.open(Path(path, file)))
            image = skimage.io.imread(Path(path, file))
            names.append(file)
            ir.append(image)
    return imgs, names, ir


# imgs, names = image_load(path_full)
# for el, name in zip(imgs, names):
#     print(el, name)
# print(image_load(path_full))


'''
Builds images annotations for the training process from a semicolon-separated-values file.
Creates a dictionary with filenames as keys and values lists of tuples containing bounding box coordinates of signs
in the image.
Parameters:
    path        : annotation file path
Returns:
    annotations : final dictionary containing signs annotations
'''

ex_train = Path('Dataset', 'GTSDB', 'Metadata', 'ex.txt')


def annotation_preprocess(path):
    annotations_dict = {}
    annotations_list = []
    with open(path) as file:
        for line in file.readlines():
            temp = line.strip('\n').split(';')
            annotations_list.append(temp)
            # temp[0][:-4] + '.jpg' for images in jpg
            id = temp[0][:-4]
            if id not in annotations_dict.keys():
                annotations_dict[id] = [(int(temp[1]), int(temp[2]), int(temp[3]), int(temp[4]), int(temp[5]))]
            else:
                # annotations_dict[temp[0]].append(int(temp[5]))
                annotations_dict[id].append((int(temp[1]), int(temp[2]), int(temp[3]), int(temp[4]), int(temp[5])))
    return annotations_dict, annotations_list


# print(annotation_preprocess(gt_train)[1])

'''
Converts a .ppm image and saves it in .jpg format
Parameters:
    path:   image path
    file:   image file name
'''

def ppm2jpg(path, file):
    # path = Path('Dataset', 'GTSDB', 'Train')
    # path2 = Path('Dataset', 'GTSDB', 'Train2')

    # for file in os.listdir(path):
    im = Image.open(Path(path, file))
    print(file[:-4] + '.jpg')
    im.save(Path(path, 'lello.jpg'))

# ppm2jpg('Dataset/GTSRB_Test', '00000.ppm')
