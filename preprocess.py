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
Carica le immagini e crea una lista con i nomi e una lista con le immagini caricate
Parametri:
    path    : il percorso della cartella da cui caricare le immagini
Ritorna  :
    imgs    : lista che contiene tutte le immagini caricate
    names   : lista che contiene i nomi di tutte le immagini
'''

path_full = Path('Dataset', 'GTSDB', 'Full')


def image_load(path):
    names = [] # Nomi delle immagini es. '00000.ppm'
    imgs = [] # Immagini aperte con Image.open es. es. <PIL.PpmImagePlugin.PpmImageFile image mode=RGB size=1360x800 at 0x7F968779DF40>
    ir = [] # Immagini lette con skimage es. [array([[[255, 255, 255], [255, 255, 255], [255, 255, 255], ...]

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
Costruisce le annotazioni delle immagini per il training a partire da un file con semicolon separated values
creando un dizionario in cui per ogni elemento, la chiave è il nome del file e il valore è una lista che contiene
le tuple delle coordinate dei bounding boxes dei cartelli presenti nella foto
Parametri:
    path        : percorso del file di testo che contiene le annotazioni
Ritorna:
    annotations : dizionario finale che contiene le annotazioni dei cartelli
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
