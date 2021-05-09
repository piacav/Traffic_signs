import os
from preprocess import annotation_preprocess
from pathlib import Path

gtpath = 'Dataset/GTSDB_Train/gt_val.txt'
datasetdir = 'Dataset/GTSDB_Train/images'
# <object-class> <x> <y> <width> <height>
def create_annot():
    annot_dict, _ = annotation_preprocess(gtpath)
    for k in annot_dict.keys():

        nome = 'Dataset/GTSDB_Val/' + k + '.txt'
        f = open(nome, 'w+')
        for elem in annot_dict[k]:
            '''
            size = (1360, 800)
            dw = 1. / size[0]
            dh = 1. / size[1]
            x = (elem[0] + elem[2]) / 2.0
            y = (elem[1] + elem[3]) / 2.0
            w = elem[2] - elem[0]
            h = elem[3] - elem[1]
            x = x * dw
            w = w * dw
            y = y * dh
            h = h * dh
            '''

            w = round(float((elem[2] - elem[0]) / 1360), 6)
            h = round(float((elem[3] - elem[1]) / 800), 6)
            x = round((float((elem[0] + elem[2]) / 2) / 1360), 6)
            y = round(float(((elem[1] + elem[3]) / 2)) / 800, 6)

            object_class = 0
            line = str(object_class) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
            print(line)
            #f.write(line)
        f.close()

def create_pathtxt():
    f = open('Dataset/GTSDB_Train/val_path.txt', 'w+')
    contenuto = ''
    annot_dict, _ = annotation_preprocess(gtpath)
    for img in annot_dict.keys():
        img_path = str(Path(datasetdir, img + '.jpg'))
        contenuto += os.path.abspath(img_path) + '\n'
    f.write(contenuto.strip())
    f.close()

