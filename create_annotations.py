import os
from pathlib import Path

from preprocess import annotation_preprocess

gt_val = str(Path('Dataset', 'GTSDB_Train', 'gt_val.txt'))
gt_train = str(Path('Dataset', 'GTSDB_Train', 'gt_train.txt'))
dataset_dir = str(Path('Dataset', 'GTSDB_Train', 'images'))
val_path = str(Path('Dataset', 'GTSDB_Train', 'val_path.txt'))
train_path = str(Path('Dataset', 'GTSDB_Train', 'train_path.txt'))


# <object-class> <x> <y> <width> <height>
def create_annotation(path):
    annotation_dict, _ = annotation_preprocess(path)
    for k in annotation_dict.keys():

        nome = dataset_dir + k + '.txt'
        f = open(nome, 'w+')
        for elem in annotation_dict[k]:

            w = round(float((elem[2] - elem[0]) / 1360), 6)
            h = round(float((elem[3] - elem[1]) / 800), 6)
            x = round((float((elem[0] + elem[2]) / 2) / 1360), 6)
            y = round(float(((elem[1] + elem[3]) / 2)) / 800, 6)

            object_class = 0
            line = str(object_class) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
            print(line)
            f.write(line)
        f.close()


def create_path_txt(gt, path):

    f = open(path, 'w+')
    content = ''
    annotation_dict, _ = annotation_preprocess(gt)
    for img in annotation_dict.keys():
        img_path = str(Path(dataset_dir, img + '.jpg'))
        content += os.path.abspath(img_path) + '\n'
    f.write(content.strip())
    f.close()


create_path_txt(gt_val, val_path)
create_path_txt(gt_train, train_path)
