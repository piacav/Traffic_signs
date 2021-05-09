from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns

"""" Environment variables """
train_set_dir = Path("Dataset", "GTSDB_Train")
val_set_dir = Path("Dataset", "GTSDB_Val")
train_set_gt = str(Path(train_set_dir, "gt_train.txt"))
val_set_gt = str(Path(val_set_dir, "gt_val.txt"))

# load ground truth
train_set = pd.read_csv(train_set_gt, sep=';', names=['path', 'left', 'top', 'right', 'bottom', 'id'])
val_set = pd.read_csv(val_set_gt, sep=';', names=['path', 'left', 'top', 'right', 'bottom', 'id'])

# open image, convert it into numpy array and check the size of array
img = Image.open(str(Path(train_set_dir, train_set['path'][0])))
# img.show()
img = np.array(img)
# plt.imshow(img[:, :, :])
plt.figure(figsize=(20, 10))
# plotting count plot with seaborn
train_set_tot = pd.concat([train_set, val_set], ignore_index=True)
sns.countplot(train_set_tot['id']).set_title('Count of class id')

prohibitory = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
danger = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
mandatory = [33, 34, 35, 36, 37, 38, 39, 40]
other = [6, 12, 13, 14, 17, 32, 41, 42]

# copying id column of data into df
df = train_set_tot['id']
train_set_tot['Object Name'] = train_set_tot['id']
# assigning new labels, 1 implies prohibitory,2 implies danger,3 implies mandatory and 4 implies other.
for i in range(len(df)):
    if df[i] in prohibitory:
        df.loc[i] = 0
        train_set_tot['Object Name'].loc[i] = 'prohibitory'
    elif df[i] in danger:
        df.loc[i] = 1
        train_set_tot['Object Name'].loc[i] = 'danger'
    elif df[i] in mandatory:
        df.loc[i] = 2
        train_set_tot['Object Name'].loc[i] = 'mandatory'
    elif df[i] in other:
        df.loc[i] = 3
        train_set_tot['Object Name'].loc[i] = 'other'
    else:
        df.loc[i] = -1
print(train_set_tot.head())


plt.figure(figsize=(10, 5))
# plotting count plot with seaborn
sns.countplot(train_set_tot['id']).set_title('Count of class id')
# the bounding box size ranges is from 32 to 248 pixels.
# Most of the boxes lie in 40 to 100 pixel range, very less boxes are more than 150 pixels
