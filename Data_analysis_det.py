from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

"""" ENVIRONMENT VARIABLES """

train_set_dir = Path("Dataset", "GTSDB_Train")
train_set_gt = str(Path(train_set_dir, "gt_train.txt"))
val_set_gt = str(Path(train_set_dir, "gt_val.txt"))

# Load ground truth
train_set = pd.read_csv(train_set_gt, sep=';', names=['path', 'left', 'top', 'right', 'bottom', 'id'])
val_set = pd.read_csv(val_set_gt, sep=';', names=['path', 'left', 'top', 'right', 'bottom', 'id'])

""" PREPROCESSING AND PREPARATION OF DATA"""

# Open image, convert it into numpy array and check the size of array
img = Image.open(str(Path(train_set_dir, 'images', train_set['path'][0][:-4] + '.jpg')))
img = np.array(img)
plt.figure(figsize=(20, 10))

# Plot the number of classes with seaborn
train_set_tot = pd.concat([train_set, val_set], ignore_index=True)
sns.countplot(train_set_tot['id']).set_title('Count of class id')
plt.show()
