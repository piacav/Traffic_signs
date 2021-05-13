import datetime
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

import CNNs

""" ENVIRONMENT VARIABLES """

model_dir = "Models"
dataset = str(Path("Dataset", "GTSRB_Train"))  # class folders with data
labels = "labels.csv"  # names of classes
batch_size_val = 50
epochs_val = 15
img_dim = (48, 48, 3)  # (32, 32, 3) for images 32x32 RGB
input_shape = (48, 48, 1)  # (32, 32, 1) for images 32x32 B&W
test_ratio = 0.2
validation_ratio = 0.2
model_type = "class_cnn_2"  # "class_cnn_2" for the second type

# Importing Images
class_count = 0
images = []
classes = []
number_classes = len(os.listdir(dataset)) - 1
print("Total Classes Detected:", number_classes)
print("Importing Classes...")
for directory in range(number_classes):
    for pic in os.listdir(Path(dataset, str(class_count))):
        image = cv2.imread(str(Path(dataset, str(class_count), pic)))
        image = cv2.resize(image, (48, 48))
        images.append(image)
        classes.append(class_count)
    print(class_count)
    class_count += 1
images = np.array(images)
classes = np.array(classes)

# Split Data
# X_train = ARRAY OF IMAGES TO TRAIN
# y_train = CORRESPONDING CLASS ID
X_train, X_test, y_train, y_test = train_test_split(images, classes, test_size=test_ratio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_ratio)

# TO CHECK IF NUMBER OF IMAGES MATCHES TO NUMBER OF LABELS FOR EACH DATA SET
print("Data shapes", '\n',
      "Train:", X_train.shape, y_train.shape, '\n',
      "Validation:", X_validation.shape, y_validation.shape, '\n',
      "Test:", X_test.shape, y_test.shape)
assert (X_train.shape[0] == y_train.shape[0]), "Number of images not equal to the number of labels in training set"
assert (X_validation.shape[0] == y_validation.shape[0]), "Number of images not equal to the number of labels in val set"
assert (X_test.shape[0] == y_test.shape[0]), "Number of images not equal to the number of labels in test set"
assert (X_train.shape[1:] == img_dim), "The dimensions of the training images are wrong"
assert (X_validation.shape[1:] == img_dim), "The dimensions of the validation images are wrong"
assert (X_test.shape[1:] == img_dim), "The dimensions of the test images are wrong"

# READ CSV FILE
data = pd.read_csv(labels)
print("Labels shape", data.shape)

# DISPLAY SOME SAMPLES IMAGES  OF ALL THE CLASSES
num_of_samples = []
cols = 5
fig, axs = plt.subplots(nrows=number_classes, ncols=cols, figsize=(5, 50))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["Name"])
            num_of_samples.append(len(x_selected))

# DISPLAY A BAR CHART SHOWING NO OF SAMPLES FOR EACH CATEGORY
print("Number of samples for each class", num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, number_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
plt.show()


# PREPROCESSING THE IMAGES

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)  # STANDARDIZE THE LIGHTING IN AN IMAGE
    img = img / 255  # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
    return img


# CREATE DATASETS
X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

# SHOW A GRAY SCALE IMAGE IN THE TRAIN SET
fig, ax = plt.subplots(1, 10, figsize=(20, 5))
fig.tight_layout()
fig.suptitle("Gray scale images in the train set", fontsize=40)
for i in range(10):
    ax[i].imshow(X_train[i].reshape(img_dim[0], img_dim[1]), cmap=plt.get_cmap("gray"))
    ax[i].axis('off')
plt.show()

# ADD A DEPTH OF 1
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_validation = X_validation.reshape((X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

# AUGMENTATION OF IMAGES
dataGen = ImageDataGenerator(width_shift_range=0.1,  # 0.1 = 10% pixels
                             height_shift_range=0.1,
                             zoom_range=0.2,  # 0.2 means can go from 0.8 to 1.2
                             shear_range=0.1,  # magnitude of shear angle
                             rotation_range=10)  # degrees
dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

# SHOW AUGMENTED IMAGE SAMPLES
fig, ax = plt.subplots(1, 10, figsize=(20, 5))
fig.tight_layout()
fig.suptitle("Augmented image samples", fontsize=40)
for i in range(10):
    ax[i].imshow(X_batch[i].reshape(img_dim[0], img_dim[1]), cmap=plt.get_cmap("gray"))
    ax[i].axis("off")
plt.show()

y_train = to_categorical(y_train, number_classes)
y_validation = to_categorical(y_validation, number_classes)
y_test = to_categorical(y_test, number_classes)

# TRAIN
_, class_cnn, class_cnn_2 = CNNs.create_cnns(input_shape, number_classes)

if model_type == "class_cnn":
    model = class_cnn
else:
    model = class_cnn_2

model.summary()
model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                    steps_per_epoch=len(X_train) // batch_size_val, epochs=epochs_val,
                    validation_data=(X_validation, y_validation), shuffle=1)

# PLOT ACCURACY AND LOSS
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

# STORE THE MODEL
name = os.path.join(model_dir, "{}{:%Y%m%dT%H%M}".format(model_type, datetime.datetime.now()))
print("Creation model:", name)
model.save(name)
del model
