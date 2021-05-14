Traffic Signs Project (guide for Google Colab)
---

Clone Traffic_signs repo from git and give all permissions

`!git clone https://github.com/piacav/Traffic_signs.git`

`!mkdir /content/Traffic_signs/Models/backup`

`!chmod +x ./Traffic_signs`


Detection GTSDB - Train
---
### Clone and Build Darknet

Clone darknet repo from git and give all permissions

`!git clone https://github.com/AlexeyAB/darknet`

`!chmod +x ./darknet`

Change makefile to have GPU and OPENCV enabled

`%cd darknet`

`!sed -i 's/OPENCV=0/OPENCV=1/' Makefile`

`!sed -i 's/GPU=0/GPU=1/' Makefile`

`!sed -i 's/CUDNN=0/CUDNN=1/' Makefile`

Verify CUDA

`!/usr/local/cuda/bin/nvcc --version`

Make darknet (build), ignore warnings

`!make`

---
### Data analysis
![Alt text](Metadata/Trainset_det_Hist_class.png?raw=true)

Visualize classes

`import pandas as pd`

`classes = pd.read_csv('/content/Traffic_signs/labels.csv')`

`print(classes)`

---
### Train

Download already trained weights
 
`!gdown -O /content/Traffic_signs/Models/backup/darknet-yolov3_last.weights --id '1-4mn5BhwKenRQsAZJG-5lwy3iCrXtq3O'`

Install requirements

`!pip3 install -r /content/Traffic_signs/requirements.txt`

`%cd /content/Traffic_signs`

Create file darknet.data
 
`def create_data():`

>`with open('/content/Traffic_signs/darknet.data', 'w') as f:`

>>`f.write('classes = 1\n')`
>>
>>`f.write('train = /content/Traffic_signs/Dataset/GTSDB_Train/train_path.txt\n')`
>>
>>`f.write('valid = /content/Traffic_signs/Dataset/GTSDB_Train/val_path.txt\n')`
>>
>>`f.write('names = /content/Traffic_signs/classes.names\n')`
>>
>>`f.write('backup = /content/Traffic_signs/Models/backup')`

`create_data()`

Create val_path.txt and train_path.txt to link images of validation set and train set

`!python3 /content/Traffic_signs/create_annotations.py`

Check weights

`import os`

`if os.path.isfile('/content/Traffic_signs/Models/backup/darknet-yolov3_last.weights'):`

>`print('WEIGHTS ALREADY PRESENT')`

`else:`
>`# check if lines 8 and 9 of /content/Traffic_signs/darknet-yolov3.cfg are uncommented`
>
>`%cd /content/darknet`
>
>`# train weights`
>
>`!./darknet detector train /content/Traffic_signs/darknet.data /content/Traffic_signs/darknet-yolov3.cfg -dont_show> /content/Traffic_signs/Models/backup/train.log`

### Resume train (only if necessary)

To resume the training, uncomment the cell below 

`!./darknet detector train /content/Traffic_signs/darknet.data /content/Traffic_signs/darknet-yolov3.cfg /content/Traffic_signs/Models/backup/darknet-yolov3_last.weights -dont_show> /content/Traffic_signs/Models/backup/train.log`

---
## Recognition GTSRB - Train
### Data analysis
![Alt text](Metadata/Trainset_rec_Hist.png?raw=true)

---
### Train

One time with 
> model_type = "class_cnn_2"

and one time with 
> model_type = "class_cnn"

`%cd /content/Traffic_signs`

`import os`

Check if weights of CNN are already present 

CNN

`if not os.path.isdir('/content/Traffic_signs/Models/class_cnn20210503T2012'):`

>`# training class_CNN`
>
>`%matplotlib inline`
>
>`%run /content/Traffic_signs/Train_rec.py`

CNN2

`import os`

`# check if weights of CNN2 are already present`

`if not os.path.isdir('/content/Traffic_signs/Models/class_cnn_220210503T2208'):`

>`# create class_CNN_2`
>
>`%matplotlib inline`
>
>`%run /content/Traffic_signs/Train_rec.py`

---
## Test 

Define helper function imShow

`def imShow(path):`

>`import cv2`
>
>`import matplotlib.pyplot as plt`
>
>`%matplotlib inline`
>
>`image = cv2.imread(path)`
>
>`height, width = image.shape[:2]`
>
>`resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)`
>
>`fig = plt.gcf()`
>
>`fig.set_size_inches(18, 10)`
>
>`plt.axis("off")`
>
>`plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))`
>
>`plt.show()`
  
Run custom detector with this command (upload an image to your google drive to test, thresh flag sets accuracy that detection must be in order to show it)

`%cd /content/darknet`

`!./darknet detector test /content/Traffic_signs/darknet.data /content/Traffic_signs/darknet-yolov3.cfg /content/Traffic_signs/Models/backup/darknet-yolov3_last.weights -thresh 0.70 /content/Traffic_signs/Dataset/GTSDB_Test/00693.jpg -dont_show` 

`imShow('predictions.jpg')`

Metrics for detection model 

`%cd /content/darknet/`

`!./darknet detector map /content/Traffic_signs/darknet.data /content/Traffic_signs/darknet-yolov3.cfg /content/Traffic_signs/Models/backup/darknet-yolov3_last.weights` 

FINAL TEST: detection and recognition 

`%cd /content/Traffic_signs`

`!python3 /content/Traffic_signs/Test_det.py --device='gpu' --image=/content/Traffic_signs/Dataset/GTSDB_Test/00649.jpg`

`imShow('/content/Traffic_signs/Output/recognition_image.jpg')`

![Alt text](Output/recognition_image.jpg?raw=true)
