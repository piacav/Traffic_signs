from pathlib import Path

import numpy as np
import pandas as pd
from keras.models import load_model
from keras_drop_block import DropBlock2D
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from preprocess import preprocessing

""" ENVIRONMENT VARIABLES """

test_set_dir = str(Path("Dataset", "GTSRB_Test"))
classIDs = pd.read_csv("labels.csv")["ClassId"].values
names = pd.read_csv("labels.csv")["Name"].values
gt_test = pd.read_csv(str(Path("Dataset", "GTSRB_Test", "gt_test.csv")))["ClassId"].values
threshold = 0.9  # PROBABILITY THRESHOLD
model_dir = "Models"
input_shape = (48, 48)
# "class_cnn" for the first type,
# "class_cnn_2" for the second type
# "both" for the mean value of the scores of class_cnn and class_cnn_2
model_type = "class_cnn_2"
# A list of one model or 2 models.
# If model_type = 'class_cnn', model_name is something like ["class_cnn20210503T2012"]
# If model_type = 'class_cnn_2', model_name is something like ["class_cnn_220210503T2208"]
# If model_type = 'both', in position 0 class_cnn and in position 1 class_cnn_2
# example: model_name = ["class_cnn20210503T2012", "class_cnn_220210503T2208"]
model_name = ["class_cnn20210503T2012", "class_cnn_220210503T2208"]
model = None
model1 = None
model2 = None

# import the correct trained model
if model_type == 'class_cnn':
    model = load_model(str(Path(model_dir, model_name[0])), custom_objects={'DropBlock2D': DropBlock2D})  # class_cnn
    model.summary()
elif model_type == 'class_cnn_2':
    model = load_model(str(Path(model_dir, model_name[1])))  # class_cnn_2
    model.summary()
elif model_type == "both":
    model1 = load_model(str(Path(model_dir, model_name[0])), custom_objects={'DropBlock2D': DropBlock2D})  # class_cnn
    model1.summary()
    model2 = load_model(str(Path(model_dir, model_name[1])))  # class_cnn_2
    model2.summary()
else:
    assert model_type, "Variable model_type must be 'class_cnn', 'class_cnn_2' or 'both'. If both, variable " \
                       "model_name must be a list of models. Es:['class_cnnX', 'class_cnn-2Y'] "

"""
Predicts the class label of the street sign.
When model_type is 'both':
    If the predicted class is the same for both models,
    it shows the average confidence, otherwise it shows the one with the highest confidence. 
When model_type is class_cnn or class_cnn_2 calls test_image_specific(image)
Parameters:
    image    : image to predict 
    pic      : image number 
Returns:
    String containing class index, class name 
    probability_value : confidence of the prediction  
"""


def test_img(image, pic):
    print('IMAGE NUMBER', pic)
    # PREDICT IMAGE
    # Two different models
    if model_type == "both":
        # Model 1
        prediction1 = model1.predict(image)
        class_index1 = model1.predict_classes(image)
        probability_value1 = np.amax(prediction1)

        # Model 2
        prediction2 = model2.predict(image)
        class_index2 = model2.predict_classes(image)
        probability_value2 = np.amax(prediction2)

        # Take the avg value or the best one
        if (probability_value1 > threshold and class_index1[0] == gt_test[int(pic)]) or \
                (probability_value2 > threshold and class_index2[0] == gt_test[int(pic)]):
            if class_index1 == class_index2:
                score_tot = round((probability_value1 + probability_value2) / 2, 8)
                detected_class = str(class_index1[0]) + ' - ' + str(names[class_index1[0]])
            elif probability_value1 >= probability_value2:
                score_tot = probability_value1
                detected_class = str(class_index1[0]) + ' - ' + str(names[class_index1[0]])
            else:
                score_tot = probability_value2
                detected_class = str(class_index2[0]) + ' - ' + str(names[class_index2[0]])
            print('DETECTED CORRECT CLASS:', detected_class)
            print('Total score: ', score_tot)
        else:
            print('DETECTION FAILED:', '\nCORRECT CLASS:', gt_test[int(pic)], '-', names[gt_test[int(pic)]])

        print('Model 1 (class_cnn):', class_index1[0], '-', names[class_index1[0]],
              '\nScore:', probability_value1,
              '\nModel 2 (class_cnn_2):', class_index2[0], '-', names[class_index2[0]],
              '\nScore:', probability_value2, '\n' + ('=' * 100))
        return str(class_index[0]) + ' - ' + str(names[class_index[0]]), probability_value
    else:
        return test_img_specific(image)


"""
Predicts the class label of the street sign using one specific model defined in the environment variable model_type
Parameters:
    image    : image to predict 
Returns:
    String containing class index, class name 
    probability_value : confidence of the prediction  
"""


def test_img_specific(image):
    # Change model_type to change the model

    prediction = model.predict(image)
    class_index = model.predict_classes(image)
    probability_value = np.amax(prediction)

    if probability_value > threshold:
        detected_class = str(class_index[0]) + ' - ' + str(names[class_index[0]])
        print('DETECTED CORRECT CLASS:', detected_class)
    else:
        print('DETECTION FAILED:', class_index[0], '-', names[class_index[0]])

    print('Model:', model_type, '\nScore:', probability_value, '\n' + ('=' * 100))

    return str(class_index[0]) + ' - ' + str(names[class_index[0]]), probability_value


"""
Compute metrics 
Parameters:
    test_set    : list of images to test
    t           : threshold
Returns:
    FRR         : False rejection rate 
    FAR         : False acceptance rate
    ACC         : Accuracy
    t           : threshold
"""


def compute_metrics(test_set, t):
    # Preprocessing and reshape of the images in train set
    test_set = np.array(list(map(preprocessing, test_set)))
    test_set = test_set.reshape((test_set.shape[0], test_set.shape[1], test_set.shape[2], 1))

    if model_type == "both":
        pred1 = model1.predict_classes(test_set)
        pred2 = model2.predict_classes(test_set)
        print('ACCURACY Model1:', round(accuracy_score(gt_test, pred1), 8))
        print('ACCURACY Model2:', round(accuracy_score(gt_test, pred2), 8))
    else:
        pred = (model.predict_proba(test_set)[:, 1] > t).astype('float')

        cr = classification_report(gt_test, pred)
        cnf_matrix = confusion_matrix(gt_test, pred)

        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        # Negative predictive value
        NPV = TN / (TN + FN)
        # Fall out or false positive rate = FAR
        FAR = FP / (FP + TN)
        # False negative rate = FRR
        FRR = FN / (TP + FN)
        # False discovery rate
        FDR = FP / (TP + FP)
        # Overall accuracy for each class
        ACC = (TP + TN) / (TP + FP + FN + TN)
        # print(cr, TN, FP, FN, TP, TPR, TNR, PPV, NPV, FAR, FRR, FDR, ACC)
        return FRR, FAR, ACC, t


''' 
# READ IMAGE
data = []
for picture in sorted(os.listdir(test_set_dir)):
    if Path(test_set_dir, picture).suffix != '.ppm':
        continue
    original_img = cv2.imread(str(Path(test_set_dir, picture)))
    name_pic = picture[:-4]
    # PREPROCESS IMAGE
    img = cv2.resize(original_img, (input_shape[0], input_shape[1]))
    data.append(img)
    img = preprocessing(img)

    # SHOW PREPROCESSED IMAGE
    # cv2.imshow("Processed Image", image)

    img = img.reshape(1, input_shape[0], input_shape[1], 1)

    # Test every image in test-set
    # test_img(img, name_pic)

X_test = np.array(data)
# compute_metrics(X_test)

metriche = []
t = 0
for t in range(0, 100, 20):
    print(t)
    metriche.append(compute_metrics(X_test, t/100))
print(metriche)'''
