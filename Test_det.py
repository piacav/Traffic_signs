import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
from pathlib import Path
from Test_rec import *

""" Environment variables """

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 608  # Width of network's input image
inpHeight = 608  # Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
parser.add_argument("--device", default="cpu", help="Device to inference on")
args = parser.parse_args()

# Img example
if args.video:
    media = args.video
elif args.image:
    media = args.image
else:
    # To test a specific image or video
    media = str(Path('Dataset', 'GTSDB_Test', '00673.jpg'))  # str(Path('Dataset', 'GTSDB_video_2s.mp4'))
    args.image = media  # for video change with args.video = media

# Input shape for recognition cnn
input_shape = (48, 48)

# Load names of classes
classesFile = "classes.names"

with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Configuration and weight files for the model
modelConfiguration = str(Path('darknet-yolov3.cfg'))
modelWeights = str(Path('Models', 'backup', 'darknet-yolov3_last.weights'))  # Modify with the correct weights
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

if args.device == "cpu":
    net.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif args.device == "gpu":
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")


# Names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def drawPred(classId, conf_det, left, top, right, bottom, class_rec, conf_rec):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

    # Create the label for detection
    label_det = '%.2f' % conf_det
    if classes:
        assert (classId < len(classes))
        label_det = '%s:%s' % (classes[classId], label_det)

    # Create the label for recognition
    label_rec = str(class_rec) + ' ' + str(round(conf_rec, 2))

    # Display the recognition label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label_rec, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 (0, 0, 255), cv.FILLED)
    cv.putText(frame, label_rec, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if detection[4] > confThreshold:
                print('DETECTION:\t', detection[4], " - ", confidence, " - th : ", confThreshold)
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        print('BOXES:\t', boxes)
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        r_width = left + width
        r_height = top + height
        img = frame[top - 5:r_height + 5, left - 5:r_width + 5, :]

        # SHOW PREDICTION
        # cv2.imshow("BBox", img)
        # cv2.waitKey(0)
        # cv2.destroyWindow("BBox")

        img = cv2.resize(img, (input_shape[0], input_shape[1]))
        img = preprocessing(img)
        img = img.reshape(1, input_shape[0], input_shape[1], 1)
        class_name, probability_val = test_single_img2(np.asarray(img))

        drawPred(classIds[i], confidences[i], left, top, left + width, top + height, class_name, probability_val)


# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

cap = None
outputFile = "yolo_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = '/Users/piacavasinni/PycharmProjects/Traffic_signs/Results_img/' + 'result_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4] + '_yolo_out_py.avi'


# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:

    # get frame from the video
    hasFrame, frame = cap.read()

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        break

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))


    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    # cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Write the frame with the detection boxes
    if args.video:
        vid_writer.write(frame.astype(np.uint8))
    else:
        cv.imwrite(str(Path('Output', 'recognition.jpg')), frame.astype(np.uint8))

    # Uncomment the following lines to show recognition image
    # cv.imshow(winName, frame)
    # cv.waitKey(0)
