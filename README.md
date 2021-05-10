# Traffic_signs
Traffic sign detection with YOLOv3 detector and recognition using CNNs.

# STEPS FOR DETECTION:
1. Download Darknet in another folder using the command: 
`git clone https://github.com/AlexeyAB/darknet`
2. Give every permission to folders of Darknet and Traffic_signs
3. Go in Traffic_signs folder
4. Run `create_annotations.py` to create `train_path.txt` and `val_path.txt` for Darknet
5. Change absolute paths in darknet.data
6. Run Darknet with this command: 
`./darknet detector train /content/drive/MyDrive/Traffic_signs/darknet.data /content/drive/MyDrive/Traffic_signs/darknet-yolov3.cfg -dont_show> /content/drive/MyDrive/Traffic_signs/Models/backup/train.log`

# STEPS FOR RECOGNITION:
1. Run `GUI_rec.py`