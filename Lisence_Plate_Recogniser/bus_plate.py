# import the necessary packagesimport cv2 as cv
import numpy as np
import time
from matplotlib import pyplot as plt
import os
from PIL import Image
import pytesseract
import argparse

#creating required variable and setting up constant values
a=[]
b=[]
c=[]
d=[]
plate_timing=1
plate_threshold=0.8

#seetting up the accurasy level we require
CONFIDENCE = 0.5
THRESHOLD = 0.3

#No_Plate Weight Path
weights_path = "yolo/my_yolo3_tiny_training/yolov3-tiny_33000_bus.weights"                 ##location of files in my computer##
config_path = "yolo/my_yolo3_tiny_training/yolov3-tiny.cfg"
label_path = "yolo/my_yolo3_tiny_training/obj.names"

#Vehicle Weight Path
weights_path_plate = "yolo/my_yolo3_tiny_training/yolov3-tiny_10400.weights"
config_path_plate = "yolo/my_yolo3_tiny_training/yolov3-tiny.cfg"                          ##location of files in my computer##
label_path_plate = "yolo/my_yolo3_tiny_training/obj.names"

#loading number of lables
Labels = open(label_path).read().strip().split("\n")
motor_count = 0


#colors
np.random.seed(42)
colours = np.random.randint(0,255, size = (len(Labels),3), dtype = "uint8")

#Loading the yolo detector from path
print("[Message]Loading the yolo detector")
net = cv.dnn.readNetFromDarknet(config_path,weights_path)
ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

#Imput video to be predicted
cap = cv.VideoCapture('videos/maingate.mp4')
writer = None
i = None
(W,H) = (None,None)
count =0

try:
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print("[Message] {} total number of frames in the video".format(length)) 
    
except:
    print("[Message] Could'nt determine the number of frames in the video")
    total = -1

while True:
    ret, frame = cap.read()       #seperating frames from the video
    count += 2
    cap.set(1,count)
    
    if not ret:
        break
    
    if W is None or H is None:
        (H,W) = frame.shape[:2]
	#print(frame.shape)
    

    blob = cv.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB = True, crop = False)
    # forward the blob to the network
    net.setInput(blob)
    
    #declaring variable to store prediction location
    layerOutputs = net.forward(ln)
    boxes = []
    confidences =[]
    classIDs = []
    
    #storing the prediction values in variable
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > CONFIDENCE:
                box = detection[0:4] * np.array([W,H,W,H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                boxes.append( [x,y,int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
		
    idxs = cv.dnn.NMSBoxes(boxes, confidences , CONFIDENCE , THRESHOLD)
    
    #locating predicted area on frames
    if len(idxs) > 0:
        e=0
        f=-1
        for i in idxs.flatten():
            (x,y) = (boxes[i][0], boxes[i][1])
            (w,h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colours[classIDs[i]]]
            ROI = frame[y:y+h , x:x+w , :]
            #if Labels[classIDs[i]] == 'car':
            cv.rectangle(frame, (x,y), (x + w, y + h), color , 2)
            text = "{}: {:.4f}".format(Labels[classIDs[i]],confidences[i])
            cv.putText(frame , text, (x, y-5) ,  cv.FONT_HERSHEY_SIMPLEX, 0.5, color , 2)

#repeating the same procedure to detect the plate inside the vehicle region
    net_plate.setInput(blob_plate)

    #
    layerOutputs_plate = net_plate.forward(ln_plate)
    boxes_plate = []
    confidences_plate =[]
    classIDs_plate = []

		
    #    
    for output_plate in layerOutputs_plate:
        for detection_plate in output_plate:
            scores_plate = detection_plate[5:]
            classID_plate = np.argmax(scores_plate)
            confidence_plate = scores_plate[classID_plate]

            if confidence_plate > CONFIDENCE_plate:
                box_plate = detection_plate[0:4] * np.array([W,H,W,H])
                (centerX_plate, centerY_plate, width_plate, height_plate) = box.astype("int")

                x_plate = int(centerX_plate - (width_plate/2))
                y_plate = int(centerY_plate - (height_plate/2))

                boxes_plate.append( [x_plate,y_plate,int(width_plate), int(height_plate)])
                confidences_plate.append(float(confidence_plate))
                classIDs_plate.append(classID_plate)

    idxs_plate = cv.dnn.NMSBoxes(boxes_plate, confidences_plate , CONFIDENCE_plate , THRESHOLD_plate)
     	    
    #
	    
    if len(idxs_plate) > 0:
        e=0
        f=-1
        for i in idxs_plate.flatten():
            (x_plate,y_plate) = (boxes_plate[i][0], boxes_plate[i][1])
            (w_plate,h_plate) = (boxes_plate[i][2], boxes_plate[i][3])
            color_plate = [int(c) for c in colours_plate[classIDs_plate[i]]]
            ROI_plate = frame_plate[y_plate:y_plate+h_plate , x_plate:x_plate+w_plate , :]
            #if Labels[classIDs[i]] == 'car':
            cv.rectangle(frame, (x_plate,y_plate), (x_plate + w_plate, y_plate + h_plate), color_plate , 2)
            text_plate = "{}: {:.4f}".format(Labels_plate[classIDs_plate[i]],confidences_plate[i])
            cv.putText(frame , text_plate, (x_plate, y_plate-5) ,  cv.FONT_HERSHEY_SIMPLEX, 0.5, color_plate , 2)


            if ROI_plate is not False:
                if Labels[classIDs_plate[i]] == 'No_Plate':
                    print(y+30,y+h-10,x+10,x+h-30,h*h)
                    cv.imshow('ROI',ROI_plate)
			

    cv.imshow('frame' , frame)
    k=cv.waitKey(1)
    if k ==ord('q'):
        break
    
    
    
    
    if writer is None:
	# initialize our video writer
        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        writer = cv.VideoWriter('output/aj.mp4', fourcc, 20,
            (frame.shape[1], frame.shape[0]), True)
    writer.write(frame) 
    if k==ord('q'):
        break
print("[Message] {} object detected".format(motor_count))
print("The Number is :",a)
cv.destroyAllWindows()
