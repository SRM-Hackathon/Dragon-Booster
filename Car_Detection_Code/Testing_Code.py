# import the necessary packages
import cv2 as cv
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

#choosing the trained data storage path 
weights_path = "yolo/sairam_training/yolov3-tiny_70800.weights"                    ##location of files in my computer##
config_path = "yolo/sairam_training/yolov3-tiny.cfg"
label_path = "yolo/sairam_training/obj.names"


#loading number of lables
Labels = open(label_path).read().strip().split("\n")
motor_count = 0


#Randomely choosing colors depending on number of lables 
np.random.seed(42)
colours = np.random.randint(0,255, size = (len(Labels),3), dtype = "uint8")

#Loading the yolo detector from path
print("[Message]Loading the yolo detector")
net = cv.dnn.readNetFromDarknet(config_path,weights_path)
ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

#Imput video to be predicted
cap = cv.VideoCapture('videos/car.mp4')
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

#prediction begins
while True:
    
    ret, frame = cap.read()      #seperating frames from the video
    count += 2
    cap.set(1,count)
    
    if not ret:
        break
    
    if W is None or H is None:
        (H,W) = frame.shape[:2]
	
        
    # forward the blob to the network
    blob = cv.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB = True, crop = False)
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
            if Labels[classIDs[i]] == 'car':                                    #font style ,colour of bounding box character of car can we changed here
                cv.rectangle(frame, (x,y), (x + w, y + h), color , 2)
                text = "{}: {:.4f}".format(Labels[classIDs[i]],confidences[i])
                cv.putText(frame , text, (x, y-5) ,  cv.FONT_HERSHEY_SIMPLEX, 0.5, color , 2)
            if Labels[classIDs[i]] == 'No_Plate' and h<37 and h>35:             #if number plate found Character Recognisation begins
                writer.write(ROI)                                        #cutting the plate from image
                ROI = frame[y+30:y+h -10, x+10:x+w -30, :]
                cv.rectangle(frame, (x,y), (x + w, y + h), color , 2)
                text = "{}: {:.4f}".format(Labels[classIDs[i]],confidences[i])
                cv.putText(frame , text, (x, y-5) ,  cv.FONT_HERSHEY_SIMPLEX, 0.5, color , 2)
                cv.imwrite('a.png',ROI)
                print(ROI)
                gray = cv.imread('a.png', 0)
                preprocess = "thresh"                                   #applaying Tesseract After Thresh Preprocessing in OpenCV
                threshold =plate_threshold
                #cv.imshow("Image", gray)
                if preprocess == "thresh":
                    gray = cv.threshold(gray, 0, 255,cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
                    gray = cv.medianBlur(gray, 3)
                elif preprocess=="blur":
                    gray = cv.medianBlur(gray, 3)
                    gray = cv.threshold(gray, 0, 255,cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
                    gray = cv.medianBlur(gray, 3)

                filename="{}.png".format(os.getpid())
                cv.imwrite(filename, gray)

                text = pytesseract.image_to_string(Image.open(filename))     #foung plate number is stored here
                os.remove(filename)

                cv.waitKey(5000)

		
            motor_count =motor_count + 1
            #ROI = frame[y+30:y+h -10, x+10:x+w -30, :]

	    
            if ROI is not False:
                if Labels[classIDs[i]] == 'No_Plate' and h<37 and h>35:
                    print(y+30,y+h-10,x+10,x+h-30,h*h)
                    cv.imshow('ROI',ROI)
			
    #usefull utility function to control the process when "Q" pressed the program terminated
    cv.imshow('frame' , frame)
    k=cv.waitKey(1)
    if k ==ord('q'):
        break
    
    
    
    
    if writer is None:              
        fourcc = cv.VideoWriter_fourcc(*"MJPG")          	# initialize our video writer

        writer = cv.VideoWriter('output/aj.mp4', fourcc, 20,
            (frame.shape[1], frame.shape[0]), True)
    writer.write(frame) 
    if k==ord('q'):
        break
print("[Message] {} object detected".format(motor_count))
print("The Number is :",a)
cv.destroyAllWindows()                    #closing all open windows
