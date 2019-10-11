from imutils.video import VideoStream
import argparse
import imutils
import cv2 as cv
import numpy as np
import time
import sys
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LoadCredentialsFile("mycreds.txt")
drive = GoogleDrive(gauth)

CONFIDENCE = 0.5
con = 0.3
THRESHOLD = 0.3
weights_path = "yolo/yolov3.weights"
config_path = "yolo/yolov3.cfg"
label_path = "yolo/coco.names"
Labels = open(label_path).read().strip().split("\n")
motor_count = 0
keep =0

#colors
np.random.seed(42)
colours = np.random.randint(0,255, size = (len(Labels),3), dtype = "uint8")

print("[Message]Loading the yolo detector")
net1 = cv.dnn.readNetFromDarknet(config_path,weights_path)
ln = net1.getLayerNames()
ln = [ln[i[0]-1] for i in net1.getUnconnectedOutLayers()]


cap = cv.VideoCapture('videos/helmet.mp4')
writer = None
(W,H) = (None,None)
count =0

try:
	length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
	print("[Message] {} total number of frames in the video".format(length)) 
	
except:
	print("[Message] Could'nt determine the number of frames in the video")
	total = -1

while True:
	ret, frame = cap.read()
	count += 10
	#cap.set(1,count)
	
	if not ret:
		break
	
	if W is None or H is None:
		(H,W) = frame.shape[:2]
		print(frame.shape)
	
	blob = cv.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB = True, crop = False)
	# forward the blob to the network
	net1.setInput(blob)
	
	
	layerOutputs = net1.forward(ln)
	boxes = []
	confidences =[]
	classIDs = []
	
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
			
	if len(idxs) > 0:
		for i in idxs.flatten():
			(x,y) = (boxes[i][0], boxes[i][1])
			(w,h) = (boxes[i][2], boxes[i][3])
			color = [int(c) for c in colours[classIDs[i]]]
			if Labels[classIDs[i]] == 'motorbike':
				cv.rectangle(frame, (x-50,y-70), (x + w+50, y +h+50), color , 2)
				text = "{}: {:.4f}".format(Labels[classIDs[i]],confidences[i])
				#cv.putText(frame , text, (x, y-5) ,  cv.FONT_HERSHEY_SIMPLEX, 0.5, color , 2)
					
				
				prototxt = 'deploy.prototxt.txt'
				model ='model.caffemodel'
				CONFIDENCE = 0.5


				# load our serialized model from disk
				print("[INFO] loading model...")
				net = cv.dnn.readNetFromCaffe(prototxt,model)
				


				while True:
					
					#print('hello')
					#frame = imutils.rotate(frame , -50)
					frame = imutils.resize(frame, width=400)
				 
					(h, w) = frame.shape[:2]
					blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0,
						(300, 300), (104.0, 177.0, 123.0))
				 
					
					net.setInput(blob)
					detections = net.forward()

					# loop over the detections
					for i in range(0, detections.shape[2]):
						# extract the confidence (i.e., probability) associated with the
						# prediction
						confidence = detections[0, 0, i, 2]
						print('detecting violator')

						# filter out weak detections by ensuring the `confidence` is
						# greater than the minimum confidence
						if confidence < con:
							continue
							
						cv.imwrite("{}.jpg".format(keep),frame)
						textfile = drive.CreateFile()
						textfile.SetContentFile('{}.jpg'.format(keep))
						textfile.Upload()
						print("Picture pushed to the drive successfully");
						#location = sys.path[0]+'/{}.jpg'.format(keep)
						#print(location)
						
					
						keep = keep + 1
							
						box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
						(startX, startY, endX, endY) = box.astype("int")
						text = "{:.2f}%".format(confidence * 100)
						y = startY - 10 if startY - 10 > 10 else startY + 10
						cv.rectangle(frame, (startX, startY), (endX, endY),
							(0, 0, 255), 2)
						cv.putText(frame, 'helmet violator', (startX, y),
							cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
														
					cv.imshow('frame',frame)
					cv.waitKey(1)
					break	
					
		
	if writer is None:
		# initialize our video writer
		fourcc = cv.VideoWriter_fourcc(*"MJPG")
		writer = cv.VideoWriter('output/helme.mp4', fourcc, 20,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
	'''if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))'''

	# write the output frame to disk
	writer.write(frame)
	'''if ROI is not False:
		cv.imshow('ROI',ROI)
		cv.waitKey(1000)
	#end = time.time()
	#elapse = 1000*(end - start)
	#print(elapse)
	
	if k==ord('q'):
		break'''
#print("[Message] {} Motor bikes detected".format(motor_count))

cv.destroyAllWindows()
