''' *** This file uses 2 (3rd detector retina face has been commented out for now) 
	detectors to cut faces from images and store them in a specified directory,
	to avoid cutting faces every time a new model is to be trained ***'''


# set the matplotlib backend so figures can be saved in the background
from turtle import shape
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from pyimagesearch.livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from imutils import paths
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os
import face_recognition as rec
from tqdm import tqdm

# construct the argument parser and parse the arguments
num = "33"

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = "/Users/ruchit/Imperial/livelinessDetection/ann_approach/liveliness/face_detector/deploy.prototxt"
modelPath = "/Users/ruchit/Imperial/livelinessDetection/ann_approach/liveliness/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-4
BS = 64
EPOCHS = 1000
DEFAULT_CONFIDENCE = 0.5
# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images("/Users/ruchit/Imperial/livelinessDetection/ann_approach/pipeline_test/"))
data = []
labels = []
# loop over all image paths
for imagePath in tqdm(imagePaths):
	# extract the class label from the filename, load the image and
	# resize it to be a fixed 32x32 pixels, ignoring aspect ratio
	split = imagePath.split(os.path.sep)
	label = split[-2]
	name = split[-1]
	image = cv2.imread(imagePath)

	# grab the frame dimensions and construct a blob from the frame
	(h, w) = image.shape[:2]
	#TODO: make 224
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0, (300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# ensure at least one face was found
	if len(detections) > 0:
		# we're making the assumption that each image has only ONE
		# face, so find the bounding box with the largest probability
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# ensure that the detection with the largest probability also
		# means our minimum probability test (thus helping filter out
		# weak detections
		if confidence > DEFAULT_CONFIDENCE:
    		# compute the (x, y)-coordinates of the bounding box for
			# the face and extract the face ROI
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			if (endX > w or endY > h):
				img = rec.load_image_file(imagePath)
				face_locations = rec.face_locations(img)

				if (len(face_locations) == 0):
					print("out of bounds by first, no detections found by second detector")
					continue
    	      		#applying third detector
					# resp = RetinaFace.detect_faces(image)
					# if (len(resp) == 0):
					# 	print("out of bounds by first, no detections found by other detectors")
					# 	continue  
					# conf = 0
					# faceRect = []
					# for item in resp:
					# 	response = resp[item]
					# 	if (response['score'] > conf):
					# 		conf = response['score']
					# 		faceRect = response['facial_area']
					# if (conf > DEFAULT_CONFIDENCE):
					# 	startX, startY, endX, endY = faceRect
					# else: 
					# 	print("out of bounds by first, no detections found by other detectors")
					# 	continue  
				else:
					startY, endX, endY, startX = face_locations[0]
		else:
			# confidence was low
			img = rec.load_image_file(imagePath)
			face_locations = rec.face_locations(img)
			if (len(face_locations) == 0):
				print("out of bounds by first, no detections found by other detectors")
				continue
				#applying third detector
				# resp = RetinaFace.detect_faces(image)
				# if (len(resp) == 0):
				# 	print("out of bounds by first, no detections found by other detectors")
				# 	continue  
				# conf = 0
				# faceRect = []
				# for item in resp:
				# 	response = resp[item]
				# 	if (response['score'] > conf):
				# 		conf = response['score']
				# 		faceRect = response['facial_area']

				# if (conf > DEFAULT_CONFIDENCE):
				# 	startX, startY, endX, endY = faceRect
				# else:
				# 	print("out of bounds by first, no detections found by other detectors")
				# 	continue
			else:
				startY, endX, endY, startX = face_locations[0]
	else:
		#no detection by first
		img = rec.load_image_file(imagePath)
		face_locations = rec.face_locations(img)
		if (len(face_locations) == 0):
			print("no detections")
			continue
			# #applying third detector
			# resp = RetinaFace.detect_faces(image)
			# if (len(resp) == 0):
			# 	print("no detections")
			# 	continue  
			# conf = 0
			# faceRect = []
			# for item in resp:
			# 	response = resp[item]
			# 	if (response['score'] > conf):
			# 		conf = response['score']
			# 		faceRect = response['facial_area']

			# if (conf > DEFAULT_CONFIDENCE):
			# 	startX, startY, endX, endY = faceRect
			# else:
			# 	print("no detections")
			# 	continue
		else:
			startY, endX, endY, startX = face_locations[0]

	#grab face ROI
	face = image[startY:endY, startX:endX]
	
	# update the data and labels lists, respectively
	shape = face.shape
	if (shape[0] > 0 and shape[1] > 0):
		face = cv2.resize(face, (32,32))
		cv2.imwrite("/Users/ruchit/Imperial/livelinessDetection/ann_approach/pipeline_test_faces/" + name, face)