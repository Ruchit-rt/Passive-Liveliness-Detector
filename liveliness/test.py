'''*** This file is used test models developed by train files 
		and then print final results 
		1) It gets images from testing directory
		2) Cuts face ROI's using 2 detectors
		3) Uses model to predict label
		4) Finally store/display result
		***'''

# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import pickle
import cv2
import glob
import face_recognition as rec
from tqdm import tqdm

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="31",
	help="path to trained model")
args = vars(ap.parse_args())

print("[INFO] loading face detector...")
protoPath = "/Users/ruchit/Imperial/livelinessDetection/ann_approach/liveliness/face_detector/deploy.prototxt"
modelPath = "/Users/ruchit/Imperial/livelinessDetection/ann_approach/liveliness/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

DEFAULT_CONFIDENCE = 0.5
THRESHOLD = 75

# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")
model = load_model("liveness.model40")
le = pickle.loads(open("le40.pickle", "rb").read())

# this tp refers to the type of images I am testing - Real or Fake
tp = "fake"

#setting counters and image path
imgPaths = "/Users/ruchit/Imperial/livelinessDetection/ann_approach/real-fake photo_datasets/16Aug/"+tp+"/*"
count = 0
not_sure_count = 0
total = 0
poor_count = 0

for path in tqdm(glob.glob(imgPaths)):
	poor_detection = False
	image = cv2.imread(path)
	original_image = cv2.imread(path)
	# grab the frame dimensions and construct a blob from the frame
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(original_image, (224, 224)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	#------------------------

	if len(detections) == 0:
		img = rec.load_image_file(path)
		face_locations = rec.face_locations(img)
		
		if (len(face_locations) == 0):
			poor_detection = True
			continue
		else:
			startY, endX, endY, startX = face_locations[0]
	else:
		# extract the confidence (i.e., probability) associated with the
		# prediction
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]
		# filter out weak detections
		if confidence > DEFAULT_CONFIDENCE:
			# compute the (x, y)-coordinates of the bounding box for
			# the face and extract the face ROI
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			startX = max(0, startX)
			startY = max(0, startY)
			if (endX > w or endY > h):
				img = rec.load_image_file(path)
				face_locations = rec.face_locations(img)

				if (len(face_locations) == 0):
					poor_detection = True
					continue
				else:
					startY, endX, endY, startX = face_locations[0]
		else:
			img = rec.load_image_file(path)
			face_locations = rec.face_locations(img)
			if (len(face_locations) == 0):
				poor_detection = True
				continue
			else:
				startY, endX, endY, startX = face_locations[0]

	# extract the face ROI and then preproces it in the exact
	# same manner as our training data
	if (poor_detection):
		poor_count += 1
		continue

	face = image[startY:endY, startX:endX]
	shape = face.shape
	if (shape[0] == 0 or shape[1] == 0 or shape[2] == 0):
		print("skipping")
		continue

	orig = face			
	try:
		face = cv2.resize(face, (32,32))
	except: 
		print("Unexpected shape: ", face.shape)
		poor_count += 1
		continue
	orig_face = face
	face = face.astype("float") / 255.0
	face = img_to_array(face)
	face = np.expand_dims(face, axis=0)
    # pass the face ROI through the trained liveness detector
    # model to determine if the face is "real" or "fake"
	preds = model.predict(face)
	j = np.argmax(preds)
	label = le.classes_[j]
	# label = classes[j]
	fake_ratio = preds[0][0] / preds[0][1]
	total += 1
	# for fake case: 
	if (tp == "fake"):
		if fake_ratio > THRESHOLD:
			#we were sure it was fake
			count += 1
		else:
			name = path.split("/")[-1]
			print("--------------")
			# cv2.imshow(" ", original_image)
			print(fake_ratio)
			if (label == tp):
				# fake with ratio between 1-THRESHOLD
				print("NOT SURE")
				not_sure_count += 1
				#to save or not to save
				# if (fake_ratio < 30):
					# cv2.imwrite("/Users/ruchit/Imperial/livelinessDetection/ann_approach/livenet/livenessnet/fail_data_29/"+tp+"/"+ name , original_image)
			else:
				# cv2.imwrite("/Users/ruchit/Imperial/livelinessDetection/ann_approach/livenet/livenessnet/fail_data_29/"+tp+"/"+ name , original_image)
				print("REAL: ", preds)
				print(path.split('/')[-1] + ":" + label)
			print("-----------------")	

			# this code can be used to view the fail cases individually on waitkey		
			# while True:	
			# 	key = cv2.waitKey(33)
			# 	if (key == ord('q')): 
			# 		break
			# 	if (key == ord('s')):
			# 		name = path.split("/")[-1]
			# 		cv2.imwrite("/Users/ruchit/Imperial/livelinessDetection/ann_approach/livenet/livenessnet/fail_data_27/"+tp+"/"+ name , original_image)
			# 		break
			# cv2.destroyAllWindows()


	# for real case 
	if (tp == "real"):
		if (label == tp):
			count += 1
		else:
			print("-----------------")
			print(fake_ratio)
			if fake_ratio <= THRESHOLD:
				print("NOT SURE")
				not_sure_count += 1
			else:
				print("FAKE: ", preds)
				print(path.split('/')[-1] + ":" + label)

				# --- for viewing fail cases ---
				# cv2.imshow(" ", original_image)
				# while True:	
				# 	key = cv2.waitKey(33)
				# 	if (key == ord('q')): 
				# 		break
				# 	if (key == ord('s')):
				# 		name = path.split("/")[-1]
				# 		cv2.imwrite("/Users/ruchit/Imperial/livelinessDetection/ann_approach/livenet/livenessnet/fail_data_34/"+tp+"/"+ name , original_image)
				# 		break

			cv2.destroyAllWindows()
			print("-----------------")			
			##save fail cases##
			# name = path.split("/")[-1]
			# # cv2.imwrite("/Users/ruchit/Imperial/livelinessDetection/ann_approach/livenet/livenessnet/fail_data_27/"+tp+"/"+ name , original_image)

#printing final results			
if (tp == "real"):
	print("accuracy percent: ", (count/total)*100)
	print("not sure percent: ", (not_sure_count/total) * 100)
	print("total accuracy (after threshold): " ,((count + not_sure_count)/total)*100)
	print("poor_detections: ", poor_count)
else:
	print("accuracy percent: ", (count/total)*100)
	print("not sure percent: ", (not_sure_count/total) * 100)
	print("poor_detections: ", poor_count)