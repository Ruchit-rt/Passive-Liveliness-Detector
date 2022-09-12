'''*** Backup copy of the testing model file (has not been cleaned) ***'''

# import the necessary packages
from pyparsing import original_text_for
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import cv2
import glob

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", type=str, required=True,
# 	help="path to trained model")
# ap.add_argument("-l", "--le", type=str, required=True,
# 	help="path to label encoder")
# args = vars(ap.parse_args())

print("[INFO] loading face detector...")
protoPath = "/Users/ruchit/Imperial/livelinessDetection/ann_approach/livenet/livenessnet/face_detector/deploy.prototxt"
modelPath = "/Users/ruchit/Imperial/livelinessDetection/ann_approach/livenet/livenessnet/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

DEFAULT_CONFIDENCE = 0.6

# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")
model = load_model("liveness.model25")
le = pickle.loads(open("le25.pickle", "rb").read())

tp = "real"
imgPaths = "/Users/ruchit/Imperial/livelinessDetection/ann_approach/8Aug/08-Aug-22/*"
# imgPaths = "/Users/ruchit/Imperial/livelinessDetection/ann_approach/livenet/livenessnet/test_data_4Aug/"+tp+"/*"
count = 0
total = 0
poor_count = 0
#---
image = cv2.imread("/Users/ruchit/Imperial/livelinessDetection/ann_approach/livenet/livenessnet/faceCheck.png")
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()
for i in range(0, detections.shape[2]):
	confidence = detections[0, 0, i, 2]
	if (confidence > 0.15):		
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		image = cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 3)

cv2.imshow("image", image)

while True:	
	key = cv2.waitKey(33)
	if (key == ord('s')): 
		cv2.destroyAllWindows()
		break

#---
for path in glob.glob(imgPaths):
	image = cv2.imread(path)
	original_image = cv2.imread(path)
	# grab the frame dimensions and construct a blob from the frame
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(original_image, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	#------------------------

	if len(detections) == 0:
		print("No detections")
		continue
	
	last_confidence = DEFAULT_CONFIDENCE
	for i in range(0, detections.shape[2]):
		poor_detection = False
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]
		# filter out weak detections
		if confidence > last_confidence:
			last_confidence = confidence
			# compute the (x, y)-coordinates of the bounding box for
			# the face and extract the face ROI
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# if (path.split("/")[-1] == "SgNe05_01-08-2022_05-05-44_file2022-08-01_050536.179373.png"):
			# 	print("startX = ", startX)
			# 	print("startX = ", startY)
			# 	print("endX = ", endX)
			# 	print("endY = ", endY)
			# 	while True:	
			# 		key = cv2.waitKey(33)
			# 		if (key == ord('s')):
			# 			break
			# ensure the detected bounding box does fall outside the
			# dimensions of the frame
			startX = max(0, startX)
			startY = max(0, startY)
			if (endX > w or endY > h):
				poor_detection = True
			endX = min(w, endX)
			endY = min(h, endY)
			image = cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 3)
			# cv2.putText(image, str(confidence), (startX, startY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
			# extract the face ROI and then preproces it in the exact
			# same manner as our training data
			if (poor_detection):
				print("poor_detection")
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

	face = face.astype("float") / 255.0
	face = img_to_array(face)
	face = np.expand_dims(face, axis=0)
    # pass the face ROI through the trained liveness detector
    # model to determine if the face is "real" or "fake"
	preds = model.predict(face)
	j = np.argmax(preds)
	label = le.classes_[j]
	total += 1
	print(preds)
	if (label == tp):
		count += 1
	else:
		cv2.imshow(path, orig)
		#show preds description on images
		real_ratio = preds[0][1] / preds[0][0]
		cv2.putText(image, "Real probs:" + str(preds[0][1]), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 4)
		cv2.putText(image, "Fake probs:" + str(preds[0][0]), (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,0,0), 4)
		cv2.putText(image, "Ratio:" + str(real_ratio), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 4)
		cv2.imshow("original image", image)
		print("-----------------")
		print(path[-7:] + " : " + label)
		print(preds)
		print("-----------------")
		while True:	
			key = cv2.waitKey(33)
			if (key == ord('s')): 
				cv2.destroyAllWindows()
				break
		# name = path.split("/")[-1]
		# cv2.imwrite("/Users/ruchit/Imperial/livelinessDetection/ann_approach/livenet/livenessnet/fail_data_25/"+tp+"/"+ name , original_image)
		# name = path.split("/")[-1]
	# 	# cv2.imwrite("/Users/ruchit/Imperial/livelinessDetection/ann_approach/livenet/livenessnet/fail_data_nokia/fake/" + name , image)
		# while True:	
		# 	key = cv2.waitKey(33)
		# 	if (key == ord('s')): 
		# 		cv2.destroyAllWindows()
		# 		break
		##save fail cases##
		# name = path.split("/")[-1]
		# cv2.imwrite("/Users/ruchit/Imperial/livelinessDetection/ann_approach/livenet/livenessnet/fail_data_25/"+tp+"/"+ name , original_image)

print("accuracy percent: ", (count/total)*100)
print("poor_detections: ", poor_count)
	#------------------------
'''
	# ensure at least one face was found
	if len(detections) > 0:
		# we're making the assumption that each image has only ONE
		# face, so find the bounding box with the largest probability
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]
	else:
		print("no detection found")

	# ensure that the detection with the largest probability also
		# means our minimum probability test (thus helping filter out
		# weak detections)
	if confidence > DEFAULT_CONFIDENCE:
		# compute the (x, y)-coordinates of the bounding box for
		# the face and extract the face ROI
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		face = image[startY:endY, startX:endX]
	else: 
		print("confidence was low")
		continue
	shape = face.shape
	if (shape[0] == 0 or shape[1] == 0 or shape[2] == 0):
		print("skipping")
		continue
''' 