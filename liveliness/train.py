'''***
	This file basically does training from scratch. 
	1) Grabs face ROI's from dataset of images and append to data [], label []
	2) Train a model 
	3) Print report and save model, label_encoder (for one hot encoding), plot 
	***'''

# set the matplotlib backend so figures can be saved in the background
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
from tensorflow.keras.models import load_model
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os
import face_recognition as rec
from tqdm import tqdm

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--number",
	help="training model number")
args = vars(ap.parse_args())
args["number"] = "48"

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = "/Users/ruchit/Imperial/livelinessDetection/ann_approach/liveliness/face_detector/deploy.prototxt"
modelPath = "/Users/ruchit/Imperial/livelinessDetection/ann_approach/liveliness/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-4
BS = 256
EPOCHS = 1000
DEFAULT_CONFIDENCE = 0.5
# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images("/Users/ruchit/Imperial/livelinessDetection/ann_approach/liveliness/dataset_copies/data"))
data = []
labels = []
# loop over all image paths
for imagePath in tqdm(imagePaths):
	# extract the class label from the filename, load the image and
	# resize it to be a fixed 32x32 pixels, ignoring aspect ratio
	split = imagePath.split(os.path.sep)
	label = split[-2]
	image = cv2.imread(imagePath)

	# grab the frame dimensions and construct a blob from the frame
	(h, w) = image.shape[:2]
	#TODO: make 224
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

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
					# cv2.imwrite("/Users/ruchit/Imperial/livelinessDetection/ann_approach/livenet/livenessnet/out_of_bounds/" + split[-1], image)
					continue
				else:
					startY, endX, endY, startX = face_locations[0]
		else:
			# confidence was low
			img = rec.load_image_file(imagePath)
			face_locations = rec.face_locations(img)
			if (len(face_locations) == 0):
				print("confidence-low by first, no detections found by second detector")
				# cv2.imwrite("/Users/ruchit/Imperial/livelinessDetection/ann_approach/livenet/livenessnet/low_confidence/" + split[-1], image)
				continue
			else:
				startY, endX, endY, startX = face_locations[0]
	else:
		#no detection by first
		img = rec.load_image_file(imagePath)
		face_locations = rec.face_locations(img)
		if (len(face_locations) == 0):
			print("no detection")
			# cv2.imwrite("/Users/ruchit/Imperial/livelinessDetection/ann_approach/livenet/livenessnet/no_detect/" + split[-1], image)
			continue
		else:
			startY, endX, endY, startX = face_locations[0]

	#grab face ROI
	face = image[startY:endY, startX:endX]

	# checking faces correct
	# cv2.imshow("face", face)
	# while True: 
	# 	if cv2.waitKey(33) == ord('q'):
	# 		cv2.destroyAllWindows
	# 		break
	
	# update the data and labels lists, respectively
	shape = face.shape
	if (shape[0] > 0 and shape[1] > 0):
		face = cv2.resize(face, (32,32))
		data.append(face)
		labels.append(label)
	
    
# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0

# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, 2)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")
# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = LivenessNet.build(width=32, height=32, depth=3,
	classes=len(le.classes_))
# model = load_model("liveness.model29")
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the network
print("[INFO] training network for {} epochs...".format(EPOCHS))
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))
# save the network to disk
print("[INFO] serializing network to '{}'...".format("liveness.model" + args["number"]))
model.save("liveness.model" + args["number"], save_format="h5")
# save the label encoder to disk
f = open("le" + args["number"] + ".pickle", "wb")
f.write(pickle.dumps(le))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot" + args["number"] + ".png")

#--- TODO: write code to savemodel, plot, le to drive