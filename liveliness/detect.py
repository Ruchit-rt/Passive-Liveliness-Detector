'''*** Main file with necessary functions for liveliness detection pipeline ***'''

# import the necessary packages
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from datetime import date
# from uuid import uuid4
import numpy as np
import json
import pickle
import cv2
import requests
import argparse
import base64
import os

#defining constants for code
MX_REAL = 0.9
MX_FAKE = 0.5
THRESHOLD = 75
MX_MAX_CALLS = 15

#defining storage path locations
save_path = os.environ['SAVEPATH']

MODEL_PATH = "/Users/ruchit/Imperial/livelinessDetection/ann_approach/liveliness"

FAIL_FAKE_PATH = save_path + "/fake/fail_fake/"
NOT_SURE_FAKE_PATH = save_path + "/fake/not_sure/"
VERIFIED_FAKE_PATH = save_path + "/fake/verified_fake/"

NOT_SURE_REAL = save_path + "/not_sure/real/"
NOT_SURE_FAKE = save_path + "/not_sure/fake/"

path_to_file = save_path + "/count.txt"

#function call MxFace API
def mxface(img):
    # print(img)
    requestHeader = { "Subscriptionkey" : "LGXDx027vWj9vxFTPb-naWeRW6CVi980","Content-Type" : "application/json" }    
    myobj = {"encoded_image" : img}

    response = requests.post('https://faceapi.mxface.ai/api/v2/face/CheckLiveness', json = myobj,headers=requestHeader)
    # json response was successful
    if (response.status_code == 200):
        data = json.loads(response)
        print("Mxface was called successfully: ", data)
        return data

    # return negative values for unsuccessful run
    else:
        data = {}
        data['confidence'] = -2          #-2 indicates a failed call to Mx api
        data['quality'] = -2
        print("MXFace api failed call: ",  response)
        return data


def space(s):
    return (str(s) + " ")

def line(s):
    return (str(s) + '\n')

def check_dir():
    #check dir structure at save_path
    os.makedirs(save_path+'/fake/fail_fake/', exist_ok=True)
    os.makedirs(save_path+'/fake/not_sure/', exist_ok=True)
    os.makedirs(save_path+'/fake/verified_fake/', exist_ok=True)
    os.makedirs(save_path+'/not_sure/fake/', exist_ok=True)
    os.makedirs(save_path+'/not_sure/real/', exist_ok=True)
    return True

def get_counter():

    #check if environment variable exists
    if "MxfaceApiCalls" in os.environ:
        count = os.environ["MxfaceApiCalls"]
        return int(count)

    #read the count file as environment varaible is not set
    if (os.path.exists(path_to_file)):
        with open(path_to_file, "r+") as file:
            lines = file.readlines()
            found = False
            count = 0
            for line in lines:
                split = line.split(" ")
                if str(date.today()) == split[0]:
                    found = True
                    count = split[1][:-1]
            
            if (found):
                os.environ["MxfaceApiCalls"] = count
                return int(count)
            else:
                file.write(str(date.today()) + " 0\n")
                os.environ["MxfaceApiCalls"] = "0"
                return 0
    else:
        with open(path_to_file, "a") as file:
            file.write(str(date.today()) + " 0\n")
            os.environ["MxfaceApiCalls"] = "0"
            return 0

# update the config gile by incremeanting the counter next to today's date
def update_file():
    lines = []
    with open(path_to_file, "r") as file:
        lines = file.readlines() 
    with open(path_to_file, "w") as file:
        found = False
        for line in lines: 
            split = line.split(" ")
            if (split[0] == str(date.today())):
                found = True
                i = int(split[1][:-1])
                i += 1
                split[1] = str(i) + '\n'
                line = split[0] + " " + split[1]
            file.write(line)

        if (not found):
            raise Exception("Unable to find date in config file")

def saver(img, name, root_path, data):
    #open metadata file 
    meta = open(root_path + "metadata.csv", "a")
    out = [space(name), space(data[0]), space(data[1]), space(data[2]), line(data[3])]
    cv2.imwrite(root_path + name, img)
    meta.writelines(out)
    meta.close()
    print("saved image at: ", root_path + name)

# Param1: this function accepts a Base64 bytestring as input 
# that it converts to image for processing 
# Param2: name of the file for storage
def is_real(img_arr, img_name):
    # fetch image to process
    nparr = np.fromstring(base64.b64decode(img_arr), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    face = cv2.resize(image, (32,32))
    
    # load model and le
    model_num = "40"
    model = load_model(MODEL_PATH + "/liveness.model" + model_num)
    le = pickle.loads(open(MODEL_PATH + "/le"+ model_num +".pickle", "rb").read())

    # predict on face
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    preds = model.predict(face)
    j = np.argmax(preds)
    label = le.classes_[j]
    fake_ratio = preds[0][0] / preds[0][1]

    #generating unique name from event id
    # name = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4()) + ".jpg"
    name = img_name
    
    #if model says real no action tbd
    if (label == "real"): 
        print("--- REAL DETECTED ---")
        return "real,model_verified"

    # my model predicts "fake"
    if (label == "fake" and fake_ratio > THRESHOLD):
        print("--- CONFIDENT FAKE DETECTED ---")

        #before calling MXFACE lets see counter for the day
        count = get_counter()
        print(count)
        if (count > MX_MAX_CALLS):
            data = [preds[0][1], preds[0][0], -1, -1]    #-1 indicates MX not called
            saver(image, name, FAIL_FAKE_PATH, data)
            return "fake,unverified"
        else:
            #calling and confirming with MxFace API
            response = mxface(str(img_arr))
            conf = response['confidence']
            qual = response['quality']
            data = [preds[0][1], preds[0][0], conf, qual]
            #update config file
            if (data[2] > 0):
                update_file()

            if (data[2] == -2):
                #failed mx call means unverified fake
                saver(image, name, NOT_SURE_FAKE_PATH, data)
                return "fake,unverified"

            if conf > MX_REAL:
                #mxface thinks its a real photo, we failed
                saver(image, name, FAIL_FAKE_PATH, data)
                return "real,mx_verified"
            elif conf < MX_FAKE:
                #mxface supports us, verified fake
                saver(image, name, VERIFIED_FAKE_PATH, data)
                return "fake,mx_verified"
            else:
                #mxface also not sure but we think its fake
                saver(image, name, NOT_SURE_FAKE_PATH, data)
                return "real,mx_not_sure"

    # my model predicts "not_sure"
    if (label == "fake" and fake_ratio <= THRESHOLD):
        print("--- NOT SURE FAKE DETECTED ---")
        #calling and confirming with MxFace API
        # NOTE: I am not calling MXFACE when my model says not sure as
        #       we are letting them pass as innocent
        #       This will also reduce mxface API calls
        # response = mxface(img_arr)
        conf = -1                 # negative value indicates 
        qual = -1                 # MX is not called
        data = [preds[0][1], preds[0][0], conf, qual]
        saver(image, name, NOT_SURE_FAKE, data)
        return "real,model_not_sure"          #---> let not_sure folks pass by as innocent
    
if (__name__ == '__main__'):
    # argumnets parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, required=True,
    	help="image to process (byte array)")
    ap.add_argument("-n", "--name", type=str, required=True,
    	help="image name for storage")
    args = vars(ap.parse_args())

    ret = is_real(args["image"], args["name"])
    print("Value returned: ", ret)
    print("EXITING...")

