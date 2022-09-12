'''*** This file is used to test pipeline for liveliness detection 
        as established by the detect.py file ***'''

from detect import is_real, check_dir
from imutils import paths
import base64

# imagePaths = list(paths.list_images("/Users/ruchit/Imperial/livelinessDetection/ann_approach/pipeline_test_faces/"))
imagePaths = list(paths.list_images("/Users/ruchit/Imperial/livelinessDetection/ann_approach/pipeline_test/"))

#test dir structure at storage path
if (not check_dir()):
    raise Exception("Directory structure was not set properly")

for imagePath in imagePaths:
    with open(imagePath, "rb") as image:
        print("")
        encoded_string = base64.b64encode(image.read())
        split = imagePath.split('/')
        ret = is_real(encoded_string, split[-1])
        label = split[-2]
        pred = ret.split(',')[0]
        if (pred == label):
            print("Correct --> ", split[-1], ":", ret)
        else: 
            print("Failed --> ", split[-1], ":", ret)
        print("")

print("Succesful testing!")