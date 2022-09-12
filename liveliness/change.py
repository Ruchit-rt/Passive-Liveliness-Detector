'''This file was used to change png images to jpg images'''


from imutils import paths
import cv2
import os
from tqdm import tqdm

imagePaths = list(paths.list_images("/Users/ruchit/Imperial/livelinessDetection/ann_approach/liveliness/jpgVSpng/png/fake/"))
for imagePath in tqdm(imagePaths):
    image = cv2.imread(imagePath)
    name = imagePath.split(os.path.sep)[-1]
    new_name = name[:-3] + "jpg"
    cv2.imwrite("/Users/ruchit/Imperial/livelinessDetection/ann_approach/liveliness/jpgVSpng/jpg/fake/" + new_name, image)