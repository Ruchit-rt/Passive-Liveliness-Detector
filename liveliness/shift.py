'''*** file was used to shift images around ***'''

from imutils import paths
import cv2
import os
from tqdm import tqdm

imagePaths = list(paths.list_images("/Users/ruchit/Imperial/livelinessDetection/ann_approach/liveliness/dataset/fake/"))
i = 0
for imagePath in tqdm(imagePaths):
    if (i > 500):
        break
    i += 1
    image = cv2.imread(imagePath)
    name = imagePath.split(os.path.sep)[-1]
    cv2.imwrite("/Users/ruchit/Imperial/livelinessDetection/ann_approach/liveliness/jpgVSpng/fake/" + name, image)