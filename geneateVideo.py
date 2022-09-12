import cv2 as cv
import glob


#this file makes a video of real images at 1 FPS


imagePath = '/Users/ruchit/Imperial/livelinessDetection/ann_approach/16Aug/real_jpg/*'

f = open("file_order_16Aug.txt", "w")
file_names = []

img_array = []
count = 0
for file in glob.glob(imagePath):
    img = cv.imread(file)
    height, width, layers = img.shape
    if (width == 720 and height == 1280):
        file_names.append(file.split("/")[-1] + "\n")
        shape = (width, height)
        img_array.append(img)
        count += 1
    else: 
        print(file)

out = cv.VideoWriter('16Aug_real.mp4', cv.VideoWriter_fourcc(*'DIVX'), 1, shape)

for i in range(count):
    out.write(img_array[i])

f.writelines(file_names)
f.close()
out.release()