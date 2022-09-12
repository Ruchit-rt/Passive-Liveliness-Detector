import cv2
import time

vid = cv2.VideoCapture("/Users/ruchit/Imperial/livelinessDetection/ann_approach/16Aug/VID20220822142121.mp4")

if (vid.isOpened()== False): 
  print("Error opening video  file")

#intialise params
count = 0

# Read until video is completed
while(vid.isOpened()):
      
  # Capture frame-by-frame
  ret, frame = vid.read()
  if ret == True:
    count += 1
    cv2.imshow(str(count), frame)
    time.sleep(0.7)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  else: 
    break
   
vid.release() 
cv2.destroyAllWindows()