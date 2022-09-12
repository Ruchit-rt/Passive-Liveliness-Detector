import cv2

#reject initial number of frames
REJECT_FRAME_TIME = 5
#fps of capture device
FPS = 30

vid = cv2.VideoCapture("/Users/ruchit/Imperial/livelinessDetection/ann_approach/16Aug/VID20220822142121.mp4")

if (vid.isOpened()== False): 
  print("Error opening video  file")

#intialise params
count = 0
frame_num = 1
MAX_FRAMES = 416

#open file order
file_order = open("file_order_16Aug.txt", "r")
names = file_order.readlines()

# Read until video is completed
while(vid.isOpened() and frame_num < MAX_FRAMES):
      
  # Capture frame-by-frame
  ret, frame = vid.read()
  if ret == True:
    count += 1

    if (count < REJECT_FRAME_TIME):
        continue
    
    if (count == REJECT_FRAME_TIME):
        #reset count to move in multiples of FPS
        count = FPS * 2

    if (count % FPS == 0):
        # cv2.imshow(str(frame_num), frame)
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        cv2.imwrite("/Users/ruchit/Imperial/livelinessDetection/ann_approach/16Aug/fake/" + names[frame_num][:-1], frame)
        frame_num += 1

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  else:
    print("vid not returning frame") 
    break

vid.release() 
file_order.close()
cv2.destroyAllWindows()


