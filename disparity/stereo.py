import numpy as np
import cv2
from matplotlib import pyplot as plt

imgLeft = cv2.imread('real/left1.jpg', 0)
imgRight = cv2.imread('real/right1.jpg', 0)

# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.imshow(imgLeft, 'gray')
# plt.axis('off')
# plt.subplot(1,2,2)
# plt.imshow(imgRight, 'gray')
# plt.axis('off')
# plt.show()

#@title
def ShowDisparity(bSize=15):
    # Initialize the stereo block matching object 
    stereo = cv2.StereoBM_create(numDisparities=80, blockSize=bSize)

    # Compute the disparity image
    disparity = stereo.compute(imgLeft, imgRight)

    # Normalize the image for representation
    min = disparity.min()
    max = disparity.max()
    disparity = np.uint8(255 * (disparity - min) / (max - min))
    
    # Plot the result
    return disparity

result = ShowDisparity(bSize=5)
print(len(result))
plt.imshow(result, 'gray')
plt.axis('off')
plt.show()

