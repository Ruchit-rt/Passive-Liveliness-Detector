import numpy as np
import cv2

###################################

# # Set the path to the images captured by the left and right cameras
# path = "data/"

# # Termination criteria for refining the detected corners
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# objp = np.zeros((1,6*9,3), np.float32)
# objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)

# img_ptsL = []
# img_ptsR = []
# obj_pts = []

# for i in range(1,12):
# 	imgL = cv2.imread(pathL+"img%d.png"%i)
# 	imgR = cv2.imread(pathR+"img%d.png"%i)
# 	imgL_gray = cv2.imread(pathL+"img%d.png"%i,0)
# 	imgR_gray = cv2.imread(pathR+"img%d.png"%i,0)

# 	outputL = imgL.copy()
# 	outputR = imgR.copy()

# 	retR, cornersR =  cv2.findChessboardCorners(outputR,(9,6),None)
# 	retL, cornersL = cv2.findChessboardCorners(outputL,(9,6),None)

# 	if retR and retL:
# 		obj_pts.append(objp)
# 		cv2.cornerSubPix(imgR_gray,cornersR,(11,11),(-1,-1),criteria)
# 		cv2.cornerSubPix(imgL_gray,cornersL,(11,11),(-1,-1),criteria)
# 		cv2.drawChessboardCorners(outputR,(9,6),cornersR,retR)
# 		cv2.drawChessboardCorners(outputL,(9,6),cornersL,retL)
# 		cv2.imshow('cornersR',outputR)
# 		cv2.imshow('cornersL',outputL)
# 		cv2.waitKey(0)

# 		img_ptsL.append(cornersL)
# 		img_ptsR.append(cornersR)


# # Calibrating left camera
# retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts,img_ptsL,imgL_gray.shape[::-1],None,None)
# hL,wL= imgL_gray.shape[:2]
# new_mtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

# # Calibrating right camera
# retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts,img_ptsR,imgR_gray.shape[::-1],None,None)
# hR,wR= imgR_gray.shape[:2]
# new_mtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,(wR,hR),1,(wR,hR))

###################################

# Reading the left and right images.

imgL = cv2.imread("fake/fakeAmit1.png",0)
imgR = cv2.imread("fake/fakeAmit2.png",0)
# imgL = cv2.imread("real/realAmit1.png",0)
# imgR = cv2.imread("real/realAmit2.png",0)

# Setting parameters for StereoSGBM algorithm
minDisparity = 1
numDisparities = 160 
blockSize = 1
disp12MaxDiff = 5
uniquenessRatio = 10
speckleWindowSize = 10
speckleRange = 42


cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',600,600)

def nothing(x):
    print(x)

cv2.createTrackbar('numDisparities','disp',160,160,nothing)
cv2.createTrackbar('blockSize','disp',1,50,nothing)
cv2.createTrackbar('uniquenessRatio','disp',10,100,nothing)
cv2.createTrackbar('speckleRange','disp',42,100,nothing)
cv2.createTrackbar('speckleWindowSize','disp',10,25,nothing)
cv2.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
cv2.createTrackbar('minDisparity','disp',1,25,nothing)


# Creating an object of StereoSGBM algorithm
stereo = cv2.StereoSGBM_create(minDisparity = minDisparity,
        numDisparities = numDisparities,
        blockSize = blockSize,
        disp12MaxDiff = disp12MaxDiff,
        uniquenessRatio = uniquenessRatio,
        speckleWindowSize = speckleWindowSize,
        speckleRange = speckleRange
    )

while True:

	# Udating the parameters based on the trackbar positions
	numDisparities = cv2.getTrackbarPos('numDisparities','disp')
	blockSize = cv2.getTrackbarPos('blockSize','disp')
	uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
	speckleRange = cv2.getTrackbarPos('speckleRange','disp')
	speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')
	disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
	minDisparity = cv2.getTrackbarPos('minDisparity','disp')
	
	# Setting the updated parameters before computing disparity map
	stereo.setNumDisparities(numDisparities)
	stereo.setBlockSize(blockSize)
	stereo.setUniquenessRatio(uniquenessRatio)
	stereo.setSpeckleRange(speckleRange)
	stereo.setSpeckleWindowSize(speckleWindowSize)
	stereo.setDisp12MaxDiff(disp12MaxDiff)
	stereo.setMinDisparity(minDisparity)

	# Clculating disparith using the StereoSGBM algorithm
	disp = stereo.compute(imgL, imgR).astype(np.float32)
	disp = cv2.normalize(disp,0,255,cv2.NORM_MINMAX)

	# Displaying the disparity map
	cv2.imshow("disparity",disp)
	
	# Close window
	if cv2.waitKey(0) == 27:
		break

# depth using the formula depth = (1/disparity) * bf
# for i in range(0,len(disp)):
# 	for j in range(0, len(disp[i])):
# 		disp[i][j] = 35 * 102 / disp[i][j]

# display depth map
# cv2.imshow("depth",disp)
cv2.waitKey(0)

####################################

# solving for M in the following equation
# ||    depth = M * (1/disparity)   ||
# for N data points coeff is Nx2 matrix with values 
# 1/disparity, 1
# and depth is Nx1 matrix with depth values
# disp_inv = 1/disp
# coeff = np.vstack([disp_inv, np.ones(len(disp_inv))]).T
# ret, sol = cv2.solve(coeff,z,flags=cv2.DECOMP_QR)
# M = sol[0,0]
# C = sol[1,0]
# print("Value of M = ",M)

####################################