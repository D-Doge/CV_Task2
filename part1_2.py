import numpy as np
import cv2 as cv
import glob




video = cv.VideoCapture('videoWW1_calibration.avi')
fps = video.get(cv.CAP_PROP_FPS)
print('frames per second =',fps)

frames = list()

for i in range(60):
    frame_id = int(fps*i)

    video.set(cv.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = video.read()
    if ret:
        frames.append(frame)


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpg')
for frame in frames:
    img = frame
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = frames[3]
cv.imshow('img', img)
cv.waitKey()
cv.imwrite('frame.png', img)
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
cv.imshow('img', dst)
cv.waitKey()
cv.imwrite('Result.png', dst)
# crop the image
x, y, w, h = roi

dst = dst[y:y+h, x:x+w]

cv.imshow('img', dst)
cv.waitKey()
cv.imwrite('ResultCropped.png', dst)