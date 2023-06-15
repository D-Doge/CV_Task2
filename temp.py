import numpy as np
import cv2 as cv
import glob
import math


CHESS_BOARD_PATTERN_WITDH_7x7 = 37.95


def calcDistanzeForFrame(K, corners, chessboardWith):
    K_inv = np.linalg.inv(K)

    pixels = np.ones(3)
    pixels[0] = corners[0][0][0] # Upper left point
    pixels[1] = corners[0][0][1]
    v1 = 1 * K_inv @ pixels

    pixels[0] = corners[6][0][0] # Upper left point
    pixels[1] = corners[6][0][1]
    v2 = 1 * K_inv @ pixels

                        # lenght of the vector
    z = chessboardWith / np.linalg.norm(v1 - v2)

    v1 = z * v1
    v2 = z * v2

    s = np.linalg.norm(v1) + np.linalg.norm(v2) + chessboardWith
    s = s/2

    A = s * (s - np.linalg.norm(v1)) * (s - np.linalg.norm(v2)) * (s - chessboardWith)
    A = np.sqrt(A)

    distance = (2 * A) / chessboardWith
    return distance # in cm

def calc_K(frame):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,6,0)
    objp = np.zeros((7*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    img = frame
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,7), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,7), corners2, ret)
        #cv.imshow('img', img)
        #cv.waitKey()
    else:
        return None

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx

    img = frames[3]
    #cv.imshow('img', img)
    #cv.waitKey()
    #cv.imwrite('frame.png', img)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    return newcameramtx, corners2

def getCornersOfFrame(frame):
        # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,6,0)
    objp = np.zeros((7*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    img = frame
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,7), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        #cv.drawChessboardCorners(img, (7,7), corners2, ret)
        #cv.imshow('img', img)
        #cv.waitKey()
    else:
        return None

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return corners2

video = cv.VideoCapture('videoHD1.avi')
fps = video.get(cv.CAP_PROP_FPS)
print('frames per second =',fps)

frame_id = frame_id = int(fps*(2*60 + 12)) # <------------------ Timestamp of frame 470cm
video.set(cv.CAP_PROP_POS_FRAMES, frame_id)
ret, frame_for_calc = video.read()

#cv.imshow('img', frame_for_calc)
#cv.waitKey()
corners = getCornersOfFrame(frame_for_calc)

succes = True
frame_count = 1270
best_frame_id = 0
best_d = math.inf
best_delta = math.inf
best_K = None
while 1270 == frame_count:
    video.set(cv.CAP_PROP_POS_FRAMES, frame_count)
    succes, frame = video.read()
    
    if(succes == False):
        continue
    K = calc_K(frame)
    if(K is None):
        frame_count = frame_count + 1
        continue
    d = calcDistanzeForFrame(K, corners, CHESS_BOARD_PATTERN_WITDH_7x7)
    if(np.abs(d - 470) < best_delta):
        best_frame_id = frame_count
        best_d = d
        best_delta = np.abs(d - 470)
        best_K = K

        print("-------------------")
        print("New Best Found")
        print("d: ", d)
        print("detla: ", np.abs(d - 470))
        print("frame_id: ", best_frame_id)
        print("-------------------")

  
    frame_count = frame_count + 1
    if frame_count % 100 == 0:
        print(frame_count)


#Far
frame_id = frame_id = int(fps*(3*60 + 7)) # <------------------ Timestamp of frame 470cm
video.set(cv.CAP_PROP_POS_FRAMES, frame_id)
ret, frame_for_calc = video.read()

#cv.imshow('img', frame_for_calc)
#cv.waitKey()
corners = getCornersOfFrame(frame_for_calc)
d = calcDistanzeForFrame(best_K, corners, CHESS_BOARD_PATTERN_WITDH_7x7)

print("Far: ", d)


#Near
frame_id = frame_id = int(fps*(1*60 + 28)) # <------------------ Timestamp of frame 470cm
video.set(cv.CAP_PROP_POS_FRAMES, frame_id)
ret, frame_for_calc = video.read()

#cv.imshow('img', frame_for_calc)
#cv.waitKey()
corners = getCornersOfFrame(frame_for_calc)
d = calcDistanzeForFrame(best_K, corners, CHESS_BOARD_PATTERN_WITDH_7x7)

print("Near: ", d)

print(best_K)