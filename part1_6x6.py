import numpy as np
import cv2 as cv
import glob


def mirror_grid_anti_diagonal(grid, size=6):
    mirrored_grid = np.empty_like(grid)
    for i in range(size):
        for j in range(size):
            mirrored_grid[i * size + j] = grid[(size - j - 1) * size + (size - i - 1)]
    return mirrored_grid

def mirro_corner_on_vertical_axis(corners):
    #print(corners[0:7])
    corners[0:6] = (corners[0:6])[::-1]
    corners[6:12] = (corners[6:12])[::-1]
    corners[12:18] = (corners[12:18])[::-1]
    corners[18:24] = (corners[18:24])[::-1]
    corners[24:30] = (corners[24:30])[::-1]
    corners[30:] = (corners[30:])[::-1]
    #print(corners[0:7])
    #print("")
    return corners

def mirro_corner_on_horizontal_axis(corners):
    temp = np.copy(corners)
    corners[0:6] = temp[30:]
    corners[6:12] = temp[24:30]
    corners[12:18] = temp[18:24]
    corners[18:24] = temp[12:18]
    corners[24:30] = temp[6:12]
    corners[30:] = temp[0:6]

    return corners


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

frames = list()
frame_count = 0
succes = True
while succes:
    video.set(cv.CAP_PROP_POS_FRAMES, frame_count)
    succes, frame = video.read()
    
    if(succes == False):
        continue
    if(frame_count % 1 == 0):
        frames.append(frame)

    frame_count = frame_count + 1

print("Number of Frames: ", len(frames))

frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)
result_chess = cv.VideoWriter('result_chess.avi', 
                         cv.VideoWriter_fourcc(*'MJPG'),
                         fps, size)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,6,0)
objp = np.zeros((6*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpg')
for frame in frames:
    img = frame
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (6,6), None)
    print(ret)
    # If found, add object points, image points (after refining them)
    if ret == True:       
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)

        if corners2[0][0][1] > corners2[7][0][1]: 
            corners2 = mirro_corner_on_horizontal_axis(corners2)

        if corners2[0][0][0] > corners2[7][0][0]: 
            corners2 = mirro_corner_on_vertical_axis(corners2)

        if corners2[0][0][1] > corners2[7][0][1] and corners2[0][0][0] > corners2[7][0][0]:
            #flip on the diagonal
            corners2 = mirror_grid_anti_diagonal(corners2)

        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (6,6), corners2, ret)
        result_chess.write(img)
        #cv.imshow('img', img)
        #cv.waitKey(500)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = frames[100]
#cv.imshow('img', img)
#cv.waitKey()
#cv.imwrite('frame.png', img)
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
#cv.imshow('img', dst)
#cv.waitKey()
#cv.imwrite('Result.png', dst)
# crop the image
x, y, w, h = roi

dst = dst[y:y+h, x:x+w]

#cv.imshow('img', dst)
#cv.waitKey()
#cv.imwrite('ResultCropped.png', dst)


   

result_noCrop = cv.VideoWriter('result_noCrop.avi', 
                         cv.VideoWriter_fourcc(*'MJPG'),
                         fps, size)
result_Crop = cv.VideoWriter('result_Crop.avi', 
                         cv.VideoWriter_fourcc(*'MJPG'),
                         fps, (w, h))


frame_count = 0

while(True):
    video.set(cv.CAP_PROP_POS_FRAMES, frame_count)
    ret, frame = video.read()
  
    if ret == True: 
  
        # Write the frame into the
        # file 'filename.avi'
        dst = cv.undistort(frame, mtx, dist, None, newcameramtx)
        result_noCrop.write(dst)

        dst = dst[y:y+h, x:x+w]
        result_Crop.write(dst)

        frame_count = frame_count + 1
        print(frame_count)
    # Break the loop
    else:
        break
  
# When everything done, release 
# the video capture and video 
# write objects
video.release()
result_chess.release()
result_noCrop.release()
result_Crop.release()