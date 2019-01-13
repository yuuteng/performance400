import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 4, 3), np.float32)
objp[:, :2] = np.mgrid[0:6, 0:4].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
#
images = glob.glob('images/photosMire/*.jpg')

count = 0
for fname in images:
    img = cv2.imread(fname)
    img = cv2.threshold(img, 150, 255, cv2.THRESH_TRUNC)[1]
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (6, 4), None)
    print(ret)
    # If found, add object points, image points (after refining them)
    if ret:
        count += 1
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (6, 4), corners2, ret)
        cv2.imwrite('images/miresCalibrees/mire' + str(count) + '.jpg', img)
        cv2.waitKey(0)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(mtx)
print(dist)

np.savetxt('matrixTxt/camera_matrix', mtx)
np.savetxt('matrixTxt/distortion_vector', dist)


# undistort
dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('images/calibresult.png', dst)
