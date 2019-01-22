import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
object_point = np.zeros((6 * 4, 3), np.float32)
object_point[:, :2] = np.mgrid[0:6, 0:4].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
object_points = []  # 3d point in real world space
img_points = []  # 2d points in image plane.
#
files = glob.glob('images/targets/*.jpg')

count = 0
for file in files:
    img = cv2.imread(file)
    # img = cv2.threshold(img, 200, 255, cv2.THRESH_TRUNC)[1]
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
        object_points.append(object_point)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        corners3 = []
        for i in corners2:
            corners3.append(i[0])
        np.savetxt('matrices/img_points' + str(count) + '_stereo_test1', corners3)
        img_points.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (6, 4), corners2, ret)
        cv2.imwrite('images/targets/calibrated/mire' + str(count) + '.jpg', img)
        cv2.waitKey(0)

(ret, mtx, dist, _, _) = cv2.calibrateCamera(object_points, img_points, gray.shape[::-1], None, None)

print(mtx)
print(dist)

np.savetxt('matrices/camera_matrix', mtx)
np.savetxt('matrices/distortion_vector', dist)
np.savetxt('matrices/obj_points_stereo_test1', object_points[0])

# undistort
dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('images/calibresult.png', dst)
