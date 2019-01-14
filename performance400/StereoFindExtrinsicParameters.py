import cv2
import numpy as np
import matplotlib.pyplot as plt

frame1 = cv2.imread('images/photosMire/IMG_2101.jpg')
frame2 = cv2.imread('images/photosMire/IMG_2102.jpg')

obj_points = np.loadtxt('matrices/obj_points_stereo_test1')
img_points1 = np.loadtxt('matrices/img_points3_stereo_test1')
img_points2 = np.loadtxt('matrices/img_points4_stereo_test1')

obj_points = np.array(obj_points, 'float32')
img_points1 = np.array(img_points1, 'float32')
img_points2 = np.array(img_points2, 'float32')
size = frame2.shape[:2]

camera_matrix = np.loadtxt('matrices/camera_matrix')
dist_coefs = np.loadtxt('matrices/distortion_vector')
R = np.array((3, 3), 'float32')
T = np.array((3, 3), 'float32')

retval, camera_matrix1, dist_coefs1, rvecs, tvecs = cv2.calibrateCamera([obj_points], [img_points1], size,
                                                                        camera_matrix, dist_coefs,
                                                                        flags=cv2.CALIB_USE_INTRINSIC_GUESS)

retval, camera_matrix2, dist_coefs2, rvecs, tvecs = cv2.calibrateCamera([obj_points], [img_points2], size,
                                                                        camera_matrix, dist_coefs,
                                                                        flags=cv2.CALIB_USE_INTRINSIC_GUESS)

retval, camera_matrix1, dist_coefs1, \
camera_matrix2, dist_coefs2, R, T, E, F = cv2.stereoCalibrate([obj_points], [img_points1], [img_points2],
                                                              camera_matrix1, dist_coefs1, camera_matrix2, dist_coefs2,
                                                              size, flags=cv2.CALIB_FIX_INTRINSIC)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(camera_matrix1, dist_coefs1, camera_matrix2,
                                                                  dist_coefs2, size, R, T,
                                                                  flags=cv2.CALIB_ZERO_DISPARITY)

np.savetxt('matrices/camera_matrix1_stereo_test1', camera_matrix1)
np.savetxt('matrices/camera_matrix2_stereo_test1', camera_matrix2)
np.savetxt('matrices/distortion_vector1_stereo_test1', dist_coefs1)
np.savetxt('matrices/distortion_vector2_stereo_test1', dist_coefs2)
np.savetxt('matrices/R_stereo_test1', R)
np.savetxt('matrices/T_stereo_test1', T)
np.savetxt('matrices/E_stereo_test1', E)
np.savetxt('matrices/F_stereo_test1', F)

# I3 = np.identity(3, 'float32')
# Zeros = np.zeros((3, 1), 'float32')

# P1 = np.append(I3, Zeros, axis=1)
# P2 = np.append(R, T, axis=1)

img_points1 = np.array(img_points1, 'float32')
img_points2 = np.array(img_points2, 'float32')

img_points1 = img_points1.reshape(2, 24)
img_points2 = img_points2.reshape(2, 24)

points4D = np.array((len(img_points1), 4), 'float32')

points4D = cv2.triangulatePoints(P1, P2, img_points1[::2], img_points2[::2], points4D)

np.savetxt('matrices/points4D_stereo_test1', points4D)

point3D = np.array((1, 3), 'float32')
point3D = cv2.convertPointsFromHomogeneous(points4D)

print(point3D)
np.savetxt('matrices/points4D_stereo_test1', point3D)

