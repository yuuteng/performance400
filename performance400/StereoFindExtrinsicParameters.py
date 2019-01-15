import cv2
import numpy as np

frame1 = cv2.imread('images/photosMire/IMG_2102.jpg')
frame2 = cv2.imread('images/photosMire/IMG_2101.jpg')

obj_points = np.loadtxt('matrices/obj_points_stereo_test1')
img_points1 = np.loadtxt('matrices/img_points2_stereo_test1')
img_points2 = np.loadtxt('matrices/img_points1_stereo_test1')

obj_points = np.array(obj_points, 'float32')
img_points1 = np.array(img_points1, 'float32')
img_points2 = np.array(img_points2, 'float32')
size = frame2.shape[:2]

camera_matrix1 = np.loadtxt('matrices/camera_matrix')
camera_matrix2 = np.loadtxt('matrices/camera_matrix')
dist_coeffs1 = np.loadtxt('matrices/distortion_vector')
dist_coeffs2 = np.loadtxt('matrices/distortion_vector')
R = np.eye(3)
T = np.eye(3)
R1 = np.zeros(shape=(3, 3))
R2 = np.zeros(shape=(3, 3))
P1 = np.zeros(shape=(3, 3))
P2 = np.zeros(shape=(3, 3))

retval1, camera_matrix1, dist_coeffs1, rvecs1, tvecs1 = cv2.calibrateCamera([obj_points], [img_points1], size,
                                                                           camera_matrix1, dist_coeffs1)

retval2, camera_matrix2, dist_coeffs2, rvecs2, tvecs2 = cv2.calibrateCamera([obj_points], [img_points2], size,
                                                                           camera_matrix2, dist_coeffs2)

# --------------------------------------------------------------------------------------------------------------------
mean_error1 = 0
mean_error2 = 0
# dessiner les axes
frame1 = cv2.drawFrameAxes(frame1, camera_matrix1, dist_coeffs1, rvecs1[0], tvecs1[0], 100)
cv2.imwrite('images/stereo_test1_frame1_axes.jpg', frame1)
frame2 = cv2.drawFrameAxes(frame2, camera_matrix2, dist_coeffs2, rvecs2[0], tvecs2[0], 100)
cv2.imwrite('images/stereo_test1_frame2_axes.jpg', frame2)

# redessiner les points source et calculer le pourcentage d'erreur
img_points1_2, jacobian = cv2.projectPoints(obj_points, rvecs1[0], tvecs1[0], camera_matrix1, dist_coeffs1)
img_points2_2, jacobian = cv2.projectPoints(obj_points, rvecs2[0], tvecs2[0], camera_matrix2, dist_coeffs2)
for px in img_points1_2:
    cv2.circle(frame1, (px[0][0], px[0][1]), 10, (0, 0, 255), 5)
cv2.imwrite('images/stereo_test1_frame1_error.jpg', frame1)
for px in img_points2_2:
    cv2.circle(frame2, (px[0][0], px[0][1]), 10, (0, 0, 255), 5)
cv2.imwrite('images/stereo_test1_frame2_error.jpg', frame2)

for i in range(0, len(img_points1_2)):
    mean_error1 += cv2.norm(img_points1[i], img_points1_2[i][0], cv2.NORM_L2) / len(img_points1_2)
print('erreur moyenne 1', mean_error1)
for i in range(0, len(img_points2_2)):
    mean_error2 += cv2.norm(img_points2[i], img_points2_2[i][0], cv2.NORM_L2) / len(img_points2_2)
print('erreur moyenne 2', mean_error2)
# --------------------------------------------------------------------------------------------------------------------
stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
stereocalib_flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH | \
                    cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5

retval, camera_matrix1, dist_coefs1, \
camera_matrix2, dist_coefs2, R, T, E, F = cv2.stereoCalibrate([obj_points], [img_points1],
                                                              [img_points2],
                                                              None, None,
                                                              None, None,
                                                              size,
                                                              criteria=stereocalib_criteria,
                                                              flags=stereocalib_flags)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(camera_matrix1, dist_coefs1, camera_matrix2,
                                                                  dist_coefs2, size, R, T, R1, R2, P1, P2,
                                                                  Q=None, flags=cv2.CALIB_ZERO_DISPARITY,
                                                                  alpha=-1, newImageSize=(0, 0))

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
#
# P1 = camera_matrix1 @ np.append(I3, Zeros, axis=1)
# P2 = camera_matrix2 @ np.append(R, T, axis=1)
window_size = 3
min_disp = 0
num_disp = 160
stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp,
                               blockSize=5, P1=8 * 3 * window_size ** 2, P2=32 * 3 * window_size ** 2,
                               disp12MaxDiff=1, uniquenessRatio=15, speckleWindowSize=0, speckleRange=2,
                               preFilterCap=63,
                               mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
disp = stereo.compute(frame1, frame2)
points = cv2.reprojectImageTo3D(disp, Q)
cv2.imwrite('Disparitymap.jpg', disp)
print(points)

# ----------------------------------------------------------------------------------------------------------------------
# img_points1 = np.array(img_points1, 'float32')
# img_points2 = np.array(img_points2, 'float32')
#
# points4D = cv2.triangulatePoints(P1, P2, img_points1.T, img_points2.T)
#
# np.savetxt('matrices/points4D_stereo_test1', points4D)
#
# points3D = points4D[:, :3] / np.repeat(points4D[:, 3], 3).reshape(-1, 3)
#
# np.savetxt('matrices/points3D_stereo_test1', points3D)
#
# img_points2_3, jacobian = cv2.projectPoints(points3D, rvecs2[0], tvecs2[0], camera_matrix2, dist_coefs2)
# for px in img_points2_3:
#     cv2.circle(frame2, (px[0][0], px[0][1]), 10, (0, 255, 0), 5)
# cv2.imwrite('images/stereo_test1_frame1_errorTotale.jpg', frame2)
#
# img_points1_3, jacobian = cv2.projectPoints(points3D, rvecs1[0], tvecs1[0], camera_matrix1, dist_coefs1)
# for px in img_points1_3:
#     cv2.circle(frame1, (px[0][0], px[0][1]), 10, (0, 255, 0), 5)
# cv2.imwrite('images/stereo_test1_frame2_errorTotale.jpg', frame1)
