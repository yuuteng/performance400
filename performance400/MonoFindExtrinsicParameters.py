
import cv2
import numpy as np

cap = cv2.VideoCapture('videos/agnesModel.MOV')
success, frame = cap.read()
cv2.imwrite('images/photosPisteCali.jpg', frame)

obj_points = [[0, 0, 0], [0, 1.22, 0], [0, 2.44, 0], [0, 3.66, 0], [0, 4.88, 0], [0, 6.1, 0], [0, 7.32, 0],
              [20, 1.22, 0], [20, 2.44, 0], [20, 3.66, 0], [20, 4.88, 0], [20, 6.1, 0], [20, 7.32, 0],
              [40, 1.22, 0], [40, 2.44, 0], [40, 3.66, 0], [40, 4.88, 0], [40, 6.1, 0], [40, 7.32, 0],
              [50, 1.22, 0], [50, 2.44, 0], [50, 3.66, 0], [50, 4.88, 0], [50, 6.1, 0], [50, 7.32, 0]]

img_points = [[1572, 1534], [1640, 1557], [1708, 1578], [1780, 1601], [1853, 1628], [1925, 1651], [1997, 1674], [2036, 768],
              [2094, 783], [2153, 799], [2213, 813], [2274, 828], [2334, 845], [2309, 226], [2361, 237],
              [2411, 247], [2463, 260], [2513, 271], [2566, 281], [2416, 15], [2464, 24], [2509, 36], [2557, 45],
              [2607, 56], [2655, 66]]

np.savetxt('matrixTxt/obj_points_testMono1', obj_points)
np.savetxt('matrixTxt/img_points_testMono1', img_points)

obj_points = np.array(obj_points, 'float32')
img_points = np.array(img_points, 'float32')
size = frame.shape[:2]

camera_matrix = np.loadtxt('matrixTxt/camera_matrix')
dist_coefs = np.loadtxt('matrixTxt/distortion_vector')

retval, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera([obj_points], [img_points], size, camera_matrix,
                                                                      dist_coefs,
                                                                      flags=cv2.CALIB_USE_INTRINSIC_GUESS)

rotation_matrix, jacobian = cv2.Rodrigues(rvecs[0])

np.savetxt('matrixTxt/adjusted_camera_matrix', camera_matrix)
np.savetxt('matrixTxt/adjusted_distortion_vector', dist_coefs)
np.savetxt('matrixTxt/rotation_vector', rvecs[0])
np.savetxt('matrixTxt/translation_vector', tvecs[0])
np.savetxt('matrixTxt/rotation_matrix', rotation_matrix)

mean_error = 0
# dessiner les axes
frame = cv2.drawFrameAxes(frame, camera_matrix, dist_coefs, rvecs[0], tvecs[0], 100)
cv2.imwrite('images/frame1_axes.jpg', frame)

# redessiner les points source et calculer le pourcentage d'erreur comis
img_points2, jacobian = cv2.projectPoints(obj_points, rvecs[0], tvecs[0], camera_matrix, dist_coefs)
for px in img_points2:
    cv2.circle(frame, (px[0][0], px[0][1]), 10, (255, 255, 255), 5)
cv2.imwrite('images/frame1_errors.jpg', frame)

for i in range(0, len(img_points2)):
    mean_error += cv2.norm(img_points[i], img_points2[i][0], cv2.NORM_L2) / len(img_points2)
print('erreur moyenne', mean_error)

