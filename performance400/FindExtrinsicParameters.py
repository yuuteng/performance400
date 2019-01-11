
import cv2
import numpy as np
import time




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

obj_points = np.array(obj_points, 'float32')
img_points = np.array(img_points, 'float32')
size = frame.shape[:2]

camera_matrix = np.zeros((3, 3), 'float32')
camera_matrix[0, 0] = 3.62326648e+03
camera_matrix[1, 1] = 3.87026292e+03
camera_matrix[2, 2] = 1.0
camera_matrix[0, 2] = 1.84108785e+03
camera_matrix[1, 2] = 9.40707567e+02

dist_coefs = np.zeros((5, 1), 'float32')
dist_coefs[0] = 5.25458116e-01
dist_coefs[1] = -6.05514178e+00
dist_coefs[2] = 6.54565053e-03
dist_coefs[3] = -1.43748025e-03
dist_coefs[4] = 2.24631481e+01

dist_coefs = np.array(dist_coefs)

retval, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera([obj_points], [img_points], size, camera_matrix,
                                                                      dist_coefs,
                                                                      flags=cv2.CALIB_USE_INTRINSIC_GUESS)
print ('camera_matrix')
print(camera_matrix)


# dessiner les axes
frame = cv2.drawFrameAxes(frame, camera_matrix, dist_coefs, rvecs[0], tvecs[0], 100)
cv2.imwrite('images/frame1_axes.jpg', frame)

# redessiner les points source
img_points2, jacobian = cv2.projectPoints(obj_points, rvecs[0], tvecs[0], camera_matrix, dist_coefs)
for px in img_points2:
    cv2.circle(frame, (px[0][0], px[0][1]), 10, (255, 255, 255), 5)
cv2.imwrite('images/frame1_errors.jpg', frame)


rotation_matrix, jacobian = cv2.Rodrigues(rvecs[0])
start_time = time.time()
# on veut retrouver X et Y sur le terrain d'un point de cote connue z à partir de u  et v  trouvées sur l'image
u = 1640
v = 1557
z = 0
uvVect = np.array([u, v, 1])
uvVect = uvVect.reshape(3, 1)
left_side_matrix = np.linalg.inv(rotation_matrix) @ np.linalg.inv(camera_matrix) @ uvVect
right_side_matrix = np.linalg.inv(rotation_matrix) @ tvecs[0]

print('tvec', tvecs[0])
print('rotation_matrix')
print( rotation_matrix)
# print('gauche', left_side_matrix)
# print('droite', right_side_matrix)
# print('inverser matrice_camera', np.linalg.inv(camera_matrix))

s = (z + right_side_matrix[2] / left_side_matrix[2])

p = np.linalg.inv(rotation_matrix) @ (s * np.linalg.inv(camera_matrix) @ uvVect - tvecs[0])
print("--- %s seconds ---" % (time.time() - start_time))
print('X et Y du point', p)

p = np.array(p, 'float32')
img_point4, jacobian = cv2.projectPoints(p.T, rvecs[0], tvecs[0], camera_matrix, dist_coefs)

print('u et v : ', img_point4)
cv2.circle(frame, (img_point4[0][0][0], img_point4[0][0][1]), 10, (0, 255, 0), 5)
cv2.imwrite('images/frame1_calcul3D.jpg', frame)
