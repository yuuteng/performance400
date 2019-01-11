
import cv2
import numpy as np


img = cv2.imread('images/maquette1.jpg')
# Trouve les paramètres intresecs et extrinsecs de la camera. Possibilité de multiplier les images pour la calibration

obj_points = [[0, 0, 0], [0, 1.22, 0], [0, 2.44, 0], [0, 3.66, 0], [0, 4.88, 0], [0, 6.1, 0], [0, 7.32, 0],
              [0, 8.54, 0], [50, 0, 0], [50, 1.22, 0], [50, 2.44, 0], [50, 3.66, 0], [50, 4.88, 0],
              [50, 6.1, 0], [50, 7.32, 0], [50, 8.54, 0], [60, 0, 0], [60, 1.22, 0], [60, 2.44, 0],
              [60, 3.66, 0], [60, 4.88, 0], [60, 6.1, 0], [60, 7.32, 0], [60, 8.54, 0], [80, 0, 0],
              [80, 1.22, 0], [80, 2.44, 0], [80, 3.66, 0], [80, 4.88, 0], [80, 6.1, 0], [80, 7.32, 0],
              [80, 8.54, 0]]
img_points = [[1926, 1769], [1892, 1828], [1853, 1828], [1821, 1858], [1778, 1888], [1740, 1922], [1697, 1957],
              [1654, 1997], [808, 1200], [776, 1215], [743, 1234], [709, 1253], [673, 1273], [640, 1294],
              [601, 1314], [556, 1336], [652, 1123], [621, 1140], [590, 1156], [555, 1174], [521, 1192], [485, 1210],
              [448, 1229], [411, 1247], [382, 988], [354, 1002], [321, 1015], [291, 1032], [259, 1047],
              [226, 1063], [193, 1080], [160, 1096]]

obj_points = np.array(obj_points, 'float32')
img_points = np.array(img_points, 'float32')

size = img.shape[:2]

camera_matrix = np.zeros((3, 3), 'float32')
camera_matrix[0, 0] = 2200.0
camera_matrix[1, 1] = 2200.0
camera_matrix[2, 2] = 1.0
camera_matrix[0, 2] = 750.0
camera_matrix[1, 2] = 750.0

print('camera_matrix initiale')
print(camera_matrix)

dist_coefs = np.zeros(4, 'float32')
print(dist_coefs)

retval, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera([obj_points], [img_points], size, camera_matrix,
                                                                      dist_coefs,
                                                                      flags=cv2.CALIB_TILTED_MODEL)

# retval, rvecs[0], tvecs[0] = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coefs, rvecs[0], tvecs[0],
#                                          False, cv2.SOLVEPNP_ITERATIVE)

print('retval : seuil optimal', retval)
print('new_camera_matrix : paramètres intrinsecs de la camera')
print(camera_matrix)
print('dist_coefs : coefficients de distortion')
print(dist_coefs)
print('rvecs : vecteurs de rotation')
print(rvecs)
print('tvecs : vecteurs de translation')
print(tvecs)

# tracé des axes spatiaux déterminés
image = cv2.drawFrameAxes(img, camera_matrix, dist_coefs, rvecs[0], tvecs[0], 100)

cv2.imwrite('axes.jpg', image)

image2 = cv2.imread('axes.jpg')

# Calcul des erreurs de projection
mean_error = 0

img_points2, jacobian = cv2.projectPoints(obj_points, rvecs[0], tvecs[0], camera_matrix, dist_coefs)
for px in img_points2:
    cv2.circle(image, (px[0][0], px[0][1]), 10, (0, 0, 255), 5)
cv2.imwrite('errors.jpg', image)
for i in range(0, len(img_points2)):
    mean_error += cv2.norm(img_points[i], img_points2[i][0], cv2.NORM_L2) / len(img_points2)

print("total error: ", mean_error / len(obj_points))

rotation_matrix, jacobian = cv2.Rodrigues(rvecs[0])

# retval, rvec, tvec =  cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coefs, rvecs[0], tvecs[0], False, cv2.SOLVEPNP_ITERATIVE)
obj_points2 = [[0, 0, 10], [0, 1.22, 10], [0, 2.44, 10], [0, 3.66, 10], [0, 4.88, 10], [0, 6.1, 10], [0, 7.32, 10],
               [0, 8.54, 10], [50, 0, 10], [50, 1.22, 10], [50, 2.44, 10], [50, 3.66, 10], [50, 4.88, 10],
               [50, 6.1, 10], [50, 7.32, 10], [50, 8.54, 10], [60, 0, 10], [60, 1.22, 10], [60, 2.44, 10],
               [60, 3.66, 10], [60, 4.88, 10], [60, 6.1, 10], [60, 7.32, 10], [60, 8.54, 10], [80, 0, 10],
               [80, 1.22, 10], [80, 2.44, 10], [80, 3.66, 10], [80, 4.88, 10], [80, 6.1, 10], [80, 7.32, 10],
               [80, 8.54, 10]]

obj_points2 = np.array(obj_points2, 'float32')

img_points3, jacobian = cv2.projectPoints(obj_points2, rvecs[0], tvecs[0], camera_matrix, dist_coefs)
for px in img_points3:
    cv2.circle(image, (px[0][0], px[0][1]), 10, (0, 0, 255), 5)
cv2.imwrite('elevated.jpg', image)

# on veut retrouver X et Y sur le terrain d'un point de cote connue z à partir de u  et v  trouvées sur l'image
u = 1926
v = 1769
z = 0
uvVect = np.array([u, v, 1])
uvVect = uvVect.reshape(3, 1)
left_side_matrix = np.linalg.inv(rotation_matrix) @ np.linalg.inv(camera_matrix) @ uvVect
right_side_matrix = np.linalg.inv(rotation_matrix) @ tvecs[0]

# print('tvec', tvecs[0])
print('rotation', rotation_matrix)
# print('gauche', left_side_matrix)
# print('droite', right_side_matrix)
# print('inverser matrice_camera', np.linalg.inv(camera_matrix))

s = (z + right_side_matrix[2] / left_side_matrix[2])

p = np.linalg.inv(rotation_matrix) @ (s * np.linalg.inv(camera_matrix) @ uvVect - tvecs[0])

print('X et Y du point', p)

p = np.array(p, 'float32')
img_point4, jacobian = cv2.projectPoints(p.T, rvecs[0], tvecs[0], camera_matrix, dist_coefs)

print('u et v : ', img_point4)
cv2.circle(image2, (img_point4[0][0][0], img_point4[0][0][1]), 10, (0, 255, 0), 5)
cv2.imwrite('calcul3D.jpg', image2)
