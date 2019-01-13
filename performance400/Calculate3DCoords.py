import numpy as np
import time as time

camera_matrix = np.loadtxt('matrixTxt/adjusted_camera_matrix')
tvecs = np.loadtxt('matrixTxt/translation_vector')
rotation_matrix = np.loadtxt('matrixTxt/rotation_matrix')

tvecs = tvecs.reshape((3, 1))

tvecs = np.array([tvecs])

u = 2557
v = 45
z = 0

start_time = time.time()
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
