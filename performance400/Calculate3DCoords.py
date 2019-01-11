import numpy as np
import time as time


camera_matrix = np.zeros((3, 3), 'float32')
camera_matrix[0, 0] = 3.91328923e+03
camera_matrix[1, 1] = 4.32611661e+03
camera_matrix[2, 2] = 1.0
camera_matrix[0, 2] = 1.78251688e+03
camera_matrix[1, 2] = 1.05257062e+03


rotation_matrix = np.zeros((3, 3), 'float32')
rotation_matrix[0, 0] = 0.36683743
rotation_matrix[0, 1] = 0.93024378
rotation_matrix[0, 2] = 0.00876395
rotation_matrix[1, 0] = -0.62405117
rotation_matrix[1, 1] = 0.2390833
rotation_matrix[1, 2] = 0.74390814
rotation_matrix[2, 0] = 0.68992061
rotation_matrix[2, 1] = -0.2783625
rotation_matrix[2, 2] = 0.66822442


tvecs = np.zeros((3, 1), 'float32')
tvecs[0] = np.array(-3.42421039)
tvecs[1] = np.array(7.041267)
tvecs[2] = np.array(64.3181479)

tvecs = np.array([tvecs])


u = 1640
v = 1557
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
