import numpy as np
import time as time


def calculate_3d_coords(u, v, z=0, verbose=False):
    camera_matrix = np.loadtxt('matrices/adjusted_camera_matrix')
    rotation_matrix = np.loadtxt('matrices/rotation_matrix')
    translation_vectors = np.loadtxt('matrices/translation_vector').reshape((3, 1))
    translation_vectors = np.array([translation_vectors])

    start_time = time.time()

    input_vector = np.array([u, v, 1]).reshape(3, 1)
    left_side_matrix = np.linalg.inv(rotation_matrix) @ np.linalg.inv(camera_matrix) @ input_vector
    right_side_matrix = np.linalg.inv(rotation_matrix) @ translation_vectors[0]

    if verbose:
        print('tvec', translation_vectors[0])
        print('rotation_matrix')
        print(rotation_matrix)
        print('gauche', left_side_matrix)
        print('droite', right_side_matrix)
        print('inverser matrice_camera', np.linalg.inv(camera_matrix))

    s = (z + right_side_matrix[2] / left_side_matrix[2])
    p = np.linalg.inv(rotation_matrix) @ (s * np.linalg.inv(camera_matrix) @ input_vector - translation_vectors[0])

    if verbose:
        print("--- %s seconds ---" % (time.time() - start_time))
        print('X et Y du point', p)

    return p[0][0], p[1][0], p[2][0]
