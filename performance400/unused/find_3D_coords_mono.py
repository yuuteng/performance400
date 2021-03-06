import numpy as np
import time as time
import cv2


def calculate_3d_coords(u, v, z=0, verbose=False):
    # camera_matrices = np.loadtxt('matrices/camera_matrices/extrinsic/stereo_1_droite_camera_matrix')
    # rotation_matrix = np.loadtxt('matrices/rotation_matrix/stereo_1_droite_rotation_matrix')
    # translation_vectors = np.loadtxt('matrices/vectors/translation/stereo_1_droite_translation_vector').reshape((3, 1))
    camera_matrix = np.loadtxt('matrices/camera_matrices/intrinsic/stereo_1_gauche_camera_matrix')
    rotation_matrix = np.loadtxt('matrices/rotation_matrix/stereo_1_gauche_rotation_matrix')
    translation_vectors = np.loadtxt('matrices/vectors/translation/stereo_1_gauche_translation_vector').reshape((3, 1))
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

