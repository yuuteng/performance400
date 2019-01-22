import cv2
import numpy as np


def calibrate(left_background, right_background, left_interest_points, right_interest_points,
              intrinsic_left_camera_matrix, intrinsic_right_camera_matrix,
              intrinsic_left_distortion_vector, intrinsic_right_distortion_vector):
    # conversion des listes au bon format pour utilisation de la fonction calibrate_camera de cv2
    left_obj_points = np.array(left_obj_points, 'float32')
    right_obj_points = np.array(right_obj_points, 'float32')
    left_img_points = np.array(left_img_points, 'float32')
    right_img_points = np.array(right_img_points, 'float32')
    left_size = left_background.shape[:2]
    right_size = right_background.shape[:2]

    # on se sert des paramètres initialement determinés par intrinsic_pre_autocalibration et on va les adapter à
    # l'image de la piste
    cali_flag = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_USE_INTRINSIC_GUESS

    # determination des parametres extrinsèques gauche
    _, extrinsic_left_camera_matrix, extrinsic_left_distortion_vector, extrinsic_left_rotation_vectors, \
    extrinsic_left_translation_vectors = cv2.calibrateCamera([left_obj_points], [left_img_points], left_size,
                                                             intrinsic_left_camera_matrix,
                                                             intrinsic_left_distortion_vector,
                                                             flags=cali_flag)

    # determination des parametres extrinsèques droite
    _, extrinsic_right_camera_matrix, extrinsic_right_distortion_vector, extrinsic_right_rotation_vectors, \
    extrinsic_right_translation_vectors = cv2.calibrateCamera([right_obj_points], [right_img_points], right_size,
                                                              intrinsic_right_camera_matrix,
                                                              intrinsic_right_distortion_vector,
                                                              flags=cali_flag)

    # on extrait les vecteurs qui nous interessent pour les enregistrer au format txt car calibrateCamera nous renvoie
    # des arrays de arrays avec 1 seul vecteur dedans
    extrinsic_left_rotation_vector = extrinsic_left_rotation_vectors[0]
    extrinsic_left_translation_vector = extrinsic_left_translation_vectors[0]
    extrinsic_right_rotation_vector = extrinsic_right_rotation_vectors[0]
    extrinsic_right_translation_vector = extrinsic_right_translation_vectors[0]

    np.savetxt('matrices/camera_matrix/extrinsic/extrinsic_left_camera_matrix', extrinsic_left_camera_matrix)
    np.savetxt('matrices/distortion_vector/extrinsic/extrinsic_left_distortion_vector',
               extrinsic_left_distortion_vector)
    np.savetxt('matrices/rotation_vector/extrinsic_left_rotation_vector', extrinsic_left_rotation_vector)
    np.savetxt('matrices/translation_vector/extrinsic_left_translation_vector', extrinsic_left_translation_vector)
    np.savetxt('matrices/camera_matrix/extrinsic/extrinsic_right_camera_matrix', extrinsic_right_camera_matrix)
    np.savetxt('matrices/distortion_vector/extrinsic/extrinsic_right_distortion_vector',
               extrinsic_right_distortion_vector)
    np.savetxt('matrices/rotation_vector/extrinsic_right_rotation_vector', extrinsic_right_rotation_vector)
    np.savetxt('matrices/translation_vector/extrinsic_right_translation_vector', extrinsic_right_translation_vector)

    pass


def get_extrinsic_parameters(left_or_right):
    if left_or_right == 0:
        extrinsic_camera_matrix = np.loadtxt('matrices/camera_matrix/extrinsic/extrinsic_left_camera_matrix')
        extrinsic_distortion_vector = np.loadtxt('matrices/distortion_vector/extrinsic/'
                                                 'extrinsic_left_distortion_vector')
        extrinsic_rotation_vector = np.loadtxt('matrices/rotation_vector/extrinsic_left_rotation_vector')
        extrinsic_translation_vector = np.loadtxt('matrices/translation_vector/extrinsic_left_translation_vector')
    else:
        extrinsic_camera_matrix = np.loadtxt('matrices/camera_matrix/extrinsic/extrinsic_right_camera_matrix')
        extrinsic_distortion_vector = np.loadtxt('matrices/distortion_vector/extrinsic/'
                                                 'extrinsic_right_distortion_vector')
        extrinsic_rotation_vector = np.loadtxt('matrices/rotation_vector/extrinsic_right_rotation_vector')
        extrinsic_translation_vector = np.loadtxt('matrices/translation_vector/extrinsic_right_translation_vector')

    return extrinsic_camera_matrix, extrinsic_distortion_vector, extrinsic_rotation_vector, extrinsic_translation_vector


def draw_axes(left_or_right, image):
    extrinsic_camera_matrix = get_extrinsic_parameters(left_or_right)[0]
    extrinsic_distortion_vector = get_extrinsic_parameters(left_or_right)[1]
    extrinsic_rotation_vector = get_extrinsic_parameters(left_or_right)[2]
    extrinsic_translation_vector = get_extrinsic_parameters(left_or_right)[3]
    if left_or_right == 0:
        prefix = "left"
    else:
        prefix = "right"

    image = cv2.drawFrameAxes(image, extrinsic_camera_matrix, extrinsic_distortion_vector, extrinsic_rotation_vector,
                              extrinsic_translation_vector, 1)
    cv2.namedWindow(prefix + "axes", image)
    cv2.imshow(prefix + "axes", image)
    cv2.waitKey(0)

    pass


def get_3d_coords(left_two_d_coords, right_two_d_coords):
    extrinsic_left_camera_matrix = get_extrinsic_parameters(0)[0]
    extrinsic_left_distortion_vector = get_extrinsic_parameters(0)[1]  # pour le test undistortPoints
    extrinsic_left_rotation_vector = get_extrinsic_parameters(0)[2]
    extrinsic_left_translation_vector = get_extrinsic_parameters(0)[3]

    extrinsic_right_camera_matrix = get_extrinsic_parameters(1)[0]
    extrinsic_right_distortion_vector = get_extrinsic_parameters(1)[1]  # pour le test undistortPoints
    extrinsic_right_rotation_vector = get_extrinsic_parameters(1)[2]
    extrinsic_right_translation_vector = get_extrinsic_parameters(1)[3]

    # on transorme les vecteurs rotation en matrices pour le calcul matriciel des matrices de projection
    left_rotation_matrix = cv2.Rodrigues(extrinsic_left_rotation_vector)
    right_rotation_matrix = cv2.Rodrigues(extrinsic_right_rotation_vector)

    # on change les formats des vecteurs tranlation pour pouvoir prendre leur transposé par la suite
    extrinsic_left_translation_vector = np.array([extrinsic_left_translation_vector])
    extrinsic_right_translation_vector = np.array([extrinsic_right_translation_vector])

    # calcul des matrices intermédiares necessaire au calcul des matrices de projection
    m1 = np.append(left_rotation_matrix, extrinsic_left_translation_vector.T, axis=1)
    m2 = np.append(right_rotation_matrix, extrinsic_right_translation_vector.T, axis=1)

    # calcul des matrices de projection
    projection_matrix_1 = extrinsic_left_camera_matrix @ m1
    projection_matrix_2 = extrinsic_right_camera_matrix @ m2

    # Test UndistortPoints pour voir si ça améliore le rendu final
    # left_two_d_coords = cv2.undistortPoints(left_two_d_coords, extrinsic_left_camera_matrix,
    #                                         extrinsic_left_distortion_vector)
    # right_two_d_coords = cv2.undistortPoints(right_two_d_coords, extrinsic_right_camera_matrix,
    #                                          extrinsic_right_distortion_vector)

    points_4d = cv2.triangulatePoints(projection_matrix_1, projection_matrix_2, left_two_d_coords.T,
                                      right_two_d_coords.T)

    # transfer des points 4D obtenus par triangulation en points 3d dans l'espace de la piste
    points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)

    # on change le format des points_3d vers une matrices N*3
    points_3d_bis = []
    for p in points_3d:
        points_3d_bis.append(p[0])
    points_3d_bis = np.asarray(points_3d_bis)

    return points_3d_bis
