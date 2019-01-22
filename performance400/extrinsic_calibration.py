import cv2 as cv
import numpy as np


def calibrate(left_background, right_background, left_interest_points, right_interest_points,
              intrinsic_parameters):
    left_image_points, left_object_points = calibrate_single(left_background, left_interest_points, 40)
    right_image_points, right_object_points = calibrate_single(right_background, right_interest_points, 40)

    # on converti nos lites au bon format pour leur utilisation dans calibrateCamera de cv2
    left_object_points = np.array(left_object_points, 'float32')
    right_object_points = np.array(right_object_points, 'float32')
    left_image_points = np.array(left_image_points, 'float32')
    right_image_points = np.array(right_image_points, 'float32')
    left_size = left_background.shape[:2]
    right_size = right_background.shape[:2]

    intrinsic_left_camera_matrix, intrinsic_right_camera_matrix, intrinsic_left_distortion_vector, \
    intrinsic_right_distortion_vector = intrinsic_parameters

    # on se sert des parametres intrinsèques calculés au préalables pour affiner la determination des paramètres 
    # extrinsèques 
    cali_flag = cv.CALIB_FIX_INTRINSIC | cv.CALIB_CB_NORMALIZE_IMAGE | cv.CALIB_USE_INTRINSIC_GUESS

    _, extrinsic_left_camera_matrix, extrinsic_left_distortion_vector, extrinsic_left_rotation_vectors, \
    extrinsic_left_translation_vectors = cv.calibrateCamera([left_object_points], [left_image_points], left_size,
                                                            intrinsic_left_camera_matrix,
                                                            intrinsic_left_distortion_vector,
                                                            flags=cali_flag)

    _, extrinsic_right_camera_matrix, extrinsic_right_distortion_vector, extrinsic_right_rotation_vectors, \
    extrinsic_right_translation_vectors = cv.calibrateCamera([right_object_points], [right_image_points], right_size,
                                                             intrinsic_right_camera_matrix,
                                                             intrinsic_right_distortion_vector,
                                                             flags=cali_flag)
    # on extrain le vecteur qui nous interesse car calibrateCamera nous renvoie des vecteurs de vecteurs avec 1 seul 
    # vecteur dedans 
    extrinsic_left_rotation_vector = extrinsic_left_rotation_vectors[0]
    extrinsic_left_translation_vector = extrinsic_left_translation_vectors[0]
    extrinsic_right_rotation_vector = extrinsic_right_rotation_vectors[0]
    extrinsic_right_translation_vector = extrinsic_right_translation_vectors[0]

    np.save('matrices/camera_matrix/extrinsic/extrinsic_left_camera_matrix', extrinsic_left_camera_matrix)
    np.save('matrices/distortion_vectors/extrinsic/extrinsic_left_distortion_vector',
            extrinsic_left_distortion_vector)
    np.save('matrices/rotation_vector/extrinsic_left_rotation_vector', extrinsic_left_rotation_vector)
    np.save('matrices/translation_vector/extrinsic_left_translation_vector', extrinsic_left_translation_vector)
    np.save('matrices/camera_matrix/extrinsic/extrinsic_right_camera_matrix', extrinsic_right_camera_matrix)
    np.save('matrices/distortion_vectors/extrinsic/extrinsic_right_distortion_vector',
            extrinsic_right_distortion_vector)
    np.save('matrices/rotation_vector/extrinsic_right_rotation_vector', extrinsic_right_rotation_vector)
    np.save('matrices/translation_vector/extrinsic_right_translation_vector', extrinsic_right_translation_vector)


def calibrate_single(image, interest_points, sensitivity):
    image_points, object_points = interest_points
    orb = cv.ORB_create(nfeatures=10, scoreType=cv.ORB_HARRIS_SCORE)
    calibrated_interest_points = ([], object_points.copy())
    removed = 0
    for j in range(len(image_points)):
        x, y = image_points[j]

        sub_image = image[y - sensitivity: y + sensitivity, x - sensitivity: x + sensitivity]
        gray = cv.cvtColor(sub_image, cv.COLOR_BGR2GRAY)

        keypoints = orb.detect(gray, None)
        keypoints, descriptors = orb.compute(gray, keypoints)

        current = 0
        name = "Calibration extrinsèque"
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        if len(keypoints) > 0:
            test = False
            while True:
                draw_keypoints(sub_image, keypoints, current)
                cv.imshow(name, sub_image)
                key = cv.waitKey(0)
                if key == ord('q'):
                    test = True
                    break
                elif key == 13:  # enter
                    calibrated_interest_points[0].append(
                        (int(x - sensitivity + keypoints[current].pt[0]),
                         int(y - sensitivity + keypoints[current].pt[1])))
                    break
                elif key == ord('s'):
                    calibrated_interest_points[1].pop(j - removed)
                    removed += 1
                    break
                elif key == 81:  # left
                    current -= 1
                    current %= len(keypoints)
                elif key == 83:  # right
                    current += 1
                    current %= len(keypoints)

            if test:
                break
        else:
            calibrated_interest_points[1].pop(j - removed)
            removed += 1
            cv.imshow(name, sub_image)
            if cv.waitKey(0) == ord('q'):
                break

    cv.destroyAllWindows()

    return calibrated_interest_points


def draw_keypoints(image, keypoints, current):
    for k in range(len(keypoints) - 1, -1, -1):
        keypoint = keypoints[k]
        color = (0, 0, 255)
        if k == current:
            color = (0, 255, 0)
        cv.circle(image, (int(keypoint.pt[0]), int(keypoint.pt[1])), 5, color, thickness=1)


def get_extrinsic_parameters(right_camera):
    if right_camera:
        extrinsic_camera_matrix = np.loadtxt('matrices/camera_matrix/extrinsic/extrinsic_right_camera_matrix')
        extrinsic_distortion_vector = np.loadtxt('matrices/distortion_vectors/extrinsic/'
                                                 'extrinsic_right_distortion_vector')
        extrinsic_rotation_vector = np.loadtxt('matrices/rotation_vector/extrinsic_right_rotation_vector')
        extrinsic_translation_vector = np.loadtxt('matrices/translation_vector/extrinsic_right_translation_vector')
    else:
        extrinsic_camera_matrix = np.loadtxt('matrices/camera_matrix/extrinsic/extrinsic_left_camera_matrix')
        extrinsic_distortion_vector = np.loadtxt('matrices/distortion_vectors/extrinsic/'
                                                 'extrinsic_left_distortion_vector')
        extrinsic_rotation_vector = np.loadtxt('matrices/rotation_vector/extrinsic_left_rotation_vector')
        extrinsic_translation_vector = np.loadtxt('matrices/translation_vector/extrinsic_left_translation_vector')

    return extrinsic_camera_matrix, extrinsic_distortion_vector, extrinsic_rotation_vector, extrinsic_translation_vector


def draw_axes(image, right_camera):
    extrinsic_camera_matrix, extrinsic_distortion_vector, extrinsic_rotation_vector, extrinsic_translation_vector \
        = get_extrinsic_parameters(right_camera)

    prefix = "right" if right_camera else "left"

    image = cv.drawFrameAxes(image, extrinsic_camera_matrix, extrinsic_distortion_vector, extrinsic_rotation_vector,
                             extrinsic_translation_vector, 1)
    cv.namedWindow(prefix + "axes", image)
    cv.imshow(prefix + "axes", image)
    cv.waitKey(0)


def get_3d_coords(left_two_d_coords, right_two_d_coords):
    # left_two_d_coords et right_two_d_coords doivent être des vecteurs N*2 mais N peut être egal à 1
    extrinsic_left_camera_matrix, extrinsic_left_distortion_vector, extrinsic_left_rotation_vector, \
    extrinsic_left_translation_vector = get_extrinsic_parameters(False)

    extrinsic_right_camera_matrix, extrinsic_right_distortion_vector, extrinsic_right_rotation_vector, \
    extrinsic_right_translation_vector = get_extrinsic_parameters(True)

    left_rotation_matrix = cv.Rodrigues(extrinsic_left_rotation_vector)
    right_rotation_matrix = cv.Rodrigues(extrinsic_right_rotation_vector)

    extrinsic_left_translation_vector = np.array([extrinsic_left_translation_vector])
    extrinsic_right_translation_vector = np.array([extrinsic_right_translation_vector])

    m1 = np.append(left_rotation_matrix, extrinsic_left_translation_vector.T, axis=1)
    m2 = np.append(right_rotation_matrix, extrinsic_right_translation_vector.T, axis=1)

    projection_matrix_1 = extrinsic_left_camera_matrix @ m1
    projection_matrix_2 = extrinsic_right_camera_matrix @ m2

    # on enleve les 1e17 qui ont ete mis au endroits où on a eu des erreurs de pointage du coureur
    ind_fail = get_positions_fails(left_two_d_coords, right_two_d_coords)
    left_two_d_coords, right_two_d_coords = delete_positions_fails(left_two_d_coords, right_two_d_coords, ind_fail)

    # Test UndistortPoits
    # left_two_d_coords = cv2.undistortPoints(left_two_d_coords, extrinsic_left_camera_matrix,
    #                                        extrinsic_left_distortion_vector)
    # right_two_d_coords = cv2.undistortPoints(right_two_d_coords, extrinsic_right_camera_matrix,
    #                                          extrinsic_right_distortion_vector)

    #  on fait la triangulation avec les points pour lesquels les 1e17 ont etes enleves
    points_4d = cv.triangulatePoints(projection_matrix_1, projection_matrix_2, left_two_d_coords.T,
                                     right_two_d_coords.T)

    points_3d = cv.convertPointsFromHomogeneous(points_4d.T)

    # on converti le format de point_3d vers un format N*3 plus facile à utiliser
    points_3d_bis = []
    for p in points_3d:
        points_3d_bis.append(p[0])
    points_3d_bis = np.asarray(points_3d_bis)

    # on replace les 1e17 aux positions initiales pour ne pas perdre d'information sur les frames pour synchroniser
    # les detections
    for x in ind_fail:
        points_3d_bis = np.append(np.append(points_3d_bis[:x], [[1e17, 1e17, 1e17]], axis=0),
                                  points_3d_bis[x:], axis=0)

    return points_3d_bis


def get_positions_fails(left_two_d_coords, right_two_d_coords):
    # retourne les indices des positions où on a eu une erreur de pointage automatique du coureur
    ind_fails = []
    for n in range(np.shape(left_two_d_coords)[0]):
        if abs(left_two_d_coords[n].any()) > 1e16 or abs(right_two_d_coords[n].any()) > 1e16:
            ind_fails.append(n)

    return ind_fails


def delete_positions_fails(left_two_d_coords, right_two_d_coords, ind_fails):
    # supprime les points où on a repéré une erreur
    count = 0
    for k in ind_fails:
        left_two_d_coords = np.delete(left_two_d_coords, (k - count), axis=0)
        right_two_d_coords = np.delete(right_two_d_coords, (k - count), axis=0)
        count += 1

    return left_two_d_coords, right_two_d_coords
