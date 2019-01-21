import cv2
import numpy as np

droite_ou_gauche = 'droite'


def find_extrinsic_parameters(img, obj_points, img_points, camera_matrix, dist_coeffs, show=True, save=False,
                              prefix=''):
    if save:
        if prefix == '':
            return print('Veuillez saisir un prefix')
        np.savetxt('matrices/points/points_image/' + prefix + '_img_points', img_points)
        np.savetxt('matrices/points/points_objet/' + prefix + '_obj_points', obj_points)

    obj_points = np.array(obj_points, 'float32')
    img_points = np.array(img_points, 'float32')
    size = img.shape[:2]

    cali_flag = cv2.CALIB_FIX_INTRINSIC
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera([obj_points], [img_points], size,
                                                                           camera_matrix,
                                                                           dist_coeffs, flags=cali_flag)

    (rotation_matrix, _) = cv2.Rodrigues(rvecs[0])

    if show:
        mean_error = 0
        # dessiner les axes
        img = cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], 5)

        # redessiner les points source et calculer le pourcentage d'erreur comis
        img_points2, jacobian = cv2.projectPoints(obj_points, rvecs[0], tvecs[0], camera_matrix, dist_coeffs)
        for px in img_points2:
            cv2.circle(img, (px[0][0], px[0][1]), 10, (255, 255, 255), 5)
        cv2.namedWindow("errors", cv2.WINDOW_NORMAL)
        cv2.imshow('errors', img)
        cv2.waitKey(0)
        for i in range(0, len(img_points2)):
            mean_error += cv2.norm(img_points[i], img_points2[i][0], cv2.NORM_L2) / len(img_points2)
        print('erreur moyenne', mean_error)
    if save:
        np.savetxt('matrices/camera_matrix/extrinsic/' + prefix + '_camera_matrix', camera_matrix)
        np.savetxt('matrices/vectors/distortion/extrinsic/' + prefix + '_distortion_vector', dist_coeffs)
        np.savetxt('matrices/vectors/rotation/' + prefix + '_rotation_vector', rvecs[0])
        np.savetxt('matrices/vectors/translation/' + prefix + '_translation_vector', tvecs[0])
        np.savetxt('matrices/rotation_matrix/' + prefix + '_rotation_matrix', rotation_matrix)

    return camera_matrix, dist_coeffs, rotation_matrix, rvecs[0], tvecs[0]


if droite_ou_gauche == 'gauche':
    doug = 'gauche0'
elif droite_ou_gauche == 'droite':
    doug = 'droite548'

obj_points = np.loadtxt('matrices/points/points_objet/stereo_1_' + droite_ou_gauche + '_obj_points')
img_points = np.loadtxt('matrices/points/points_image/stereo_1_' + droite_ou_gauche + '_img_points')
img = cv2.imread('images/piste_camera_' + doug + '.jpg')
camera_matrix = np.loadtxt('matrices/camera_matrix/intrinsic/stereo_1_' + droite_ou_gauche + '_camera_matrix')
dist_coeffs = np.loadtxt('matrices/vectors/distortion/intrinsic/stereo_1_' + droite_ou_gauche + '_distortion_vector')

find_extrinsic_parameters(img, obj_points, img_points, camera_matrix, dist_coeffs, True, True,
'stereo_1_' + droite_ou_gauche)