import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def find_3d_coords_stereo(img_gauche, img_droite, obj_points, img_points_gauche, img_points_droite,
                          camera_matrix_gauche, camera_matrix_droite, dist_coeffs_gauche, dist_coeffs_droite,
                          rotation_matrix_gauche, rotation_matrix_droite,
                          positions_gauche=None, positions_droite=None, show=False, save=False, prefix='',
                          rvec_gauche=None, tvec_gauche=None, rvec_droite=None, tvec_droite=None):
    obj_points = np.array(obj_points, 'float32')
    img_points_gauche = np.array(img_points_gauche, 'float32')
    img_points_droite = np.array(img_points_droite, 'float32')
    size = img_gauche.shape[:2]


    R1 = np.zeros(shape=(3, 3))
    R2 = np.zeros(shape=(3, 3))
    P1 = np.zeros(shape=(3, 3))
    P2 = np.zeros(shape=(3, 3))

    stereo_flag = cv2.CALIB_FIX_INTRINSIC
    retval, camera_matrix_gauche, dist_coeffs_gauche, \
    camera_matrix_droite, dist_coeffs_droite, R, T, E, F = cv2.stereoCalibrate([obj_points], [img_points_gauche],
                                                                               [img_points_droite],
                                                                               camera_matrix_gauche, dist_coeffs_gauche,
                                                                               camera_matrix_droite, dist_coeffs_droite,
                                                                               size, flags=stereo_flag)

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(camera_matrix_gauche, dist_coeffs_gauche,
                                                                      camera_matrix_droite, dist_coeffs_droite,
                                                                      size, R, T, R1, R2, P1, P2,
                                                                      cv2.CALIB_ZERO_DISPARITY, alpha=0)
    tvec_gauche= np.array([tvec_gauche])
    tvec_droite = np.array([tvec_droite])
    m1 = np.append(rotation_matrix_gauche, tvec_gauche.T, axis=1)
    m2 = np.append(rotation_matrix_droite, tvec_droite.T, axis=1)

    projection_matrix_1 = camera_matrix_gauche @ m1
    projection_matrix_2 = camera_matrix_droite @ m2
    print('Q')
    print(Q)
    print('P1_1')
    print(projection_matrix_1)
    print('P1_2')
    print(P1)
    print('P2_1')
    print(projection_matrix_2)
    print('P2_2')
    print(P2)


    if positions_gauche == None or positions_droite == None:
        positions_gauche = img_points_gauche
        positions_droite = img_points_droite
    points4D = cv2.triangulatePoints(projection_matrix_1, projection_matrix_2, positions_gauche.T, positions_droite.T)
    points3D = cv2.convertPointsFromHomogeneous(points4D.T)

    if save:
        if prefix == '':
            return print('Veuillez saisir un prefix')
        points3D_bis = []
        for punkt in points3D:
            points3D_bis.append([punkt[0][0], punkt[0][1], punkt[0][2]])
        np.savetxt('matrices/camera_matrix/stereo_calib/' + prefix + '_gauche_camera_matrix', camera_matrix_gauche)
        np.savetxt('matrices/camera_matrix/stereo_calib/' + prefix + '_droite_camera_matrix', camera_matrix_droite)
        np.savetxt('matrices/vectors/distortion/stereo_calib/' + prefix + '_gauche_distortion_vector',
                   dist_coeffs_gauche)
        np.savetxt('matrices/vectors/distortion/stereo_calib/' + prefix + '_droite_distortion_vector',
                   dist_coeffs_droite)
        np.savetxt('matrices/stereo_calib/' + prefix + '_R', R)
        np.savetxt('matrices/stereo_calib/' + prefix + '_T', T)
        np.savetxt('matrices/stereo_calib/' + prefix + '_E', E)
        np.savetxt('matrices/stereo_calib/' + prefix + '_F', F)
        np.savetxt('matrices/stereo_rectify/' + prefix + '_R_gauche', R1)
        np.savetxt('matrices/stereo_rectify/' + prefix + '_R_droite', R2)
        np.savetxt('matrices/stereo_rectify/' + prefix + '_P_gauche', P1)
        np.savetxt('matrices/stereo_rectify/' + prefix + '_P_droite', P2)
        np.savetxt('matrices/stereo_rectify/' + prefix + '_Q', Q)
        np.savetxt('matrices/points/points3D/' + prefix + '_points_3d', points3D_bis)

    if show:
        if len(rvec_gauche) == 0 or len(rvec_droite) == 0 or len(tvec_gauche) == 0 or len(tvec_droite) == 0:
            return print('Veuillez donner les vecteurs de rotation et translation gauche et droite')

        ax = plt.axes(projection='3d')
        ax.scatter3D(points3D.T[0, :], points3D.T[1, :], points3D.T[2, :], 'blue')
        plt.show()

        img_gauche = cv2.drawFrameAxes(img_gauche, camera_matrix_gauche, dist_coeffs_gauche, rvec_gauche, tvec_gauche,
                                       3)
        img_points2, jacobian = cv2.projectPoints(points3D, rvec_gauche, tvec_gauche,
                                                  camera_matrix_gauche, dist_coeffs_gauche)
        for px in img_points2:
            cv2.circle(img_gauche, (px[0][0], px[0][1]), 3, (255, 255, 0), 20)
        img_droite = cv2.drawFrameAxes(img_droite, camera_matrix_droite, dist_coeffs_droite, rvec_droite, tvec_droite,
                                       3)
        img_points3, jacobian = cv2.projectPoints(points3D, rvec_droite, tvec_droite,
                                                  camera_matrix_droite, dist_coeffs_droite)
        for px in img_points3:
            cv2.circle(img_droite, (px[0][0], px[0][1]), 3, (0, 0, 255), 20)

        cv2.namedWindow('image gauche', cv2.WINDOW_NORMAL)
        cv2.imshow('image gauche', img_gauche)
        cv2.namedWindow('image droite', cv2.WINDOW_NORMAL)
        cv2.imshow('image droite', img_droite)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return points3D_bis


img_gauche = cv2.imread('images/piste_camera_gauche0.jpg')
img_droite = cv2.imread('images/piste_camera_droite548.jpg')

obj_points = np.loadtxt('matrices/points/points_objet/stereo_1_obj_points')
img_points_gauche = np.loadtxt('matrices/points/points_image/stereo_1_gauche_img_points')
img_points_droite = np.loadtxt('matrices/points/points_image/stereo_1_droite_img_points')

camera_matrix_gauche = np.loadtxt('matrices/camera_matrix/extrinsic/stereo_1_gauche_camera_matrix')
camera_matrix_droite = np.loadtxt('matrices/camera_matrix/extrinsic/stereo_1_droite_camera_matrix')
dist_coeffs_gauche = np.loadtxt('matrices/vectors/distortion/extrinsic/stereo_1_gauche_distortion_vector')
dist_coeffs_droite = np.loadtxt('matrices/vectors/distortion/extrinsic/stereo_1_droite_distortion_vector')

rvec_gauche = np.loadtxt('matrices/vectors/rotation/stereo_1_gauche_rotation_vector')
rvec_droite = np.loadtxt('matrices/vectors/rotation/stereo_1_droite_rotation_vector')
tvec_gauche = np.loadtxt('matrices/vectors/translation/stereo_1_gauche_translation_vector')
tvec_droite = np.loadtxt('matrices/vectors/translation/stereo_1_droite_translation_vector')

rotation_matrix_gauche = np.loadtxt('matrices/rotation_matrix/stereo_1_gauche_rotation_matrix')
rotation_matrix_droite = np.loadtxt('matrices/rotation_matrix/stereo_1_droite_rotation_matrix')

find_3d_coords_stereo(img_gauche, img_droite, obj_points, img_points_gauche, img_points_droite, camera_matrix_gauche,
                      camera_matrix_droite, dist_coeffs_gauche, dist_coeffs_droite, rotation_matrix_gauche,
                      rotation_matrix_droite, positions_gauche=None,
                      positions_droite=None, show=True, save=True, prefix='stereo_1', rvec_gauche=rvec_gauche,
                      tvec_gauche=tvec_gauche, rvec_droite=rvec_droite, tvec_droite=tvec_droite)
