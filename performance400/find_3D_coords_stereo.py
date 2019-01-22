import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math as m
import scipy.signal

num_course = '2'
def get_wrong_points_3d(points_3d):
    ind_none = []
    for n in range(np.shape(points_3d)[0]):
        if abs(points_3d[n][2]) > 6 or abs(points_3d[0][2]) > 1e16 or abs(points_3d[1][2]) > 1e16:
            ind_none.append(n)
        if n > 0 and not ind_none.__contains__(n) and (
                abs(points_3d[n][0] - points_3d[n - 1][0]) > 0.5 or abs(points_3d[n][1] - points_3d[n - 1][1]) > 0.5):
            ind_none.append(n)
        ind_none.sort()
    return ind_none


def get_wrong_points_3d_2(points_3d):
    ind_none = []
    for n in range(np.shape(points_3d)[0]):
        if abs(points_3d[n][2]) > 1e16:
            ind_none.append(n)
        ind_none.sort()
    return ind_none


def get_positions_fails(positions_gauche, positions_droite):
    ind_fails = []
    for n in range(np.shape(positions_gauche)[0]):
        if abs(positions_gauche[n][0]) > 1e16 or abs(positions_gauche[n][1]) > 1e16 \
                or abs(positions_droite[n][0]) > 1e16 or abs(positions_droite[n][1]) > 1e16:
            ind_fails.append(n)

    return ind_fails


def delete_wrong_points(points_3d, ind_none):
    count = 0
    for k in ind_none:
        points_3d = np.delete(points_3d, (k - count), axis=0)
        count += 1

    return points_3d


def delete_positions_fails(positions_gauche, positions_droite, ind_fails):
    count = 0
    for k in ind_fails:
        positions_gauche = np.delete(positions_gauche, (k - count), axis=0)
        positions_droite = np.delete(positions_droite, (k - count), axis=0)
        count += 1
    return positions_gauche, positions_droite


def points_filtering(points_3d):
    points_3d.T[0] = scipy.signal.savgol_filter(points_3d.T[0], 21, 5)
    points_3d.T[1] = scipy.signal.savgol_filter(points_3d.T[1], 21, 5)
    points_3d.T[2] = scipy.signal.savgol_filter(points_3d.T[2], 21, 5)


def find_3d_coords_stereo(img_gauche, img_droite, camera_matrix_gauche, camera_matrix_droite, dist_coeffs_gauche,
                          dist_coeffs_droite, rotation_matrix_gauche, rotation_matrix_droite,
                          positions_gauche, positions_droite, show=False, save=False, prefix='',
                          rvec_gauche=None, tvec_gauche=None, rvec_droite=None, tvec_droite=None):
    size = img_gauche.shape[:2]

    tvec_gauche = np.array([tvec_gauche])
    tvec_droite = np.array([tvec_droite])
    m1 = np.append(rotation_matrix_gauche, tvec_gauche.T, axis=1)
    m2 = np.append(rotation_matrix_droite, tvec_droite.T, axis=1)

    projection_matrix_1 = camera_matrix_gauche @ m1
    projection_matrix_2 = camera_matrix_droite @ m2

    ind_fail = get_positions_fails(positions_gauche, positions_droite)

    positions_gauche, positions_droite = delete_positions_fails(positions_gauche, positions_droite, ind_fail)
    positions_gauche.T[0] = scipy.signal.savgol_filter(positions_gauche.T[0], 21, 5)
    positions_gauche.T[1] = scipy.signal.savgol_filter(positions_gauche.T[1], 21, 5)
    positions_droite.T[0] = scipy.signal.savgol_filter(positions_droite.T[0], 21, 5)
    positions_droite.T[1] = scipy.signal.savgol_filter(positions_droite.T[1], 21, 5)

    points_4d = cv2.triangulatePoints(projection_matrix_1, projection_matrix_2, positions_gauche.T,
                                      positions_droite.T)
    points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)

    points_3d_bis = []
    for p in points_3d:
        points_3d_bis.append(p[0])
    points_3d_bis = np.asarray(points_3d_bis)

    ind_none = get_wrong_points_3d(points_3d_bis)

    if show:
        img_gauche = cv2.drawFrameAxes(img_gauche, camera_matrix_gauche, dist_coeffs_gauche, rvec_gauche, tvec_gauche,
                                       1)
        img_points2, jacobian = cv2.projectPoints(points_3d_bis, rvec_gauche, tvec_gauche,
                                                  camera_matrix_gauche, dist_coeffs_gauche)

        for j in range(len(img_points2)):
            if 0 < img_points2[j][0][0] < size[1] and 0 < img_points2[j][0][1] < size[0]:
                if ind_none.__contains__(j):
                    cv2.circle(img_gauche, (m.floor((img_points2[j])[0][0]), m.floor((img_points2[j])[0][1])), 3,
                               (0, 0, 255), 20)
                else:
                    cv2.circle(img_gauche, (m.floor(img_points2[j][0][0]), m.floor(img_points2[j][0][1])), 3,
                               (255, 0, 0), 20)

        img_droite = cv2.drawFrameAxes(img_droite, camera_matrix_droite, dist_coeffs_droite, rvec_droite, tvec_droite,
                                       2)
        img_points3, jacobian = cv2.projectPoints(points_3d_bis, rvec_droite, tvec_droite,
                                                  camera_matrix_droite, dist_coeffs_droite)

        for j in range(len(img_points3)):
            if 0 < img_points3[j][0][0] < size[1] and 0 < img_points3[j][0][1] < size[0]:
                if ind_none.__contains__(j):
                    cv2.circle(img_droite, (m.floor(img_points3[j][0][0]), m.floor(img_points3[j][0][1])),
                               3, (0, 0, 255), 20)
                else:
                    cv2.circle(img_droite, (m.floor(img_points3[j][0][0]), m.floor(img_points3[j][0][1])),
                               3, (255, 0, 0), 20)

        cv2.namedWindow('image gauche', cv2.WINDOW_NORMAL)
        cv2.imshow('image gauche', img_gauche)
        cv2.namedWindow('image droite', cv2.WINDOW_NORMAL)
        cv2.imshow('image droite', img_droite)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    IND_NONE = [ind_none]
    points_3d_bis = delete_wrong_points(points_3d_bis, ind_none)
    while len(IND_NONE[-1]) > 1:
        IND_NONE.append(get_wrong_points_3d(points_3d_bis))
        points_3d_bis = delete_wrong_points(points_3d_bis, IND_NONE[-1])
    if show:
        if len(rvec_gauche) == 0 or len(rvec_droite) == 0 or len(tvec_gauche) == 0 or len(tvec_droite) == 0:
            return print('Veuillez donner les vecteurs de rotation et translation gauche et droite')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(points_3d_bis.T[0, :], points_3d_bis.T[1, :], points_3d_bis.T[2, :], 'blue')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        plt.show()

    for ind_nonei in IND_NONE:
        for g in ind_nonei:
            points_3d_bis = np.append(np.append(points_3d_bis[:g], [[1e17, 1e17, 1e17]], axis=0),
                                      points_3d_bis[g:], axis=0)
    for x in ind_fail:
        points_3d_bis = np.append(np.append(points_3d_bis[:x], [[1e17, 1e17, 1e17]], axis=0),
                                  points_3d_bis[x:], axis=0)
    if save:
        if prefix == '':
            return print('Veuillez saisir un prefix')
        np.savetxt('matrices/points/points3D/' + prefix + '_points_3d', points_3d_bis)
    return points_3d_bis


video = cv2.VideoCapture("/home/colozz/workspace/performance400/performance400/videos/runway/Course 2 gauche SD.mkv")
img_gauche = video.read()[1]
video.release()
video = cv2.VideoCapture("/home/colozz/workspace/performance400/performance400/videos/runway/Course 2 droite SD.mkv")
img_droite = video.read()[1]
video.release()


camera_matrix_gauche = np.loadtxt('matrices/camera_matrices/extrinsic/stereo_'+num_course+'_gauche_camera_matrix')
camera_matrix_droite = np.loadtxt('matrices/camera_matrices/extrinsic/stereo_'+num_course+'_droite_camera_matrix')
dist_coeffs_gauche = np.loadtxt('matrices/vectors/distortion/extrinsic/stereo_'+num_course+'_gauche_distortion_vector')
dist_coeffs_droite = np.loadtxt('matrices/vectors/distortion/extrinsic/stereo_'+num_course+'_droite_distortion_vector')

rvec_gauche = np.loadtxt('matrices/vectors/rotation/stereo_'+num_course+'_gauche_rotation_vector')
rvec_droite = np.loadtxt('matrices/vectors/rotation/stereo_'+num_course+'_droite_rotation_vector')
tvec_gauche = np.loadtxt('matrices/vectors/translation/stereo_'+num_course+'_gauche_translation_vector')
tvec_droite = np.loadtxt('matrices/vectors/translation/stereo_'+num_course+'_droite_translation_vector')

rotation_matrix_gauche = np.loadtxt('matrices/rotation_matrix/stereo_'+num_course+'_gauche_rotation_matrix')
rotation_matrix_droite = np.loadtxt('matrices/rotation_matrix/stereo_'+num_course+'_droite_rotation_matrix')

positions_gauche = np.loadtxt('matrices/points/positions/stereo_'+num_course+'_gauche')
positions_droite = np.loadtxt('matrices/points/positions/stereo_'+num_course+'_droite')

points_3d = find_3d_coords_stereo(img_gauche, img_droite,
                                  camera_matrix_gauche, camera_matrix_droite,
                                  dist_coeffs_gauche, dist_coeffs_droite,
                                  rotation_matrix_gauche, rotation_matrix_droite,
                                  positions_gauche=positions_gauche, positions_droite=positions_droite,
                                  show=True, save=True, prefix='stereo_'+num_course,
                                  rvec_gauche=rvec_gauche,
                                  tvec_gauche=tvec_gauche,
                                  rvec_droite=rvec_droite,
                                  tvec_droite=tvec_droite)
