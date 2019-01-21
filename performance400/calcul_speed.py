import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import cv2
from performance400.find_3D_coords_stereo import get_wrong_points_3d
from performance400.find_3D_coords_stereo import delete_wrong_points
from performance400.find_3D_coords_stereo import points_filtering
from performance400.find_3D_coords_stereo import find_3d_coords_stereo

VIDEO_REFRESH_RATE = 30
nbr = 1


def get_velocity(points_3d, show=True):
    ind_none = get_wrong_points_3d(points_3d)
    length = len(points_3d)
    t = np.arange(0, length, 1)
    points_3d = delete_wrong_points(points_3d, ind_none)
    points_filtering(points_3d)

    velocity = [
        np.linalg.norm(np.asarray(points_3d[i - 1, :1]) - np.asarray(points_3d[i + 1, :1])) * VIDEO_REFRESH_RATE / 2
        for i in range(1, len(points_3d) - 1)]

    for z in range(len(ind_none)):
        ind_none[z] -= z

    lset = dict([(k, ind_none.count(k)) for k in set(ind_none)])
    sum = 0
    for j in range(len(velocity)):
        if lset.get(j + 1) is not None:
            sum += lset.get(j + 1)
        if lset.get(j + 2) is not None:
            sum += lset.get(j + 2)
        if sum is not None and sum > 0:
            velocity[j] = velocity[j] * 2 / (sum + 2)
        sum = 0
    for k in ind_none:
        if k == 0 or k == length - 1:
            velocity.insert(0, np.mean(velocity))
        if 0 < k < length - 1:
            velocity.insert(k, (velocity[k - 1] + velocity[k]) / 2)

    velocity = np.array(velocity, 'float32')
    velocity = velocity * 3.6

    velocity = scipy.signal.savgol_filter(velocity, 21, 5)

    if show:
        plt.plot(t[1: len(t) - 1], velocity, marker='+')
        plt.xlabel("Distance parcourue en m")
        plt.ylabel("num frame")
        plt.show()
        print('moyenne vitesses', np.mean(velocity))
        print('ecart type velocity', np.std(velocity))

    return velocity, t


img_gauche = cv2.imread('images/piste_camera_gauche0.jpg')
img_droite = cv2.imread('images/piste_camera_droite548.jpg')

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

V = []
for i in range(nbr):
    positions_gauche = np.loadtxt('matrices/points/positions/stereo_1_homo_gauche_positions' + str(i))
    positions_droite = np.loadtxt('matrices/points/positions/stereo_1_homo_droite_positions' + str(i))

    points_3d = find_3d_coords_stereo(img_gauche, img_droite,
                                      camera_matrix_gauche,
                                      camera_matrix_droite, dist_coeffs_gauche, dist_coeffs_droite,
                                      rotation_matrix_gauche,
                                      rotation_matrix_droite, positions_gauche=positions_gauche,
                                      positions_droite=positions_droite,
                                      show=False, save=False, prefix='stereo_1', rvec_gauche=rvec_gauche,
                                      tvec_gauche=tvec_gauche,
                                      rvec_droite=rvec_droite, tvec_droite=tvec_droite)
    v, t = get_velocity(points_3d, False)
    V.append(v)
    x2 = None
for j in range(len(V)):
    x1 = plt.subplot(int(len(V) / 2) + 1, 2, j + 1, sharex=x2)
    x1.plot(t[1:len(t) - 1], V[j])
    x2 = plt.subplot(int(len(V) / 2) + 1, 2, j + 1, sharex=x1)
    x2.plot(t[1:len(t) - 1], V[j])
    plt.xlabel("Numereau de frame")
    plt.ylabel("Vitesse km/h")
    plt.title("Set de points " + str(j + 1))

VF = []
if nbr >1:
    for k in range(len(v)):
        vf = 0
        for l in range(len(V)):
            vf += V[l][k]
        vf = vf/len(V)
        VF.append(vf)
    print(len(VF))
    print(VF)
    plt.subplot(int(len(V) / 2) + 1, 2, nbr+1)
    plt.plot(t[1:len(t)-1], VF)
    plt.xlabel("Numereau de frame")
    plt.ylabel("Vitesse m/s")
    plt.title("Vitesse moyenne des points")
plt.show()