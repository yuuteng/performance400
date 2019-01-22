import numpy as np
import matplotlib.pyplot as plt
import cv2
from performance400.unused.find_3D_coords_stereo import delete_wrong_points
from performance400.unused.find_3D_coords_stereo import find_3d_coords_stereo
from performance400.unused.find_3D_coords_stereo import get_wrong_points_3d_2

VIDEO_REFRESH_RATE = 30
nbr = 1
num_course = '2'


def get_velocity(points_3d, show=True):
    ind_none = get_wrong_points_3d_2(points_3d)
    length = len(points_3d)
    t = np.arange(0, length, 1)
    points_3d = delete_wrong_points(points_3d, ind_none)
    velocity = [
        np.linalg.norm(np.asarray(points_3d[i - 1, :2]) - np.asarray(points_3d[i + 1, :2])) * VIDEO_REFRESH_RATE / 2
        for i in range(2, len(points_3d) - 1)]
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
    ind_none.reverse()
    for k in ind_none:
        velocity.insert(k, -100)
    velocity_connu = []
    ind_connus = []
    ind_inconnus = []
    for l in range(len(velocity)):
        if abs(velocity[l]) < 50:
            velocity_connu.append(velocity[l])
            ind_connus.append(l)
        else:
            ind_inconnus.append(l)
    for q in ind_inconnus:
        velocity[q] = np.interp(q, ind_connus, velocity_connu)

    velocity = np.array(velocity, 'float32')
    velocity = velocity * 3.6
    # velocity = scipy.signal.savgol_filter(velocity, 21, 5)
    if show:
        plt.plot(t[2: len(t) - 1], velocity, marker='+')
        plt.xlabel("Numero frame")
        plt.ylabel("vitesse m/s")
        plt.show()
        print('moyenne vitesses', np.mean(velocity))
        print('ecart type velocity', np.std(velocity))

    return velocity, t


video = cv2.VideoCapture("/home/colozz/workspace/performance400/performance400/videos/runway/left_run.mkv")
img_gauche = video.read()[1]
video.release()
video = cv2.VideoCapture("/home/colozz/workspace/performance400/performance400/videos/runway/right_run.mkv")
img_droite = video.read()[1]
video.release()

camera_matrix_gauche = np.loadtxt('matrices/camera_matrices/extrinsic/stereo_' + num_course + '_gauche_camera_matrix')
camera_matrix_droite = np.loadtxt('matrices/camera_matrices/extrinsic/stereo_' + num_course + '_droite_camera_matrix')
dist_coeffs_gauche = np.loadtxt(
    'matrices/vectors/distortion/extrinsic/stereo_' + num_course + '_gauche_distortion_vector')
dist_coeffs_droite = np.loadtxt(
    'matrices/vectors/distortion/extrinsic/stereo_' + num_course + '_droite_distortion_vector')

rvec_gauche = np.loadtxt('matrices/vectors/rotation/stereo_' + num_course + '_gauche_rotation_vector')
rvec_droite = np.loadtxt('matrices/vectors/rotation/stereo_' + num_course + '_droite_rotation_vector')
tvec_gauche = np.loadtxt('matrices/vectors/translation/stereo_' + num_course + '_gauche_translation_vector')
tvec_droite = np.loadtxt('matrices/vectors/translation/stereo_' + num_course + '_droite_translation_vector')

rotation_matrix_gauche = np.loadtxt('matrices/rotation_matrix/stereo_' + num_course + '_gauche_rotation_matrix')
rotation_matrix_droite = np.loadtxt('matrices/rotation_matrix/stereo_' + num_course + '_droite_rotation_matrix')

V = []
for i in range(nbr):
    positions_gauche = np.loadtxt('matrices/points/positions/stereo_' + num_course + '_gauche_positions')
    positions_droite = np.loadtxt('matrices/points/positions/stereo_' + num_course + '_droite_positions')

    points_3d = find_3d_coords_stereo(img_gauche, img_droite,
                                      camera_matrix_gauche,
                                      camera_matrix_droite, dist_coeffs_gauche, dist_coeffs_droite,
                                      rotation_matrix_gauche,
                                      rotation_matrix_droite, positions_gauche=positions_gauche,
                                      positions_droite=positions_droite,
                                      show=False, save=False, rvec_gauche=rvec_gauche,
                                      tvec_gauche=tvec_gauche,
                                      rvec_droite=rvec_droite, tvec_droite=tvec_droite)
    v, t = get_velocity(points_3d, False)
    V.append(v)
    x2 = None

if nbr > 1:
    for j in range(len(V)):
        x1 = plt.subplot(int(len(V) / 2) + 1, 2, j + 1, sharex=x2)
        x1.plot(t[2:len(t) - 1], V[j])
        x2 = plt.subplot(int(len(V) / 2) + 1, 2, j + 1, sharex=x1)
        x2.plot(t[2:len(t) - 1], V[j])
        plt.xlabel("Numero de frame")
        plt.ylabel("Vitesse km/h")
        plt.title("Set de points " + str(j + 1))

VF = []
if nbr > 1:
    for k in range(len(v)):
        vf = 0
        for l in range(len(V)):
            vf += V[l][k]
        vf = vf / len(V)
        VF.append(vf)

    plt.subplot(int(len(V) / 2) + 1, 2, nbr + 1)
    plt.plot(t[2:len(t) - 1], VF)
    plt.xlabel("Numero de frame")
    plt.ylabel("Vitesse m/s")
    plt.title("Vitesse moyenne des points")

if nbr == 1:
    plt.title("Set de points " + str(1))
    plt.xlabel("Numero de frame")
    plt.ylabel("Vitesse km/h")
    plt.plot(t[2:len(t) - 1], V[0])
plt.show()
