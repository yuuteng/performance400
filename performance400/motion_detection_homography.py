import cv2
import numpy as np
import matplotlib.pyplot as plot
import scipy.signal

from performance400.find_3D_coords_mono import calculate_3d_coords

left_calibration_points = np.array(
    [(735, 280), (219, 788), (797, 698), (1054, 906), (182, 1041)]) * 2
right_calibration_points = np.array(
    [(1486, 270), (948, 688), (1491, 702), (1682, 945), (848, 886)]) * 2
H = np.eye(3)
count = 0
nbr = 10


def get_bf_matches(m_H, m_left_image, m_right_image, m_kp1, m_kp2, m_des1, m_des2, nbr):
    m_matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    m_matches = m_matcher.match(m_des1, m_des2)

    # list.sort(m_matches, key=lambda e: get_transfer_error(e, kp1, kp2))
    list.sort(m_matches, key=lambda e: get_transfer_error(m_H, e, m_kp1, m_kp2))

    m_good = []
    m_best = []
    e = []
    count2 = 0
    for match in m_matches:
        m_good.append(match)
        e.append(get_transfer_error(m_H, match, m_kp1, m_kp2))
    length = len(m_good)
    ind1 = 0
    ind2 = 0
    while ind2 < length:
        if e[ind1] == min(e):
            m_best.append(m_good[ind2])
            e.remove(min(e))
            count2 += 1
            ind1 -= 1
        ind1 += 1
        ind2 += 1
        if count2 > nbr - 1:
            break
    return m_best


def get_flann_matches(m_H, m_left_image, m_right_image, m_kp1, m_kp2, m_des1, m_des2):
    FLANN_INDEX_LSH = 6
    m_index_params = dict(algorithm=FLANN_INDEX_LSH,
                          table_number=12,  # 12
                          key_size=1,  # 20
                          multi_probe_level=1)  # 2
    m_search_params = dict(checks=50)

    m_flann = cv2.FlannBasedMatcher(m_index_params, m_search_params)
    m_matches = m_flann.knnMatch(m_des1, m_des2, k=2)
    m_matches_mask = [[0, 0] for j in range(len(m_matches))]
    m_good = []

    for j in range(len(m_matches)):
        if len(m_matches[j]) != 2:
            continue
        (m, n) = m_matches[j]
        if m.distance < 0.7 * n.distance:
            m_matches_mask[j] = [1, 0]
            m_good.append(n)

    m_filtered_good = []
    for good in m_good:
        if get_transfer_error(m_H, good, m_kp1, m_kp2) < 4:
            m_filtered_good.append(good)

    # m_draw_params = dict(matchColor=(0, 255, 0),
    #                      singlePointColor=(255, 0, 0),
    #                      matchesMask=m_matches_mask,
    #                      flags=0)

    # m_img = cv2.drawMatchesKnn(m_left_image, m_kp1, m_right_image, m_kp2, m_matches, None, **m_draw_params)

    return m_filtered_good


def get_transfer_error(m_H, m_match, m_kp1, m_kp2):
    m_xp = np.transpose([m_kp1[m_match.queryIdx].pt[0], m_kp1[m_match.queryIdx].pt[1], 1])
    m_xpp = np.transpose([m_kp2[m_match.trainIdx].pt[0], m_kp2[m_match.trainIdx].pt[1], 1])

    m_transformed_xp = m_H @ m_xp
    m_transformed_xp[2] = 1

    m_transformed_xpp = np.linalg.inv(m_H) @ m_xpp
    m_transformed_xpp[2] = 1

    e1 = (np.linalg.norm(m_transformed_xp - m_xpp) * 0.01) ** 2
    e2 = (np.linalg.norm(m_transformed_xpp - m_xp) * 0.01) ** 2

    # print(f"1 {m_xp} -> {m_transformed_xp} vs {m_xpp} ; e: {e1}")
    # print(f"2 {m_xpp} -> {m_transformed_xpp} vs {m_xp} ; e: {e2}")

    return e1 + e2


# Détermine les formes qui ont changé par rapport à background
def get_frames(m_frame, m_background):
    m_gray_frame = cv2.cvtColor(m_frame, cv2.COLOR_BGR2GRAY)
    m_gray_frame = cv2.GaussianBlur(m_gray_frame, (GAUSSIAN_BLUR, GAUSSIAN_BLUR), 0)

    if m_background is None:
        m_background = m_gray_frame

    m_difference_frame = cv2.absdiff(m_background, m_gray_frame)
    m_threshold_frame = cv2.threshold(m_difference_frame, DETECTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    # m_threshold_frame =cv2.adaptiveThreshold(m_difference_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    m_threshold_frame = cv2.dilate(m_threshold_frame, None, iterations=NUMBER_OF_DILATATION)

    return m_gray_frame, m_difference_frame, m_threshold_frame, m_background


# Renvoie le plus grand contours de la collection m_contours
def get_largest_contour(m_contours):
    m_largest_contour = m_contours[0]

    for m_contour in m_contours:
        if cv2.contourArea(m_contour) > cv2.contourArea(m_largest_contour):
            m_largest_contour = m_contour

    return m_largest_contour


# Dessine la trajectoire du centre du plus grand contour sur m_frame
def draw_trajectory(m_trajectory, m_frame, m_color=(0, 255, 0)):
    cv2.polylines(m_frame, [np.array(m_trajectory, 'int32').reshape((-1, 1, 2))], isClosed=False, color=m_color,
                  thickness=10, lineType=cv2.LINE_AA)

    return m_frame


# Enleve les pints indésirable, retourne les points et les index
def trajectory_cleaning_wrong_points(m_trajectory, m_window_length=10, m_threshold=3):
    m_consecutive_squared_distances = [0]
    for j in range(1, len(m_trajectory)):
        m_consecutive_squared_distances.append(
            (m_trajectory[j][0] - m_trajectory[j - 1][0]) ** 2 + (m_trajectory[j][1] - m_trajectory[j - 1][1]) ** 2)
    m_corrected_trajectory = []
    m_corrected_trajectory_missing_index = []
    m_corrected_trajectory_suppr = []
    m_corrected_trajectory_suppr_index = []
    for j in range(m_window_length, len(m_trajectory) - m_window_length):
        if (m_trajectory[j][0] - m_trajectory[j - 1][0]) ** 2 + (
                m_trajectory[j][1] - m_trajectory[j - 1][1]) ** 2 < m_threshold * np.mean(
            m_consecutive_squared_distances[j - m_window_length:j + m_window_length]):
            m_corrected_trajectory.append(m_trajectory[j])
            m_corrected_trajectory_suppr.append(m_trajectory[j])
            m_corrected_trajectory_suppr_index.append(j)
        else:
            m_corrected_trajectory.append((None, None))
            m_corrected_trajectory_missing_index.append(j)

    m_corrected_trajectory = trajectory_filtering(m_corrected_trajectory)

    return m_corrected_trajectory_suppr, m_corrected_trajectory_suppr_index, m_corrected_trajectory, m_corrected_trajectory_missing_index


# utilise la fonction trajectory_cleaning et np.transpose pour ressortir une courbe lissé
def trajectory_reconstructing(m_trajectory):
    a, b, c, d = trajectory_cleaning_wrong_points(m_trajectory)
    Ax = np.transpose(a)[0]
    Ay = np.transpose(a)[1]
    Amissx = np.interp(d, b, Ax)
    Amissy = np.interp(d, b, Ay)
    m_merge_trajectory = c[::]
    indexA = 0
    indexM = 0
    k = 0
    while k < len(m_merge_trajectory) and indexM < len(d) and indexA < len(b):
        if (b[indexA] < d[indexM]):
            m_merge_trajectory[k] = (Ax[indexA], Ay[indexA])
            indexA += 1
        else:
            m_merge_trajectory[k] = (Amissx[indexM], Amissy[indexM])
            indexM += 1
        k += 1
    return m_merge_trajectory


# Filtre la trajectoire m_trajectory
def trajectory_filtering(m_trajectory):
    m_shaped_trajectory = np.transpose(np.asarray(m_trajectory))
    m_filtered_x = scipy.signal.savgol_filter(m_shaped_trajectory[0], 21, 5)
    m_filtered_y = scipy.signal.savgol_filter(m_shaped_trajectory[1], 21, 5)

    m_filtered_trajectory = [(m_filtered_x[k], m_filtered_y[k]) for k in range(0, len(m_filtered_x))]

    return m_filtered_trajectory
    # return m_trajectory


FIRST_FRAME_INDEX = 0
LAST_FRAME_INDEX = 210
VIDEO_REFRESH_RATE = 30
DETECTION_THRESHOLD = 10
MIN_CONTOUR_AREA = 2000
GAUSSIAN_BLUR = 25
NUMBER_OF_DILATATION = 2

videog = cv2.VideoCapture('videos/runway/gauche.mp4')
videod = cv2.VideoCapture('videos/runway/droite.mp4')
# video = cv2.VideoCapture('videos/runway/droite.mp4')
# pensez à faire le changement de matrice dans find_coord_3D si on change de video

frame_width = int(videog.get(3))
frame_height = int(videog.get(4))

backgroundg = None
backgroundd = None

corners_trajectories_gauche = [[], [], [], []]
corners_trajectories_droite = [[], [], [], []]  # Top left hand corner then CCW
trajectory_camera_coord_gauche = [[]]
trajectory_camera_coord_droite = [[]]

for i in range(-FIRST_FRAME_INDEX, LAST_FRAME_INDEX):
    # On s'assure que la frame courante est bonne et nous intéresse
    frameg = videog.read()[1]
    framed = videod.read()[1]
    if frameg is None or framed is None:
        break

    if i < 0:
        continue

    # On récupère les formes en mouvement
    gray_frame_gauche, difference_frame_gauche, threshold_frame_gauche, backgroundg = get_frames(frameg, backgroundg)
    gray_frame_droite, difference_frame_droite, threshold_frame_droite, backgroundd = get_frames(framed, backgroundd)

    # On détermine leurs contours
    contours_gauche = cv2.findContours(threshold_frame_gauche.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    contours_droite = cv2.findContours(threshold_frame_droite.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]

    if contours_gauche is not None and contours_droite is not None:
        if len(contours_gauche) > 0 and len(contours_droite) > 0:
            # On récupère la plus grande forme, et si elle est assez grande, on dessine son contour, on détermine son
            # centre et on calcule sa trajectoire
            largest_contour_gauche = get_largest_contour(contours_gauche)
            largest_contour_droite = get_largest_contour(contours_droite)

            if cv2.contourArea(largest_contour_gauche) > MIN_CONTOUR_AREA \
                    and cv2.contourArea(largest_contour_droite) > MIN_CONTOUR_AREA:
                (xg, yg, wg, hg) = cv2.boundingRect(largest_contour_gauche)
                (xd, yd, wd, hd) = cv2.boundingRect(largest_contour_droite)

                cv2.rectangle(frameg, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 3)
                cv2.rectangle(framed, (xd, yd), (xd + wd, yd + hd), (0, 255, 0), 3)

                orb = cv2.ORB_create()
                kp1, des1 = orb.detectAndCompute(frameg[yg:yg + hg, xg:xg + wg], None)
                kp2, des2 = orb.detectAndCompute(framed[yd:yd + hd, xd:xd + wd], None)
                if (kp1 is None or kp2 is None or des1 is None or des2 is None):
                    for ind in range(nbr):
                        trajectory_camera_coord_gauche = np.append(trajectory_camera_coord_gauche, [[1e17, 1e17]],
                                                                   axis=0)
                        trajectory_camera_coord_droite = np.append(trajectory_camera_coord_droite, [[1e17, 1e17]],
                                                                   axis=0)
                    continue
                matches = get_bf_matches(H, frameg[yg:yg + hg, xg:xg + wg], framed[yd:yd + hd, xd:xd + wd],
                                         kp1, kp2, des1, des2, nbr)
                left_point = np.asarray([kp1[matches[i].queryIdx].pt for i in range(len(matches))])
                right_point = np.asarray([kp2[matches[i].trainIdx].pt for i in range(len(matches))])
                left_point[:, 0] += xg
                left_point[:, 1] += yg
                right_point[:, 0] += xd
                right_point[:, 1] += yd
                while len(left_point) < nbr:
                    left_point = np.append(left_point, [[1e17, 1e17]], axis=0)
                    right_point = np.append(right_point, [[1e17, 1e17]], axis=0)

                img = cv2.drawMatches(frameg[yg:yg + hg, xg:xg + wg], kp1, framed[yd:yd + hd, xd:xd + wd],
                                      kp2, matches, None, flags=2)

                cv2.namedWindow("ok", cv2.WINDOW_NORMAL)
                cv2.imshow("ok", img)
                key = cv2.waitKey(1)
                if key == 'q':
                    break
                if count > 0:
                    trajectory_camera_coord_gauche = np.append(trajectory_camera_coord_gauche, left_point, axis=0)
                    trajectory_camera_coord_droite = np.append(trajectory_camera_coord_droite, right_point, axis=0)
                if count == 0:
                    trajectory_camera_coord_gauche = left_point
                    trajectory_camera_coord_droite = right_point
                    count += 1

                incr = 0

while incr < nbr:
    trajectory_camera_coord_gauche_bis = [trajectory_camera_coord_gauche[0+incr].tolist()]
    trajectory_camera_coord_droite_bis = [trajectory_camera_coord_droite[0+incr].tolist()]
    for indice in range(1, int(len(trajectory_camera_coord_gauche) / nbr)):
        trajectory_camera_coord_gauche_bis = np.append(trajectory_camera_coord_gauche_bis,
                                                       [trajectory_camera_coord_gauche[incr + nbr * indice].tolist()],
                                                       axis=0)
        trajectory_camera_coord_droite_bis = np.append(trajectory_camera_coord_droite_bis,
                                                       [trajectory_camera_coord_droite[incr + nbr * indice].tolist()],
                                                       axis=0)
    np.savetxt('matrices/points/positions/stereo_1_homo_gauche_positions' + str(incr),
               trajectory_camera_coord_gauche_bis)
    np.savetxt('matrices/points/positions/stereo_1_homo_droite_positions' + str(incr),
               trajectory_camera_coord_droite_bis)
    incr += 1
