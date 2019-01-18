import cv2
import numpy as np
import matplotlib.pyplot as plot
import scipy.signal

from performance400.find_3D_coords_mono import calculate_3d_coords


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

droiteougauche='droite'
video = cv2.VideoCapture('videos/runway/'+droiteougauche+'.mp4')
# video = cv2.VideoCapture('videos/runway/droite.mp4')
# pensez à faire le changement de matrice dans find_coord_3D si on change de video

frame_width = int(video.get(3))
frame_height = int(video.get(4))

video_save = out = cv2.VideoWriter('videos/Results/motion_detection.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                   10, (frame_width, frame_height))

background = None

corners_trajectories = [[], [], [], []]  # Top left hand corner then CCW
trajectory_camera_coord = []

for i in range(-FIRST_FRAME_INDEX, LAST_FRAME_INDEX):
    # On s'assure que la frame courante est bonne et nous intéresse
    frame = video.read()[1]
    if frame is None:
        break

    if i < 0:
        continue

    # On récupère les formes en mouvement
    gray_frame, difference_frame, threshold_frame, background = get_frames(frame, background)

    # On détermine leurs contours
    contours = cv2.findContours(threshold_frame.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]

    if contours is not None:
        if len(contours) > 0:
            # On récupère la plus grande forme, et si elle est assez grande, on dessine son contour, on détermine son
            # centre et on calcule sa trajectoire
            largest_contour = get_largest_contour(contours)

            if cv2.contourArea(largest_contour) > MIN_CONTOUR_AREA:
                (x, y, w, h) = cv2.boundingRect(largest_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                (x1, y1, _) = calculate_3d_coords(x, y)
                (x2, y2, _) = calculate_3d_coords(x, y + h)
                (x3, y3, _) = calculate_3d_coords(x + w, y + h)
                (x4, y4, _) = calculate_3d_coords(x + w, y)

                # TODO change me
                trajectory_camera_coord.append((x + w / 2, y + w / 2))

                corners_trajectories[0].append((x1, y1))
                corners_trajectories[1].append((x2, y2))
                corners_trajectories[2].append((x3, y3))
                corners_trajectories[3].append((x4, y4))
            else:
                n = (1e17, 1e17)
                # TODO change me
                trajectory_camera_coord.append(n)
                corners_trajectories[0].append(n)
                corners_trajectories[1].append(n)
                corners_trajectories[2].append(n)
                corners_trajectories[3].append(n)
        else:
            n = (1e17, 1e17)
            # TODO change me
            trajectory_camera_coord.append(n)
            corners_trajectories[0].append(n)
            corners_trajectories[1].append(n)
            corners_trajectories[2].append(n)
            corners_trajectories[3].append(n)


    else:
        n = (1e17, 1e17)
        # TODO change me
        trajectory_camera_coord.append(n)
        corners_trajectories[0].append(n)
        corners_trajectories[1].append(n)
        corners_trajectories[2].append(n)
        corners_trajectories[3].append(n)

    video_save.write(frame)
    cv2.namedWindow("Color Frame", cv2.WINDOW_NORMAL)
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    if i % 30 == 0:
        print(f"{int(i / LAST_FRAME_INDEX * 100)}% done")

video.release()
video_save.release()
cv2.destroyAllWindows()

# On lisse la trajectoire
corners_trajectories[0] = trajectory_filtering(corners_trajectories[0])
corners_trajectories[1] = trajectory_filtering(corners_trajectories[1])
corners_trajectories[2] = trajectory_filtering(corners_trajectories[2])
corners_trajectories[3] = trajectory_filtering(corners_trajectories[3])

# TODO improve me
trajectory = corners_trajectories[0]
trajectory = trajectory_filtering(trajectory)


size = len(trajectory)
time = np.linspace(0, size / VIDEO_REFRESH_RATE, size)
# On déduit la vitesse de la forme en mouvement à partir de sa trajectoire lissée
velocity = [np.linalg.norm(np.asarray(trajectory[i]) - np.asarray(trajectory[i - 1])) / VIDEO_REFRESH_RATE for i in
            range(1, size)]

# On représente les données obtenues
plot.subplot(2, 2, 1)
plot.title("Profil de position")
plot.xlabel("Temps (s)")
plot.ylabel("Position (m)")
lines = plot.plot(trajectory)
plot.legend(lines, ["Position suivant x", "Position suivant y"])

plot.subplot(2, 2, 2)
plot.title("Profil de vitesse")
plot.xlabel("Temps (s)")
plot.ylabel("Vitesse (m/s)")
plot.plot(time[:-1], velocity)

plot.subplot(2, 2, 3)
plot.title("Deplacement sur le plan de la piste")
plot.xlabel("Y")
plot.ylabel("X")
plot.plot(np.transpose(trajectory)[0], np.transpose(trajectory)[1])

plot.subplot(2, 2, 4)
plot.title("Test")
plot.xlabel("Y")
plot.ylabel("X")
plot.plot(np.transpose(corners_trajectories[3])[0])

#
# plot.subplot(2, 2, 1)
# plot.plot( np.transpose(corners_trajectories[0])[0])
#
# plot.subplot(2, 2, 2)
# plot.plot( np.transpose(corners_trajectories[3])[0])
#
# plot.subplot(2, 2, 3)
# plot.plot( np.transpose(corners_trajectories[2])[0])
#
# plot.subplot(2, 2, 4)
# plot.plot( np.transpose(corners_trajectories[1])[0])


# on enregistre
print(trajectory_camera_coord)
np.savetxt('matrices/points/positions/stereo_1_'+droiteougauche, trajectory_camera_coord)
# attention la courbe n'est pas filtrée

plot.show()

