import cv2
import numpy as np
import matplotlib.pyplot as plot
import scipy.signal

from performance400.calculate_3d_coords import calculate_3d_coords


# Détermine les formes qui ont changé par rapport à background
def get_frames(m_frame, m_background):
    m_gray_frame = cv2.cvtColor(m_frame, cv2.COLOR_BGR2GRAY)
    m_gray_frame = cv2.GaussianBlur(m_gray_frame, (GAUSSIAN_BLUR, GAUSSIAN_BLUR), 0)

    if m_background is None:
        m_background = m_gray_frame

    m_difference_frame = cv2.absdiff(m_background, m_gray_frame)
    m_threshold_frame = cv2.threshold(m_difference_frame, DETECTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    m_threshold_frame = cv2.dilate(m_threshold_frame, None, iterations=NUMBER_OF_DILATATION)

    return m_gray_frame, m_difference_frame, m_threshold_frame, m_background


# Renvoie le plus grand contours de la collection m_contours
def get_largest_contour(m_contours):
    m_largest_contour = contours[0]

    for m_contour in m_contours:
        if cv2.contourArea(m_contour) > cv2.contourArea(m_largest_contour):
            m_largest_contour = m_contour

    return m_largest_contour


# Renvoie le centre du contour m_largest_contour et le dessine sur m_frame
def draw_position(m_largest_contour, m_frame):
    (x, y, w, h) = cv2.boundingRect(m_largest_contour)
    x1, y1 = x, y
    x2, y2 = x1 + w, y1 + h
    cv2.rectangle(m_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return (x1 + x2) / 2, (y1 + y2) / 2, m_frame


# Dessine la trajectoire du centre du plus grand contour sur m_frame
def draw_trajectory(m_trajectory, m_frame, color=(0, 255, 0)):
    cv2.polylines(m_frame, [np.array(m_trajectory, 'int32').reshape((-1, 1, 2))], isClosed=False, color=color,
                  thickness=10, lineType=cv2.LINE_AA)

    return m_frame


# Lisse la trajectoire m_trajectory
def trajectory_smoothing(m_trajectory, m_window_length, m_threshold):
    m_consecutive_squared_distances = [0]
    for j in range(1, len(m_trajectory)):
        m_consecutive_squared_distances.append(
            (m_trajectory[j][0] - m_trajectory[j - 1][0]) ** 2 + (m_trajectory[j][1] - m_trajectory[j - 1][1]) ** 2)
    m_corrected_trajectory = []
    for j in range(m_window_length, len(m_trajectory) - m_window_length):
        if (m_trajectory[j][0] - m_trajectory[j - 1][0]) ** 2 + (
                m_trajectory[j][1] - m_trajectory[j - 1][1]) ** 2 < m_threshold * np.mean(
                m_consecutive_squared_distances[j - m_window_length:j + m_window_length]):
            m_corrected_trajectory.append(m_trajectory[j])

    m_corrected_trajectory = trajectory_filtering(m_corrected_trajectory)

    return m_corrected_trajectory


# Filtre la trajectoire m_trajectory
def trajectory_filtering(m_trajectory):
    m_shaped_trajectory = np.transpose(np.asarray(m_trajectory))
    m_filtered_x = scipy.signal.savgol_filter(m_shaped_trajectory[0], 21, 5)
    m_filtered_y = scipy.signal.savgol_filter(m_shaped_trajectory[1], 21, 5)

    m_filtered_trajectory = [(m_filtered_x[k], m_filtered_y[k]) for k in range(0, len(m_filtered_x))]

    return m_filtered_trajectory


FIRST_FRAME_INDEX = 0
LAST_FRAME_INDEX = 210
VIDEO_FREQUENCY = 30
DETECTION_THRESHOLD = 10
MIN_CONTOUR_AREA = 2000
GAUSSIAN_BLUR = 21
NUMBER_OF_DILATATION = 2

video = cv2.VideoCapture('videos/runway/gauche.mp4')

background = None
trajectory = []
trajectory_camera_coord=[]

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
                xc, yc, frame = draw_position(largest_contour, frame)
                xcc, ycc = calculate_3d_coords(xc, yc)[:2]
                trajectory_camera_coord.append((xc, yc))
                trajectory.append((xcc[0], ycc[0]))
                # frame = draw_trajectory(trajectory, frame)
    else:
        trajectory.append((None, None))

    # cv2.imshow("Diff Frame", diff_frame)
    # cv2.imshow("Threshold Frame", thresh_frame)
    cv2.namedWindow("Color Frame", cv2.WINDOW_NORMAL)
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    if np.random.rand() < 0.05:
        print(f"{int(i / LAST_FRAME_INDEX * 100)}% done")

video.release()
cv2.destroyAllWindows()

# On lisse la trajectoire
trajectory = trajectory_smoothing(trajectory, 10, 3)
time = np.linspace(0, len(trajectory) / VIDEO_FREQUENCY, len(trajectory))
# On déduit la vitesse de la forme en mouvement à partir de sa trajectoire lissée
velocity = [np.linalg.norm(np.asarray(trajectory[i]) - np.asarray(trajectory[i - 1])) * VIDEO_FREQUENCY for i in
            range(1, len(trajectory))]

# On représente les données obtenues
plot.subplot(2, 1, 1)
plot.title("Profil de position")
plot.xlabel("Temps (s)")
plot.ylabel("Position (m)")
lines = plot.plot(time, trajectory)
plot.legend([lines[0], lines[1]], ["Position suivant x", "Position suivant y"])

plot.subplot(2, 1, 2)
plot.title("Profil de vitesse")
plot.xlabel("Temps (s)")
plot.ylabel("Vitesse (m/s)")
plot.plot(time[:-1], velocity)
plot.show()

#on enregistre
np.savetxt('trajectoirecoorcamera.txt',trajectory_camera_coord)
#attention la courbe n'est pas filtré
