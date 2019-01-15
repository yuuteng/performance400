import cv2
import numpy as np
import matplotlib.pyplot as plot
import scipy.signal

from performance400.calculate_3d_coords import calculate_3d_coords


def get_frames(m_frame, m_background):
    m_gray_frame = cv2.cvtColor(m_frame, cv2.COLOR_BGR2GRAY)
    m_gray_frame = cv2.GaussianBlur(m_gray_frame, (21, 21), 0)

    if m_background is None:
        m_background = m_gray_frame

    m_difference_frame = cv2.absdiff(m_background, m_gray_frame)
    m_threshold_frame = cv2.threshold(m_difference_frame, 10, 255, cv2.THRESH_BINARY)[1]
    m_threshold_frame = cv2.dilate(m_threshold_frame, None, iterations=2)

    return m_gray_frame, m_difference_frame, m_threshold_frame, m_background


def get_largest_contour(m_contours):
    m_largest_contour = contours[0]

    for m_contour in m_contours:
        if cv2.contourArea(m_contour) > cv2.contourArea(m_largest_contour):
            m_largest_contour = m_contour

    return m_largest_contour


def draw_position(m_largest_contour, m_frame):
    (x, y, w, h) = cv2.boundingRect(m_largest_contour)
    x1, y1 = x, y
    x2, y2 = x1 + w, y1 + h
    cv2.rectangle(m_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return (x1 + x2) / 2, (y1 + y2) / 2, m_frame


def draw_trajectory(m_trajectory, m_frame, color=(0, 255, 0)):
    cv2.polylines(m_frame, [np.array(m_trajectory, 'int32').reshape((-1, 1, 2))], isClosed=False, color=color,
                  thickness=10, lineType=cv2.LINE_AA)

    return m_frame


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


def trajectory_filtering(m_trajectory):
    m_shaped_trajectory = np.transpose(np.asarray(m_trajectory))
    m_filtered_x = scipy.signal.savgol_filter(m_shaped_trajectory[0], 21, 5)
    m_filtered_y = scipy.signal.savgol_filter(m_shaped_trajectory[1], 21, 5)

    m_filtered_trajectory = [(m_filtered_x[k], m_filtered_y[k]) for k in range(0, len(m_filtered_x))]

    return m_filtered_trajectory


FIRST_FRAME_INDEX = 0
LAST_FRAME_INDEX = 210

background = None
trajectory = []

video = cv2.VideoCapture('videos/runway/gauche.mp4')
for i in range(-FIRST_FRAME_INDEX, LAST_FRAME_INDEX):
    frame = video.read()[1]
    if frame is None:
        break

    if i < 0:
        continue

    gray_frame, difference_frame, threshold_frame, background = get_frames(frame, background)

    contours = cv2.findContours(threshold_frame.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]

    if contours is not None:
        if len(contours) > 0:
            largest_contour = get_largest_contour(contours)

            if cv2.contourArea(largest_contour) > 2000:
                xc, yc, frame = draw_position(largest_contour, frame)
                xcc, ycc = calculate_3d_coords(xc, yc)[:2]
                # trajectory.append((xcc[0], ycc[0]))
                trajectory.append((xc, yc))
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


trajectory = trajectory_smoothing(trajectory, 10, 3)
time = np.linspace(0, 7 * len(trajectory) / (LAST_FRAME_INDEX - FIRST_FRAME_INDEX), len(trajectory))
velocity = [np.linalg.norm(np.asarray(trajectory[i]) - np.asarray(trajectory[i - 1])) for i in
            range(1, len(trajectory))]

plot.subplot(2, 1, 1)
plot.title("Profil de position")
plot.xlabel("Temps (s)")
plot.ylabel("Position")
lines = plot.plot(time, trajectory)
plot.legend([lines[0], lines[1]], ["Position suivant x", "Position suivant y"])

plot.subplot(2, 1, 2)
plot.title("Profil de vitesse")
plot.xlabel("Temps (s)")
plot.ylabel("Vitesse")
plot.plot(time[:-1], velocity)
plot.show()
