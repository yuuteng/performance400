
import cv2
import numpy as np
import matplotlib.pyplot as plot

from performance400.calculate_3d_coords import calculate_3d_coords

video = cv2.VideoCapture('videos/V0run.MOV')
FIRST_FRAME_INDEX = -150
LAST_FRAME_INDEX = 650


def get_frames(m_frame, m_background):
    m_gray_frame = cv2.cvtColor(m_frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (11, 11), 0)

    if m_background is None:
        m_background = m_gray_frame

    m_difference_frame = cv2.absdiff(m_background, m_gray_frame)
    m_threshold_frame = cv2.threshold(m_difference_frame, 30, 255, cv2.THRESH_BINARY)[1]
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
    dr = 400
    x1, y1 = x + w - dr, y
    x2, y2 = x1 + dr, y1 + dr
    cv2.rectangle(m_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return (x1 + x2) / 2, (y1 + y2) / 2, m_frame


def draw_trajectory(m_trajectory, m_frame, color=(0, 255, 0)):
    cv2.polylines(m_frame, [np.array(m_trajectory, 'int32').reshape((-1, 1, 2))], isClosed=False, color=color,
                  thickness=10, lineType=cv2.LINE_AA)

    return m_frame


def lissagetrajectoire(m_trajectory, dist, coeff):
    liste_diff2 = [0]
    for j in range(1, len(m_trajectory)):
        liste_diff2.append(
            (m_trajectory[j][0] - m_trajectory[j - 1][0]) ** 2 + (m_trajectory[j][1] - m_trajectory[j - 1][1]) ** 2)
    m_trajectory_corrected = []
    for j in range(dist, len(m_trajectory) - dist):
        if (m_trajectory[j][0] - m_trajectory[j - 1][0]) ** 2 + (
                m_trajectory[j][1] - m_trajectory[j - 1][1]) ** 2 < coeff * np.mean(liste_diff2[j - dist:j + dist]):
            m_trajectory_corrected.append(m_trajectory[j])
    return m_trajectory_corrected


def signal_remplissage_moyenne(m_trajectory, p):
    for k in range(len(m_trajectory)):
        if (m_trajectory[k] == (None, None)):
            Nb = 0
            for l in range(k - p, k + p):
                if (m_trajectory[l][1] != None):
                    m_trajectory[k] += m_trajectory[l]
                    Nb += 1
                m_trajectory[k] = m_trajectory[k][0] / Nb
                m_trajectory[k] = m_trajectory[k][1] / Nb

    return m_trajectory




background = None
trajectory = []

for i in range(FIRST_FRAME_INDEX, LAST_FRAME_INDEX):
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

            if cv2.contourArea(largest_contour) > 50000:
                xc, yc, frame = draw_position(largest_contour, frame)
                xcc, ycc = calculate_3d_coords(xc, yc)[0:2]
                trajectory.append((xcc, ycc))
                # frame = draw_trajectory(trajectory, frame)
    else:
        trajectory.append((None, None))
        print("zvnbzeiurfnhoairehnvoziherhrhvaeuog")

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

trajectory = lissagetrajectoire(trajectory, 10, 3)

velocity = [np.linalg.norm(np.asarray(trajectory[i]) - np.asarray(trajectory[i - 1])) for i in
            range(1, len(trajectory))]

plot.title("Profil de vitesse")
plot.xlabel("Temps")
plot.ylabel("Vitesse")
plot.plot(velocity)
plot.show()
