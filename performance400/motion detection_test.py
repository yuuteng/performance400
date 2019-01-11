import cv2
import numpy as np
import matplotlib.pyplot as plot

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

def lissagetrajectoire(trajectory, dist, coeff):
    liste_diff2 = [0]
    for i in range(1, len(trajectory)):
        liste_diff2.append(
            (trajectory[i][0] - trajectory[i - 1][0]) ** 2 + (trajectory[i][1] - trajectory[i - 1][1]) ** 2)
    trajectory_corrected = []
    for i in range(dist, len(trajectory) - dist):
        if (trajectory[i][0] - trajectory[i - 1][0]) ** 2 + (
                trajectory[i][1] - trajectory[i - 1][1]) ** 2 < coeff * np.mean(liste_diff2[i - dist:i + dist]):
            trajectory_corrected.append(trajectory[i])
    return trajectory_corrected


background = None
trajectory = []

for j in range(FIRST_FRAME_INDEX, LAST_FRAME_INDEX):
    frame = video.read()[1]
    if frame is None:
        break

    if j < 0:
        continue

    gray_frame, difference_frame, threshold_frame, background = get_frames(frame, background)

    contours = cv2.findContours(threshold_frame.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]

    if contours is not None:
        if len(contours) > 0:
            largest_contour = get_largest_contour(contours)

            if cv2.contourArea(largest_contour) > 50000:
                xc, yc, frame = draw_position(largest_contour, frame)
                trajectory.append((xc, yc))
                # frame = draw_trajectory(trajectory, frame)

    # cv2.imshow("Diff Frame", diff_frame)
    # cv2.imshow("Threshold Frame", thresh_frame)
    # cv2.namedWindow("Color Frame", cv2.WINDOW_NORMAL)
    # cv2.imshow("Color Frame", frame)

    # key = cv2.waitKey(1)

    # if key == ord('q'):
    #     break

    if np.random.rand() < 0.1:
        print(f"{int(j / LAST_FRAME_INDEX * 100)}% done")

video.release()
cv2.destroyAllWindows()

trajectory_cor=lissagetrajectoire(trajectory,50,30)

velocity = [np.linalg.norm(np.asarray(trajectory[i]) - np.asarray(trajectory[i - 1])) for i in
            range(1, len(trajectory))]





plot.figure()
#plot.subplot(5,1,1)
#plot.plot(lissagetrajectoire(trajectory,10,2))
#plot.subplot(5,1,2)
plot.plot(lissagetrajectoire(trajectory,10,3))
#plot.subplot(5,1,3)
#plot.plot(lissagetrajectoire(trajectory,20,30))
#plot.subplot(5,1,4)
#plot.plot(lissagetrajectoire(trajectory,50,1))
#plot.subplot(5,1,5)
#plot.plot(lissagetrajectoire(trajectory,20,1))
plot.show()
