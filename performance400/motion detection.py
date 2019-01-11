import cv2
import time
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


def draw_trajectory(m_trajectory, m_frame,color=(0,255,0)):
    cv2.polylines(m_frame, [np.array(m_trajectory, 'int32').reshape((-1, 1, 2))], isClosed=False, color=color,
                  thickness=10, lineType=cv2.LINE_AA)

    return m_frame


background = None
trajectory = []
trajectory_corrected=[]
k=100
seuil=1.1


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
                trajectory.append((xc, yc))
                if ((len(trajectory) > (k)) & (len(trajectory) < 2 * k)):
                    trajectory_corrected.append(trajectory[len(trajectory) - 1 - k])
                elif (len(trajectory) > 2 * k):
                    cumulerr = 0
                    for i in range(k):
                        cumulerr += (trajectory[len(trajectory) - 1 - i - k][0]-trajectory[len(trajectory)  - i - k][0])**2+(trajectory[len(trajectory) - 1 - i - k][1]-trajectory[len(trajectory)  - i - k][1])**2
                    if ((xc - trajectory[len(trajectory) - 1][0]))**2+((yc - trajectory[len(trajectory) - 1][1])**2) <seuil* cumulerr:
                        print("coucou")
                        trajectory_corrected.append((xc, yc))

                frame = draw_trajectory(trajectory, frame)
                frame= draw_trajectory(trajectory_corrected, frame,(255,0,255))

    # cv2.imshow("Diff Frame", diff_frame)
    # cv2.imshow("Threshold Frame", thresh_frame)
    # cv2.namedWindow("Color Frame", cv2.WINDOW_NORMAL)
    # cv2.imshow("Color Frame", frame)

    # key = cv2.waitKey(1)

    # if key == ord('q'):
    #     break

    if np.random.rand() < 0.1:
        print(f"{int(i / LAST_FRAME_INDEX * 100)}% done")

video.release()
cv2.destroyAllWindows()

velocity = [np.linalg.norm(np.asarray(trajectory[i]) - np.asarray(trajectory[i - 1])) for i in
            range(1, len(trajectory))]

plot.title("Profil de vitesse")
plot.xlabel("Position")
plot.ylabel("Vitesse")
plot.plot(velocity)
plot.show()
