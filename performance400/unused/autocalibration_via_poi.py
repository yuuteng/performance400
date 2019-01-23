# Autocalibration via points of interest
import cv2 as cv
import numpy as np


# pois in an array of tuples (x, y)
def autocalibration_via_poi(image, known_coordinates, pois, sensitivity):
    orb = cv.ORB_create(nfeatures=10, scoreType=cv.ORB_HARRIS_SCORE)
    res = (known_coordinates.copy(), [])
    removed = 0
    for j in range(len(pois)):
        poi = pois[j]
        x, y = poi
        sub_image = image[y - sensitivity: y + sensitivity, x - sensitivity: x + sensitivity]

        gray = cv.cvtColor(sub_image, cv.COLOR_BGR2GRAY)
        # gray = cv.blur(gray, (4, 4))
        # gray = cv.threshold(gray, 170, 255, cv.THRESH_TOZERO)[1]
        # gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, -5)

        kp = orb.detect(gray, None)
        kp, des = orb.compute(gray, kp)

        # thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]

        cv.namedWindow("thresh", cv.WINDOW_NORMAL)
        cv.imshow("thresh", gray)

        current = 0
        cv.namedWindow("res", cv.WINDOW_NORMAL)
        print(known_coordinates[j])
        if len(kp) > 0:
            test = False
            while True:
                draw_keypoints(sub_image, kp, current)
                cv.imshow("res", sub_image)
                key = cv.waitKey(0)
                if key == ord('q'):
                    test = True
                    break
                elif key == 13:  # enter
                    res[1].append(
                        (int(x - sensitivity + kp[current].pt[0]), int(y - sensitivity + kp[current].pt[1])))
                    break
                elif key == ord('s'):
                    res[0].pop(j - removed)
                    removed += 1
                    break
                elif key == 81:  # left
                    current -= 1
                    current %= len(kp)
                elif key == 83:  # right
                    current += 1
                    current %= len(kp)

            if test:
                break
        else:
            res[0].pop(j - removed)
            removed += 1
            cv.imshow("res", sub_image)
            if cv.waitKey(0) == ord('q'):
                break

    cv.destroyAllWindows()

    return res


def draw_keypoints(m_image, m_keypoints, m_current):
    for k in range(len(m_keypoints) - 1, -1, -1):
        m_keypoint = m_keypoints[k]
        m_color = (0, 0, 255)
        if k == m_current:
            m_color = (0, 255, 0)
        cv.circle(m_image, (int(m_keypoint.pt[0]), int(m_keypoint.pt[1])), 5, m_color, thickness=1)

    return


def draw_circle(m_event, m_x, m_y, m_flags, m_param):
    if m_event == cv.EVENT_LBUTTONUP:
        m_image = m_param[0]
        m_param[1].append((m_x, m_y))
        cv.circle(m_image, (m_x, m_y), 9, (255, 0, 255), 4)

    return


def mark_pois(image):
    cv.namedWindow("image", cv.WINDOW_NORMAL)
    res = []
    cv.setMouseCallback("image", draw_circle, param=[image, res])

    while True:
        cv.imshow("image", image)
        key = cv.waitKey(1)
        if key == ord('q') or key == ord('k') or key == 13:
            break

    cv.destroyAllWindows()

    return res


video = cv.VideoCapture("videos/runway/right_run.mkv")
img = video.read()[1]
video.release()
n_pois = mark_pois(img.copy())
dx = 1.815
dy = 1.22
kc = []
for i in range(12):
    kc.append((int(i / 4) * dx + 3, i % 4 * dy + dy, 0))
known, kp = autocalibration_via_poi(img, kc, n_pois, 40)
np.savetxt("matrices/points/calibration_object_points/stereo_2_droite_obj_points", np.array(known))
np.savetxt("matrices/points/calibration_image_points/stereo_2_droite_img_points", np.array(kp))
# cv.namedWindow("img",cv.WINDOW_NORMAL)
# cv.imshow("img",cv.WINDOW_NORMAL)
# print(cv.waitKey(0))
