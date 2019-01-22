import cv2 as cv
import numpy as np


def draw_circle(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONUP:
        image = param[0]
        param[1].append((x, y))
        cv.circle(image, (x, y), 9, (255, 0, 255), 4)

    return


def calibrate(image):
    name = "Pre-calibration extrinsèque"
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    image_points = []
    cv.setMouseCallback(name, draw_circle, param=[image, image_points])

    while True:
        cv.imshow(name, image)
        key = cv.waitKey(1)
        if key == ord('q') or key == ord('k') or key == 13:
            break

    cv.destroyAllWindows()

    return image_points


def calibrate(left_background, right_background, left_object_points, right_object_points):
    left_interest_points = [calibrate(left_background), left_object_points]
    right_interest_points = [calibrate(right_background), right_object_points]
    np.save("matrices/interest_points/left", left_interest_points)
    np.save("matrices/interest_points/right", right_interest_points)
    return


def get_interest_points():
    return np.loadtxt("matrices/object_points/left"), np.loadtxt("matrices/object_points/right")
