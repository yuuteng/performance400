import cv2 as cv
import numpy as np


def draw_circle(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONUP:
        image = param[0]
        param[1].append((x, y))
        cv.circle(image, (x, y), 9, (255, 0, 255), 4)


def calibrate_single(image):
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

    return np.array(image_points)


def calibrate(left_background, right_background, left_object_points, right_object_points):
    np.savetxt("matrices/interest_points/left_images", calibrate_single(left_background))
    np.savetxt("matrices/interest_points/right_images", calibrate_single(right_background))
    np.savetxt("matrices/interest_points/left_objects", left_object_points)
    np.savetxt("matrices/interest_points/right_objects", right_object_points)


def get_interest_points():
    left_interest_points = (
    np.loadtxt("matrices/interest_points/left_images"), np.loadtxt("matrices/interest_points/left_objects"))
    right_interest_points = (
    np.loadtxt("matrices/interest_points/right_images"), np.loadtxt("matrices/interest_points/right_objects"))
    print(left_interest_points)
    print(right_interest_points)
    return left_interest_points, right_interest_points
