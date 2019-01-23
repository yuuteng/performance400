import cv2 as cv
import numpy as np


def draw_circle(event, x, y, flags, param):
    """

    Draw a circle at every left mouse click  and append the coordinate in param[1]

    """
    if event == cv.EVENT_LBUTTONUP:
        image = param[0]
        param[1].append((x, y))
        cv.circle(image, (x, y), 9, (255, 0, 255), 4)


def calibrate_single(image):
    """
    Manually point Points on an image

    :param image: the image to be calibrate
    :return: array with all the points
    """
    name = "Pre-calibration extrinseque"
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


def calibrate(left_background, right_background):
    """

    Call calibrate single on the right and left background and save the results

    :param left_background:
    :param right_background:
    """
    np.savetxt("matrices/interest_points/image_points/left", calibrate_single(left_background.copy()))
    np.savetxt("matrices/interest_points/image_points/right", calibrate_single(right_background.copy()))


def get_interest_points():
    """
    Load interest points saved by extrinsic_pre_calibration.calibrate

    :return: The list of interest points

    """
    left_interest_points = (
        np.loadtxt("matrices/interest_points/image_points/left"), np.loadtxt("matrices/interest_points/object_points/left"))
    right_interest_points = (
        np.loadtxt("matrices/interest_points/image_points/right"), np.loadtxt("matrices/interest_points/object_points/right"))
    return left_interest_points, right_interest_points
