import cv2 as cv
import numpy as np
import scipy.signal
import math
from matplotlib import pyplot
from performance400 import extrinsic_calibration

DETECTION_THRESHOLD = 10
MIN_CONTOUR_AREA = 50
GAUSSIAN_BLUR = 25
NUMBER_OF_DILATATION = 2


def get_trajectory(left_video, right_video, left_lower_bound=(0, 0), left_upper_bound=(3840, 2160),
                   right_lower_bound=(0, 0), right_upper_bound=(3840, 2160)):
    """
    Transforms the two trajectories in the camera coords
    into one trajectory in the runway coords
    :param left_video:
    :param right_video:
    :param left_lower_bound:
    :param left_upper_bound:
    :param right_lower_bound:
    :param right_upper_bound:
    """
    left_camera_trajectory = get_camera_trajectory(left_video, left_lower_bound, left_upper_bound)
    right_camera_trajectory = get_camera_trajectory(right_video, right_lower_bound, right_upper_bound)
    return extrinsic_calibration.get_3d_coords(left_camera_trajectory, right_camera_trajectory)


def get_camera_trajectory(video, lower_bound, upper_bound):
    """
    Get the trajectory of the runner in the camera coords
    :param video:
    """
    background = None
    corners_trajectories = [[], [], [], []]  # Top left hand corner then CCW

    while True:
        # On s'assure que la frame courante est bonne et nous intéresse
        check, frame = video.read()
        if not check or frame is None:
            break

        # On récupère les formes en mouvement
        gray_frame, difference_frame, threshold_frame, background = get_frames(frame, background)

        # On détermine leurs contours
        contours = cv.findContours(threshold_frame.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[1]
        remove_out_of_bounds_contours(contours, lower_bound, upper_bound)

        test = True
        if contours is not None:
            if len(contours) > 0:
                # On récupère la plus grande forme, et si elle est assez grande, on dessine son contour,
                # on détermine son centre et on calcule sa trajectoire

                largest_contour = get_largest_contour(contours)

                if cv.contourArea(largest_contour) > MIN_CONTOUR_AREA:
                    x, y, w, h = cv.boundingRect(largest_contour)
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                    x1, y1 = x, y
                    x2, y2 = x, y + h
                    x3, y3 = x + w, y + h
                    x4, y4 = x + w, y

                    corners_trajectories[0].append((x1, y1))
                    corners_trajectories[1].append((x2, y2))
                    corners_trajectories[2].append((x3, y3))
                    corners_trajectories[3].append((x4, y4))

                    test = False

        if test:
            n = (1e17, 1e17)
            corners_trajectories[0].append(n)
            corners_trajectories[1].append(n)
            corners_trajectories[2].append(n)
            corners_trajectories[3].append(n)

        cv.namedWindow("Color Frame", cv.WINDOW_NORMAL)
        cv.imshow("Color Frame", frame)

        key = cv.waitKey(1)

        if key == ord('q'):
            break

    video.release()
    cv.destroyAllWindows()

    # FIXME
    trajectory = corners_trajectories[0]

    return trajectory


def remove_out_of_bounds_contours(contours, lower_bound, upper_bound):
    """
    Used to rempve every contour outside of the box lower_bound -> upper_bound
    :param contours:
    :param lower_bound:
    :param upper_bound:
    """
    x0, y0 = lower_bound
    w0 = upper_bound[0] - x0
    h0 = upper_bound[1] - y0
    removed = 0
    for i in range(len(contours)):
        contour = contours[i - removed]
        x, y, w, h = cv.boundingRect(contour)
        if x < x0 or y < y0 or x + w > x0 + w0 or y + h > y0 + h0:
            del contours[i - removed]
            removed += 1


def draw_trajectory(background, trajectory, extrinsic_parameters):
    """
    Draws the trajectory directly on the image background
    :param background:
    :param trajectory:
    :param extrinsic_parameters:
    """

    removed = 0
    for i in range(len(trajectory)):
        vec = trajectory[i - removed]
        if vec[0] > 1e+16:
            trajectory = np.delete(trajectory, i - removed, axis=0)
            removed += 1

    extrinsic_camera_matrix, extrinsic_distortion_vector, extrinsic_rotation_vector, \
    extrinsic_translation_vector = extrinsic_parameters
    trajectory = np.array(trajectory, 'float32')
    image_points, _ = cv.projectPoints(trajectory, extrinsic_rotation_vector, extrinsic_translation_vector,
                                       extrinsic_camera_matrix, extrinsic_distortion_vector)

    size = background.shape[:2]

    for j in range(len(image_points)):
        if 0 < image_points[j][0][0] < size[1] and 0 < image_points[j][0][1] < size[0]:
            cv.circle(background, (math.floor(image_points[j][0][0]), math.floor(image_points[j][0][1])),
                      3, (0, 0, 255), 20)


def get_frames(frame, background):
    """
    Extract the moving objects from the image frame relative to the image background
    :param frame:
    :param background:
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (GAUSSIAN_BLUR, GAUSSIAN_BLUR), 0)

    if background is None:
        background = gray

    difference = cv.absdiff(background, gray)
    threshold = cv.threshold(difference, DETECTION_THRESHOLD, 255, cv.THRESH_BINARY)[1]

    lower_bound = (0, 0, 40)
    upper_bound = (130, 130, 150)
    sky_threshold = cv.inRange(frame, lower_bound, upper_bound)
    sky_threshold = cv.dilate(sky_threshold, None, iterations=2)

    threshold &= sky_threshold
    threshold = cv.dilate(threshold, None, iterations=NUMBER_OF_DILATATION)

    return gray, difference, threshold, background


def get_largest_contour(contours):
    """
    Returns the largest contour from contours
    :param contours:
    """
    largest_contour = contours[0]

    for contour in contours:
        if cv.contourArea(contour) > cv.contourArea(largest_contour):
            largest_contour = contour

    return largest_contour


def trajectory_filtering(trajectory):
    """
    Filters the trajectory
    :param trajectory:
    """
    shaped_trajectory = np.transpose(np.asarray(trajectory))
    filtered_x = scipy.signal.savgol_filter(shaped_trajectory[0], 21, 5)
    filtered_y = scipy.signal.savgol_filter(shaped_trajectory[1], 21, 5)

    filtered_trajectory = [(filtered_x[k], filtered_y[k]) for k in range(0, len(filtered_x))]

    return filtered_trajectory
