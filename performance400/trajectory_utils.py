import cv2 as cv
import numpy as np
import scipy.signal
import math
from performance400 import extrinsic_calibration

DETECTION_THRESHOLD = 10
MIN_CONTOUR_AREA = 2000
GAUSSIAN_BLUR = 25
NUMBER_OF_DILATATION = 2


def get_trajectory(left_video, right_video):
    left_camera_trajectory = get_camera_trajectory(left_video)
    right_camera_trajectory = get_camera_trajectory(right_video)
    return extrinsic_calibration.get_3d_coords(left_camera_trajectory, right_camera_trajectory)


def get_camera_trajectory(video):
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


def draw_trajectory(background, trajectory, right_camera, extrinsic_parameters):
    extrinsic_camera_matrix, extrinsic_distortion_vector, extrinsic_rotation_vector, \
    extrinsic_translation_vector = extrinsic_parameters

    image_points, _ = cv.projectPoints(trajectory, extrinsic_rotation_vector, extrinsic_translation_vector,
                                       extrinsic_camera_matrix, extrinsic_distortion_vector)

    size = background.shape[:2]

    for j in range(len(image_points)):
        if 0 < image_points[j][0][0] < size[1] and 0 < image_points[j][0][1] < size[0]:
            cv.circle(background, (math.floor(image_points[j][0][0]), math.floor(image_points[j][0][1])),
                      3, (0, 0, 255), 20)


# Détermine les formes qui ont changé par rapport à background
def get_frames(frame, background):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (GAUSSIAN_BLUR, GAUSSIAN_BLUR), 0)

    if background is None:
        background = gray

    difference = cv.absdiff(background, gray)
    threshold = cv.threshold(difference, DETECTION_THRESHOLD, 255, cv.THRESH_BINARY)[1]
    threshold = cv.dilate(threshold, None, iterations=NUMBER_OF_DILATATION)

    return gray, difference, threshold, background


# Renvoie le plus grand contours de la collection m_contours
def get_largest_contour(contours):
    largest_contour = contours[0]

    for contour in contours:
        if cv.contourArea(contour) > cv.contourArea(largest_contour):
            largest_contour = contour

    return largest_contour


# Filtre la trajectoire m_trajectory
def trajectory_filtering(trajectory):
    shaped_trajectory = np.transpose(np.asarray(trajectory))
    filtered_x = scipy.signal.savgol_filter(shaped_trajectory[0], 21, 5)
    filtered_y = scipy.signal.savgol_filter(shaped_trajectory[1], 21, 5)

    filtered_trajectory = [(filtered_x[k], filtered_y[k]) for k in range(0, len(filtered_x))]

    return filtered_trajectory
