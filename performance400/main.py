import glob
import cv2 as cv
import numpy as np
from matplotlib import pyplot
from performance400 import intrinsic_pre_autocalibration, extrinsic_pre_calibration, extrinsic_calibration, \
    trajectory_utils, speed_utils

EXTRACT_MIRE = True
INTRINSIC_CALIBRATION = False
PRE_EXTRINSIC_CALIBRATION = False
MAIN_RESOLUTION = False


REFRESH_RATE = 30

if EXTRACT_MIRE:
    left_mire = cv.VideoCapture("videos/mires/left_mire.MP4")
    right_mire = cv.VideoCapture("videos/mires/right_mire2.MP4")
    intrinsic_pre_autocalibration.extract_targets(left_mire, 50, False)
    intrinsic_pre_autocalibration.extract_targets(right_mire, 50, True)

left_video = cv.VideoCapture("videos/runway/left_run.mkv")
right_video = cv.VideoCapture("videos/runway/right_run.mkv")
left_check, left_background = left_video.read()
right_check, right_background = right_video.read()

if INTRINSIC_CALIBRATION:
    left_targets = glob.glob("images/targets/left/*.jpg")
    right_targets = glob.glob("images/targets/right/*.jpg")
    intrinsic_pre_autocalibration.autocalibrate(left_targets, right_targets, 10, 7)

if PRE_EXTRINSIC_CALIBRATION:
    left_object_points = np.loadtxt("matrices/interest_points/object_points/left")
    right_object_points = np.loadtxt("matrices/interest_points/object_points/right")
    extrinsic_pre_calibration.calibrate(left_background, right_background,
                                        left_object_points,
                                        right_object_points)
if MAIN_RESOLUTION:
    left_interest_points, right_interest_points = extrinsic_pre_calibration.get_interest_points()
    intrinsic_parameters = intrinsic_pre_autocalibration.get_intrinsic_parameters()
    extrinsic_calibration.calibrate(left_background, right_background, left_interest_points, right_interest_points,
                                intrinsic_parameters)
    left_extrinsic_parameters = extrinsic_calibration.get_extrinsic_parameters(False)
    right_extrinsic_parameters = extrinsic_calibration.get_extrinsic_parameters(True)

    trajectory = trajectory_utils.get_trajectory(left_video, right_video)
    trajectory_utils.draw_trajectory(left_background, trajectory, left_extrinsic_parameters)
    trajectory_utils.draw_trajectory(right_background, trajectory, right_extrinsic_parameters)
    extrinsic_calibration.draw_axes(left_background, False)
    extrinsic_calibration.draw_axes(right_background, True)


    cv.namedWindow("Trajectoire de gauche", cv.WINDOW_NORMAL)
    cv.namedWindow("Trajectoire de droite", cv.WINDOW_NORMAL)
    cv.imshow("Trajectoire de gauche", left_background)
    cv.imshow("Trajectoire de droite", right_background)

    cv.waitKey(0)
    cv.destroyAllWindows()
    left_video.release()
    right_video.release()

    speed_profile,index_speed = speed_utils.get_speed_raw_profile(trajectory, REFRESH_RATE)
    speed_utils.export_speed_profiles(trajectory,REFRESH_RATE)
    pyplot.title("Profil de vitesse")
    pyplot.xlabel("Distance (m)")
    pyplot.ylabel("Vitesse (m/s)")
    pyplot.plot(index_speed,speed_profile)
    pyplot.show()
