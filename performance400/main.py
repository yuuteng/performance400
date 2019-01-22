import glob
import cv2 as cv
import numpy as np
from performance400 import intrinsic_pre_autocalibration, extrinsic_pre_calibration, extrinsic_calibration

PRE_PROCESS = False

left_video = cv.VideoCapture("videos/runway/left_run.mkv")
right_video = cv.VideoCapture("videos/runway/right_run.mkv")
left_check, left_background = left_video.read()
right_check, right_background = right_video.read()

if PRE_PROCESS:
    left_targets = glob.glob("images/targets/left/*.jpg")
    right_targets = glob.glob("images/targets/right/*.jpg")
    intrinsic_pre_autocalibration.autocalibrate(left_targets, right_targets)

    left_object_points = np.load("matrices/points/object_points/left")
    right_object_points = np.load("matrices/points/object_points/right")
    extrinsic_pre_calibration.calibrate(left_background, right_background,
                                        left_object_points,
                                        right_object_points)

interest_points = extrinsic_pre_calibration.get_interest_points()
intrinsic_parameters = intrinsic_pre_autocalibration.get_intrinsic_parameters()
extrinsic_calibration.calibrate(left_background, right_background, interest_points, intrinsic_parameters)
