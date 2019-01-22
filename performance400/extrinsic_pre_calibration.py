import numpy as np


def calibrate(left_background, right_background, left_object_points, right_object_points):
    pass  # left_interest_points, right_interest_points


def get_interest_points():
    return np.loadtxt("matrices/points/object_points/left"), np.loadtxt("matrices/points/object_points/right")
