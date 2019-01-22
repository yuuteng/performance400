import numpy as np
import math as m
import scipy.signal


def get_speed_profiles(trajectory, refresh_rate, windows_size_for_mean=6, savgol1=13, savgol2=7, medfilt=11):
    param = int(windows_size_for_mean / 2)
    speedX = []
    speedY = []
    speedZ = []
    index_speed = []

    for i in range(len(trajectory) - 1):
        if (m.fsum(trajectory[i]) < 1e17) and (m.fsum(trajectory[i + 1]) < 1e17):  # only take the consecutive points
            speedX.append((trajectory[i + 1][0] - trajectory[i][0]) / refresh_rate)
            speedY.append((trajectory[i + 1][1] - trajectory[i][1]) / refresh_rate)
            speedZ.append((trajectory[i + 1][2] - trajectory[i][2]) / refresh_rate)
            index_speed.append(i)

    norm_speed_XY = [m.sqrt(speedX[i] ** 2 + speedY[i] ** 2) for i in range(len(speedY))]
    norm_speed_XY_SavFil = scipy.signal.savgol_filter(norm_speed_XY, savgol1, savgol2)
    norm_speed_XY_Medfilt = scipy.signal.medfilt(norm_speed_XY, medfilt)

    norm_speed_XY_mean = norm_speed_XY[:param] + [np.mean(norm_speed_XY[i - param:i + param]) for i in
                                                  range(param, len(norm_speed_XY) - param)] + norm_speed_XY[len(
        norm_speed_XY) - param:]

    return norm_speed_XY, norm_speed_XY_mean, norm_speed_XY_Medfilt, norm_speed_XY_SavFil


def get_speed_raw_profile(trajectory, refresh_rate):
    speedX = []
    speedY = []
    speedZ = []
    index_speed = []

    for i in range(len(trajectory) - 1):
        if (m.fsum(trajectory[i]) < 1e17) and (m.fsum(trajectory[i + 1]) < 1e17):  # only take true consecutive points
            speedX.append((trajectory[i + 1][0] - trajectory[i][0]) / refresh_rate)
            speedY.append((trajectory[i + 1][1] - trajectory[i][1]) / refresh_rate)
            speedZ.append((trajectory[i + 1][2] - trajectory[i][2]) / refresh_rate)
            index_speed.append(i)

    norm_speed_XY = [m.sqrt(speedX[i] ** 2 + speedY[i] ** 2) for i in range(len(speedY))]

    return norm_speed_XY
