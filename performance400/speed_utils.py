import math as m
import time

import matplotlib.pyplot as pl
import numpy as np
import scipy.signal


def get_speed_profiles(trajectory, refresh_rate, windows_size_for_average=6, savgol1=13, savgol2=7, medfilt=11):
    """

    Calculate speed profiles

    :param trajectory: the trajectory as generated by trajectory_utils.py
    :param refresh_rate: the refresh rate associate withe the trajectory
    :param windows_size_for_average:  number of points taken for the sliding windows average
    :param savgol1:  first parameter for scipy.savgol filter
    :param savgol2: second parameter for scipy.savgol filter
    :param medfilt: size of the median filter, must be odd
    :return: the raw speed profile,the filtered speed profile, the median speed profile, the mean speed profile
            and the index needed to interpret all this profiles
    """
    param = int(windows_size_for_average / 2)
    speed_2d, speed_indices = differentiate(trajectory, refresh_rate)
    speed_2d_savgol_filtered = scipy.signal.savgol_filter(speed_2d, savgol1, savgol2)
    speed_2d_median_filtered = scipy.signal.medfilt(speed_2d, medfilt)

    speed_2d_mean = speed_2d[:param] + [np.mean(speed_2d[i - param:i + param]) for i in
                                        range(param, len(speed_2d) - param)] + speed_2d[len(speed_2d) - param:]

    return speed_2d, speed_2d_mean, speed_2d_median_filtered, speed_2d_savgol_filtered, speed_indices


def get_speed_raw_profile(trajectory, refresh_rate):
    """
    Generate the raw speed profile

    :param trajectory: the trajectory as generated by trajectory_utils.py
    :param refresh_rate: the refresh rate associate withe the trajectory
    :return: the raw speed profile and the index associate
    """
    return differentiate(trajectory, refresh_rate)

def differentiate(trajectory, refresh_rate):
    """
    Returns a euler approximation of the differentiation
    FIXME: whether used by calculating with each components or using linalg, the results are slightly different
    :param trajectory:
    :param refresh_rate:
    """
    speed_profile_x = []
    speed_profile_y = []
    speed_profile_z = []
    speed_indices = []

    for i in range(len(trajectory) - 1):
        if (m.fsum(trajectory[i]) < 1e17) and (m.fsum(trajectory[i + 1]) < 1e17):  # only take true consecutive points
            speed_x = (trajectory[i + 1][0] - trajectory[i][0]) * refresh_rate
            speed_y = (trajectory[i + 1][0] - trajectory[i][0]) * refresh_rate
            speed_z = (trajectory[i + 1][0] - trajectory[i][0]) * refresh_rate
            if is_abnormal_value([speed_x, speed_y, speed_z]):
                continue
            speed_profile_x.append(speed_x)
            speed_profile_y.append(speed_y)
            speed_profile_z.append(speed_z)
            speed_indices.append(i)

    speed_2d = [m.sqrt(speed_profile_x[i] ** 2 + speed_profile_y[i] ** 2) for i in range(len(speed_profile_y))]

    return speed_2d, speed_indices


def is_abnormal_value(speeds):
    """
    Cheat method to quickly remove abnormal values
    Better not to use in the final version
    :param speeds:
    """
    return abs(speeds[0]) > 10 or abs(speeds[1]) > 10 or abs(speeds[2]) > 10


def export_speed_profiles(trajectory, refresh_rate):
    """
    Export speed profile to png using get_speed_profiles function

    :param trajectory: the trajectory as generated by trajectory_utils.py
    :param refresh_rate: the refresh rate associate withe the trajectory
    :return: None

    Note : the exports goes to the path export/profile_name_h_min_s_day_month_year.png
    """
    speed_2d, speed_2d_mean, speed_2d_median_filtered, speed_2d_savgol_filtered, index_speed = get_speed_profiles(
        trajectory, refresh_rate)
    pl.figure()
    pl.title('Profil de vitesse brut')
    pl.ylabel('Vitesse en m/s')
    pl.xlabel('Num de frame')
    pl.plot(index_speed, speed_2d)
    pl.savefig(
        'export/Profil_brut_' + str(time.localtime().tm_hour) + 'h_' + str(time.localtime().tm_min) + 'min_' + str(
            time.localtime().tm_sec) + 'sec_' +
        str(time.localtime().tm_mday) + '-' + str(time.localtime().tm_mon) + '-' + str(
            time.localtime().tm_year) + '.png')
    pl.close()
    pl.figure()
    pl.title('Profil de vitesse median')
    pl.ylabel('Vitesse en m/s')
    pl.xlabel('Num de frame')
    pl.plot(index_speed, speed_2d_median_filtered)
    pl.savefig(
        'export/Profil_median_' + str(time.localtime().tm_hour) + 'h_' + str(time.localtime().tm_min) + 'min_' + str(
            time.localtime().tm_sec) + 'sec_' +
        str(time.localtime().tm_mday) + '-' + str(time.localtime().tm_mon) + '-' + str(
            time.localtime().tm_year) + '.png')
    pl.close()
    pl.figure()
    pl.title('Profil de vitesse filtre')
    pl.plot(index_speed, speed_2d_savgol_filtered)
    pl.savefig(
        'export/Profil_filtreSV_' + str(time.localtime().tm_hour) + 'h_' + str(time.localtime().tm_min) + 'min_' + str(
            time.localtime().tm_sec) + 'sec_' +
        str(time.localtime().tm_mday) + '-' + str(time.localtime().tm_mon) + '-' + str(
            time.localtime().tm_year) + '.png')
    pl.close()
    pl.figure()
    pl.title('profil de vitesse moyenne')
    pl.ylabel('Vitesse en m/s')
    pl.xlabel('Num de frame')
    pl.plot(index_speed, speed_2d_mean)
    pl.savefig(
        'export/Profil_moyen_' + str(time.localtime().tm_hour) + 'h_' + str(time.localtime().tm_min) + 'min_' + str(
            time.localtime().tm_sec) + 'sec_' +
        str(time.localtime().tm_mday) + '-' + str(time.localtime().tm_mon) + '-' + str(
            time.localtime().tm_year) + '.png')
    pl.close()
