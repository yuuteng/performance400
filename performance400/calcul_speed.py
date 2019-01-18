import numpy as np
import math as m
import matplotlib.pyplot as plt

VIDEO_REFRESH_RATE = 30
points_3d = np.loadtxt('matrices/points/points3D/stereo_1_points_3d_filtres')
velocity = [np.linalg.norm(np.asarray(points_3d[i-1, :1]) - np.asarray(points_3d[i+1, :1])) * VIDEO_REFRESH_RATE/2
            for i in range(1, len(points_3d)-1)]
d=0
D = [d]
for i in range(0, len(points_3d)-1):
    d += np.linalg.norm(np.asarray(points_3d[i, :1]) - np.asarray(points_3d[i+1, :1]))
    D.append(d)



plt.plot(D[1: len(D) - 1], velocity)
plt.show()
