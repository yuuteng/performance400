import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from performance400.find_3D_coords_stereo import get_wrong_points_3d
from performance400.find_3D_coords_stereo import delete_wrong_points
from performance400.find_3D_coords_stereo import points_filtering

VIDEO_REFRESH_RATE = 30
points_3d = np.loadtxt('matrices/points/points3D/stereo_1_points_3d')

ind_none = get_wrong_points_3d(points_3d)
points_3d = delete_wrong_points(points_3d, ind_none)
points_filtering(points_3d)

velocity = [np.linalg.norm(np.asarray(points_3d[i - 1, :1]) - np.asarray(points_3d[i + 1, :1])) * VIDEO_REFRESH_RATE / 2
            for i in range(1, len(points_3d) - 1)]

d = 0
D = [d]
ld = []
for i in range(0, len(points_3d) - 1):
    ld.append(np.linalg.norm(np.asarray(points_3d[i, :1]) - np.asarray(points_3d[i + 1, :1])))
    d += ld[-1]
    D.append(d)


for z in range(len(ind_none)):
    ind_none[z] -= z
set = dict([(k, ind_none.count(k)) for k in set(ind_none)])
sum = 0
print(set)
for j in range(len(velocity)):
    if set.get(j + 1) is not None:
        sum += set.get(j + 1)
    if set.get(j + 2) is not None:
        sum += set.get(j + 2)
    if sum is not None and sum > 0:
        velocity[j] = velocity[j] * 2 / (sum)
    sum = 0


velocity = np.array(velocity, 'float32')
D = np.array(D, 'float32')

velocity = velocity*3.6

velocity = scipy.signal.savgol_filter(velocity, 21, 5)
D = scipy.signal.savgol_filter(D, 21, 5)

print('moyenne vitesses', np.mean(velocity))
print('ecart type velocity', np.std(velocity))
print('moyenne distances', np.mean(ld))
print('ecart type disctances', np.std(ld))

plt.plot(D[1: len(D) - 1], velocity, marker='+')
plt.xlabel("Distance parcourue en m")
plt.ylabel("Vitesse en km/h")
plt.show()
