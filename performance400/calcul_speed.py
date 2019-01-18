import numpy as np
import matplotlib.pyplot as plt
from performance400.find_3D_coords_stereo import trajectory_wrong_points
import scipy

VIDEO_REFRESH_RATE = 30
points_3d = np.loadtxt('matrices/points/points3D/stereo_1_points_3d')
a=np.load('matrices/points/positions/stereo_1_droite_positions.npy')
print(a)
indexinterdit = trajectory_wrong_points(points_3d)
count = 0
indexinterdit = np.sort(indexinterdit)

velocity = [np.linalg.norm(np.asarray(points_3d[i - 1, :1]) - np.asarray(points_3d[i + 1, :1])) * VIDEO_REFRESH_RATE / 2
            for i in range(1, len(points_3d) - 1)]

velocity = scipy.signal.savgol_filter(velocity, 21, 5)

d = 0
D = [d]
ld = []
for i in range(0, len(points_3d) - 1):
    ld.append(np.linalg.norm(np.asarray(points_3d[i, :1]) - np.asarray(points_3d[i + 1, :1])))
    d += ld[-1]
    D.append(d)

D = scipy.signal.savgol_filter(D, 21, 5)

for i in indexinterdit:
    points_3d = np.delete(points_3d, (i - count), axis=0)
    count += 1

print('moyenne vitesses', np.mean(velocity))
print('ecart type velocity', np.std(velocity))
print('moyenne distances', np.mean(ld))
print('ecart type disctances', np.std(ld))



plt.plot(D[1: len(D) - 1], velocity)
plt.show()
