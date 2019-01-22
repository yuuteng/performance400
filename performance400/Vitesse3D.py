import numpy as np
import math as m
import matplotlib.pyplot as pl
import scipy.signal

frameperseconde = 30
periode = 1 / frameperseconde

points3D=np.loadtxt('matrices/points/points3D/stereo_2_points_3d_bruts')
# points3D = [(1, 2, 3), (4, 5, 8), (7, 8, 9), (1e17,1e17,1e17),(18, 17, 16), (18, 17, 16), (1, 1, 1)]

# calcul vitesse si filtrer

# vitesseX = []
# vitesseY = []
# vitesseZ = []
#
# for i in range(len(points3D) - 1):
#     vitesseX.append((points3D[i + 1][0] - points3D[i][0]) / periode)
#     vitesseY.append((points3D[i + 1][1] - points3D[i][1]) / periode)
#     vitesseZ.append((points3D[i + 1][2] - points3D[i][2]) / periode)

# calcul vitesse si partiellement filtré (existe des 1e17 mais pas de pts aberrants)
vitesseX = []
vitesseY = []
vitesseZ = []
indice_vitesse=[]

for i in range(len(points3D) - 1):
    if(m.fsum(points3D[i])<1e17)and(m.fsum(points3D[i+1])<1e17):  #on ne prend que les pts consécutif
        vitesseX.append((points3D[i + 1][0] - points3D[i][0]) / periode)
        vitesseY.append((points3D[i + 1][1] - points3D[i][1]) / periode)
        vitesseZ.append((points3D[i + 1][2] - points3D[i][2]) / periode)
        indice_vitesse.append(i)

norme_vitesse_XY = [m.sqrt(vitesseX[i] ** 2 + vitesseY[i] ** 2) for i in range(len(vitesseY))]
norme_vitesse_XY_SavFil=scipy.signal.savgol_filter(norme_vitesse_XY,13,7)
norme_vitesse_XY_Medfilt=scipy.signal.medfilt(norme_vitesse_XY,11)
param=3
norme_vitesse_XY_mean=norme_vitesse_XY[:param]+[np.mean(norme_vitesse_XY[i-param:i+param]) for i in range(param,len(norme_vitesse_XY)-param)]+norme_vitesse_XY[len(norme_vitesse_XY)-param:]
#pl.plot(indice_vitesse[param:len(norme_vitesse_XY)-param],norme_vitesse_XY_mean[param:len(norme_vitesse_XY)-param])
pl.figure()
pl.plot(indice_vitesse,norme_vitesse_XY_SavFil)
pl.savefig('tesurveiubvibviribzscibdst.png')
pl.show()
