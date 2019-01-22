import numpy as np
import math as m

frameperseconde = 1
periode = 1 / frameperseconde

# points3D=np.loadtxt('matrices/points/points3D/stereo_1_points_3d.txt')
points3D = [(1, 2, 3), (4, 5, 8), (7, 8, 9), (1e17,1e17,1e17),(18, 17, 16), (18, 17, 16), (1, 1, 1)]

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
    if(m.fsum(points3D[i])<1e17)and(m.fsum(points3D[i])<1e17):  #on ne prend que les pts consécutif
        vitesseX.append((points3D[i + 1][0] - points3D[i][0]) / periode)
        vitesseY.append((points3D[i + 1][1] - points3D[i][1]) / periode)
        vitesseZ.append((points3D[i + 1][2] - points3D[i][2]) / periode)
        indice_vitesse.append(i)

norme_vitesse_XY = [m.sqrt(vitesseX[i] ** 2 + vitesseY[i] ** 2) for i in range(len(vitesseY))]
