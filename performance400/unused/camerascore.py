import numpy as np
import matplotlib.pyplot as pl
import cv2


#
# fonction score pour chaque video fonction de la frame courante et des scores précédants
# param ->[frame1,frame2,frame3,frame4] [frame1,frame2,frame3,frame4],[score1,score2, score3,score4]
# avec framei=(isthereanyobject, taille,coord)
def score(ancienneframes, newframes, scores):
    a_scores=[0,0,0,0]
    coeff_persistance = 0
    coeff_diff = 1
    coeff_taille = 1
    k = 1
    for ind in range(4):
        if  (newframes[ind][0]and ancienneframes[ind][0]):
            print(ancienneframes[ind][2])
            a_scores[ind] = coeff_persistance * scores[ind] + coeff_taille * newframes[ind][1] - k * abs(
                (coeff_diff * (np.linalg.norm(ancienneframes[ind][2] - newframes[ind][2]))))
        else:
            a_scores[ind]=0
    return a_scores


cam1=np.load('videos/Test4Cam/frames_cam_1.npy')
cam2=np.load('videos/Test4Cam/frames_cam_2.npy')
cam3=np.load('videos/Test4Cam/frames_cam_3.npy')
cam4=np.load('videos/Test4Cam/frames_cam_4.npy')
anciennesframes1=cam1[:150]
newframes1=cam1[1:151]
anciennesframes2=cam2[:150]
newframes2=cam2[1:151]
anciennesframes3=cam3[:150]
newframes3=cam3[1:151]
anciennesframes4=cam4[:150]
newframes4=cam4[1:151]
# anciennesframes=np.transpose(anciennesframes)
# newframe=np.transpose(newframe)

real_score=[[1,1,1,1]]

for i in range(150):
    print(real_score)
    real_score.append(score([anciennesframes1[i],anciennesframes2[i],anciennesframes3[i],anciennesframes4[i]],
                            [newframes1[i],newframes2[i],newframes3[i],newframes4[i]],real_score[i]))



plot1=pl.plot(real_score,)
pl.legend(plot1,['cam1','cam2','cam3','cam4'])
pl.show()