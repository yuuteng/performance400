import numpy as np
import cv2 as cv
import time
cap = cv.VideoCapture('videos/Course_2_pers/test2pers.MOV')
# take first frame of the video
for i in range(180):
    ret, frame =cap.read()
ret,frame = cap.read()
# setup initial location of window
r,h,c,w = 934,230,1340,320  # simply hardcoded the values
track_window = (c,r,w,h)
# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 100, 10 )
cv.namedWindow('img2',cv.WINDOW_GUI_NORMAL)
while(1):
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # apply meanshift to get the new location
        ret, track_window = cv.CamShift(dst, track_window, term_crit)
        # Draw it on image
        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv.polylines(frame,[pts],True, 255,10)
        cv.rectangle(img2,(934,1340),(934+230,1340+320),color=(255,0,255),thickness=-1)
        cv.imshow('img2',img2)
        #time.sleep(0.1)
        k = cv.waitKey(11) & 0xff
        if k == 27:
            break
        else:
            cv.imwrite(chr(k)+".jpg",img2)
    else:
        break
cv.destroyAllWindows()
cap.release()