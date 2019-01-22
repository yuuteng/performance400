import numpy as np
import cv2 as cv

# controle la calib (xc,yc) de limage img
# clic gauche redéfinir xc,yc
# press e = effacer
# press r = reset
# press k= ok

global xc, yc

Calibok = False
xc, yc = 258, 258
xcs, ycs = xc, yc
img = np.zeros((512, 512, 3), np.uint8)
imgcopie = img.copy()
cv.circle(imgcopie, (xc, yc), 9, (0, 0, 255), 4)


def draw_calib_point(event, x, y, flags, param):
    global xc, yc
    if event == cv.EVENT_LBUTTONDOWN:
        xc, yc = x, y
        cv.circle(imgcopie, (xc, yc), 9, (255, 0, 255), 4)


cv.namedWindow('image')
cv.setMouseCallback('image', draw_calib_point)
while (1):
    cv.imshow('image', imgcopie)
    k = cv.waitKey(1) & 0xFF
    if k == ord('e'):
        imgcopie = img.copy()
        cv.circle(imgcopie, (xc, yc), 9, (0, 0, 255), 4)
    elif k == ord('k'):
        break
    elif k == ord('r'):
        imgcopie = img.copy()
        xc, yc = xcs, ycs
        cv.circle(imgcopie, (xc, yc), 9, (0, 0, 255), 4)

cv.destroyAllWindows()
