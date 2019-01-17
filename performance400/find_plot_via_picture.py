import cv2
import numpy as np

template = cv2.imread('images/plot.png', cv2.IMREAD_GRAYSCALE)
width, height = template.shape[::-1]
# img = cv2.imread('images/piste_un_plot.png')
img = cv2.imread('images/piste_deux_plots.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
loc = np.where(result >= 0.7)

for pt in zip(*loc[::-1]):
    print(pt)
    cv2.rectangle(img, pt, (pt[0] + width, pt[1] + height), (0, 255, 0), 1)

# save object
np.savetxt('matrices/points/points_objet/find_objet', loc)

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
