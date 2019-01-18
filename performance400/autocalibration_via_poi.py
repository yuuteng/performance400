# Autocalibration via points of interest
import cv2
import numpy as np


# pois in an array of tuples (x, y)
def autocalibration_via_poi(image, pois, sensitivity):
    orb = cv2.ORB_create()
    for poi in pois:
        x, y = poi
        sub_image = image[y - sensitivity: y + sensitivity, x - sensitivity: x + sensitivity]

        gray = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)

        kp = orb.detect(gray, None)
        kp, des = orb.compute(gray, kp)

        sub_image = cv2.drawKeypoints(sub_image, kp, None, color=(255, 0, 0), flags=0)

        # thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]
        # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -5)

        cv2.namedWindow("res", cv2.WINDOW_NORMAL)
        cv2.imshow("res", sub_image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread("images/first_frame_left.png")
autocalibration_via_poi(img,
                        [(144, 2137), (502, 2091), (631, 2073), (1040, 2021), (1194, 1999), (1657, 1941), (1827, 1917),
                         (2366, 1849), (2569, 1821), (464, 2009), (905, 1948), (1394, 1879), (1949, 1799),
                         (2577, 1713)], 10)
