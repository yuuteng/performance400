import cv2
import numpy as np

image = cv2.imread("images/first_frame_right.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

_, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    if cv2.contourArea(contour) > 1000000:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.fillConvexPoly(image, np.asarray([(x, y), (x + w, y), (x + w, y + h), (x, y + h)]), (0, 0, 0))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(gray, 3)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, -5)

thresh = cv2.erode(thresh, None, iterations=1)

_, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    if cv2.contourArea(contour) > 50:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.fillConvexPoly(thresh, np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)]), (0, 0, 0))

pixels = cv2.findNonZero(thresh)

w, h = image.shape[:2]
pixels_matrix = np.zeros((h, w))

for pixel in pixels:
    pixels_matrix[pixel[0][0]][pixel[0][1]] = 1

good = [(pixels[i][0][0], pixels[i][0][1]) for i in range(len(pixels))]

# good = []
# for i in range(1, h - 1):
#     for j in range(1, w - 1):
#         if pixels_matrix[i - 1][j - 1] + pixels_matrix[i - 1][j] + pixels_matrix[i - 1][j + 1] + pixels_matrix[i][
#             j - 1] + pixels_matrix[i][j + 1] + pixels_matrix[i + 1][j - 1] + pixels_matrix[i + 1][j] + \
#                 pixels_matrix[i + 1][j + 1] > 1:
#             good.append((i, j))
#
#     print(f"{i * 100 / (h - 1)}% done")

for pixel in good:
    cv2.circle(image, pixel, 1, (255, 0, 0), thickness=2)

cv2.namedWindow("res", cv2.WINDOW_NORMAL)
cv2.imshow("res", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
