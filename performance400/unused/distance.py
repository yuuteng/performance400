# import the necessary packages
from __future__ import print_function

import cv2
import imutils

cap = cv2.VideoCapture(0)

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
# 　使用opencv默认的SVM分类器
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 打开摄像头
camera = cv2.VideoCapture(0)

while camera.isOpened():
    # get a frame
    (grabbed, frame) = camera.read()

    # 如果不能抓取到一帧，说明我们到了视频的结尾
    if not grabbed:
        break

    frame = imutils.resize(frame, width=min(400, frame.shape[1]))

    # marker = find_marker(frame)
    marker = find_person(frame)

    # inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
    for (xA, yA, xB, yB) in marker:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        ya_max = yA
        yb_max = yB

    pix_person_height = yb_max - ya_max
    if pix_person_height == 0:
        # pix_person_height = 1
        continue
        print(pix_person_height)
    # print (pix_person_height)
    inches = distance_to_camera(KNOW_PERSON_HEIGHT, focalLength, pix_person_height)
    print("%.2fcm" % (inches * 30.48 / 12))
    # draw a bounding box around the image and display it
    # box = np.int0(cv2.cv.BoxPoints(marker))
    # cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
    cv2.putText(frame, "%.2fcm" % (inches * 30.48 / 12),
                (frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (0, 255, 0), 3)

    # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
