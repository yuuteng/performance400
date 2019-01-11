import cv2
import time
import numpy as np

static_back = None
pts = []
video = cv2.VideoCapture('videos/V0run.MOV')

for i in range(-150, 630):
    frame = video.read()[1]
    if frame is None:
        break

    if i < 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if static_back is None:
        static_back = gray
        continue

    diff_frame = cv2.absdiff(static_back, gray)
    thresh_frame = cv2.threshold(diff_frame, 40, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    contours = cv2.findContours(thresh_frame.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]

    if contours is not None:
        if len(contours) > 0:
            biggestContour = contours[0]

            for contour in contours:
                if cv2.contourArea(contour) > cv2.contourArea(biggestContour):
                    biggestContour = contour

            if cv2.contourArea(biggestContour) > 50000:
                (x, y, w, h) = cv2.boundingRect(biggestContour)
                dr = 400
                x1, y1 = x + w - dr, y + dr
                x2, y2 = x1 + dr, y1 - dr
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                pts.append((int(x1 + dr / 2), int(y1 - dr / 2)))
                ptsmodif = np.array(pts, 'int32')
                ptsmodif = ptsmodif.reshape((-1, 1, 2))
                cv2.polylines(frame, [ptsmodif], isClosed=False, color=(0, 0, 0), thickness=10, lineType=cv2.LINE_AA)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, '   x=' + str(x1 + dr / 2) + ' y= ' + str(y1 - dr / 2),
                            (int(x1 + dr / 2), int(y1 - dr / 2)), font,
                            1.2,
                            (0, 255, 0), 2, cv2.LINE_AA)

    # cv2.imshow("Diff Frame", diff_frame)
    # cv2.imshow("Threshold Frame", thresh_frame)
    cv2.namedWindow("Color Frame", cv2.WINDOW_NORMAL)
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    time.sleep(0)

video.release()
cv2.destroyAllWindows()
