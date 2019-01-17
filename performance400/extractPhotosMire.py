import cv2
import numpy as np

cap = cv2.VideoCapture('videos/runway/gauche.mp4')

video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

if cap.isOpened() and video_length > 0:
        frame_ids = [0]
        if video_length >= 4:
            frame_ids = [10]
        count = 0
        success, image = cap.read()
        print(frame_ids)
        while success:
            if count in frame_ids:
                print('ah', count)
                cv2.imwrite('images/piste_camera_gauche' + str(count) + '.jpg', image)
            success, image = cap.read()
            count += 1

