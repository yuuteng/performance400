import cv2
import numpy as np

cap = cv2.VideoCapture('videos/videoMire.MOV')

video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

if cap.isOpened() and video_length > 0:
        frame_ids = [0]
        if video_length >= 4:
            frame_ids = [0,
                         round(video_length * 0.10),
                         round(video_length * 0.20),
                         round(video_length * 0.30),
                         round(video_length * 0.40),
                         round(video_length * 0.50),
                         round(video_length * 0.60),
                         round(video_length * 0.70),
                         round(video_length * 0.80),
                         round(video_length * 0.90),
                         video_length - 1]
        count = 0
        success, image = cap.read()
        print(frame_ids)
        while success:
            if count in frame_ids:
                print('ah', count)
                cv2.imwrite('images/targets/targets' + str(count) + '.jpg', image)
            success, image = cap.read()
            count += 1

