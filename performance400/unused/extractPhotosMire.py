import cv2
import numpy as np

cap = cv2.VideoCapture('videos/runway/course_3_droite.MOV')

video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

if cap.isOpened() and video_length > 0:
        frame_ids = [0]
        if video_length >= 4:
            frame_ids = [round(video_length*0.70),
                         round(video_length * 0.72),
                         round(video_length * 0.74),
                         round(video_length * 0.76),
                         round(video_length * 0.78),
                         round(video_length * 0.80),
                         round(video_length * 0.82),
                         round(video_length * 0.84),
                         round(video_length * 0.86),
                         round(video_length * 0.88),
                         round(video_length * 0.90),
                         round(video_length * 0.92),
                         round(video_length * 0.94),
                         round(video_length * 0.96),
                         round(video_length * 0.98),
                         -1]
        count = 0
        success, image = cap.read()
        print(frame_ids)
        while success:
            if count in frame_ids:
                print('ah', count)
                cv2.imwrite('images/targets/target' + str(count) + '.jpg', image)
            success, image = cap.read()
            count += 1
