import cv2
import numpy as np

cap = cv2.VideoCapture('videos/videoMire.MOV')

video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
frames = []
if cap.isOpened() and video_length > 0:
        frame_ids = [0]
        if video_length >= 4:
            frame_ids = [0,
                         round(video_length * 0.25),
                         round(video_length * 0.33),
                         round(video_length * 0.5),
                         round(video_length * 0.66),
                         round(video_length * 0.75),
                         round(video_length * 0.9),
                         video_length - 1]
        count = 0
        success, image = cap.read()
        while success:
            if count in frame_ids:
                frames.append(image)
                cv2.imwrite('images/photosMire/mire' + str(count) + '.jpg', image)
            success, image = cap.read()
            count += 1

