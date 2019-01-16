import cv2
import numpy as np
import time
from performance400.find_3D_coords_mono import calculate_3d_coords


def get_bf_matches(m_H, m_left_image, m_right_image, m_kp1, m_kp2, m_des1, m_des2):
    m_matcher = cv2.BFMatcher()
    m_matches = m_matcher.match(m_des1, m_des2)

    # list.sort(m_matches, key=lambda e: get_transfer_error(e, kp1, kp2))
    list.sort(m_matches, key=lambda e: get_transfer_error(m_H, e, m_kp1, m_kp2))

    m_img = cv2.drawMatches(m_left_image, m_kp1, m_right_image, m_kp2, m_matches[:2], None, flags=2)

    return m_matches, m_img


def get_flann_matches(m_H, m_left_image, m_right_image, m_kp1, m_kp2, m_des1, m_des2):
    FLANN_INDEX_LSH = 6
    m_index_params = dict(algorithm=FLANN_INDEX_LSH,
                          table_number=6,  # 12
                          key_size=12,  # 20
                          multi_probe_level=1)  # 2
    m_search_params = dict(checks=50)

    m_flann = cv2.FlannBasedMatcher(m_index_params, m_search_params)
    m_matches = m_flann.knnMatch(m_des1, m_des2, k=2)
    m_matches_mask = [[0, 0] for i in range(len(m_matches))]
    m_good = []

    for j in range(len(m_matches)):
        if len(m_matches[j]) != 2:
            continue
        (m, n) = m_matches[j]
        if m.distance < 0.7 * n.distance:
            m_matches_mask[j] = [1, 0]
            m_good.append(m)

    list.sort(m_good, key=lambda e: get_transfer_error(m_H, e, m_kp1, m_kp2))


    # m_draw_params = dict(matchColor=(0, 255, 0),
    #                      singlePointColor=(255, 0, 0),
    #                      matchesMask=m_matches_mask,
    #                      flags=0)

    # m_img = cv2.drawMatchesKnn(m_left_image, m_kp1, m_right_image, m_kp2, m_matches, None, **m_draw_params)
    m_img = cv2.drawMatches(m_left_image, m_kp1, m_right_image, m_kp2, m_good[:10], None, flags=2)

    return m_good, m_img


def get_transfer_error(m_H, m_match, m_kp1, m_kp2):
    m_xp = [m_kp1[m_match.queryIdx].pt[0], m_kp1[m_match.queryIdx].pt[1], 0]
    m_xpp = [m_kp2[m_match.trainIdx].pt[0], m_kp2[m_match.trainIdx].pt[1], 0]

    return np.linalg.norm(m_xp @ m_H - m_xpp) + np.linalg.norm(m_xpp @ np.linalg.inv(m_H) - m_xp)


left_video = cv2.VideoCapture("videos/runway/gauche.mp4")  # query
right_video = cv2.VideoCapture("videos/runway/droite.mp4")  # train

while True:
    left_check, left_image = left_video.read()
    right_check, right_image = right_video.read()

    # left_check = True
    # right_check = True

    # left_image = cv2.imread("videos/runway/gauche.mp4")
    # right_image = cv2.imread("videos/runway/droite.mp4")

    if not (left_check and right_check):
        break

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(left_image, None)
    kp2, des2 = orb.detectAndCompute(right_image, None)

    p1 = np.array([(1486, 270), (948, 688), (1491, 702), (1824, 490)])
    p2 = np.array([(735, 280), (219, 788), (797, 698), (1050, 457)])

    H, _ = cv2.findHomography(p1, p2)

    (matches, img) = get_flann_matches(H, left_image, right_image, kp1, kp2, des1, des2)
    # (matches, img) = get_bf_matches(H, left_image, right_image, kp1, kp2, des1, des2)

    p1 = np.asarray([kp1[matches[i].queryIdx].pt for i in range(len(matches))])
    p2 = np.asarray([kp2[matches[i].trainIdx].pt for i in range(len(matches))])

    # H, _ = cv2.findHomography(p1, p2)

    w, h = left_image.shape[:2]
    # img = cv2.warpPerspective(right_image, H, (h, w))

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", img)

    if cv2.waitKey(1) == ord('q'):
        break

    # time.sleep(1)

cv2.destroyAllWindows()
left_video.release()
right_video.release()
