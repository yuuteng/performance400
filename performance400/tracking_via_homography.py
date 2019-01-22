import cv2
import numpy as np


def get_bf_matches(m_H, m_left_image, m_right_image, m_kp1, m_kp2, m_des1, m_des2):
    m_matcher = cv2.BFMatcher()
    m_matches = m_matcher.match(m_des1, m_des2)

    # list.sort(m_matches, key=lambda e: get_transfer_error(e, kp1, kp2))
    list.sort(m_matches, key=lambda e: get_transfer_error(m_H, e, m_kp1, m_kp2))

    m_good = []
    for match in m_matches:
        if get_transfer_error(m_H, match, m_kp1, m_kp2) < 100:
            m_good.append(match)

    m_matches = m_good

    # m_img = cv2.drawMatches(m_left_image, m_kp1, m_right_image, m_kp2, m_matches, None, flags=2)

    # return m_matches, m_img
    return m_matches


def get_flann_matches(m_H, m_left_image, m_right_image, m_kp1, m_kp2, m_des1, m_des2):
    FLANN_INDEX_LSH = 6
    m_index_params = dict(algorithm=FLANN_INDEX_LSH,
                          table_number=6,  # 12
                          key_size=12,  # 20
                          multi_probe_level=1)  # 2
    m_search_params = dict(checks=50)

    m_flann = cv2.FlannBasedMatcher(m_index_params, m_search_params)
    m_matches = m_flann.knnMatch(m_des1, m_des2, k=2)
    m_matches_mask = [[0, 0] for j in range(len(m_matches))]
    m_good = []

    for j in range(len(m_matches)):
        if len(m_matches[j]) != 2:
            continue
        (m, n) = m_matches[j]
        if m.distance < 0.7 * n.distance:
            m_matches_mask[j] = [1, 0]
            m_good.append(n)

    m_filtered_good = []
    for good in m_good:
        if get_transfer_error(m_H, good, m_kp1, m_kp2) < 4:
            m_filtered_good.append(good)

    # m_draw_params = dict(matchColor=(0, 255, 0),
    #                      singlePointColor=(255, 0, 0),
    #                      matchesMask=m_matches_mask,
    #                      flags=0)

    # m_img = cv2.drawMatchesKnn(m_left_image, m_kp1, m_right_image, m_kp2, m_matches, None, **m_draw_params)

    return m_filtered_good


def get_transfer_error(m_H, m_match, m_kp1, m_kp2):
    m_xp = np.transpose([m_kp1[m_match.queryIdx].pt[0], m_kp1[m_match.queryIdx].pt[1], 1])
    m_xpp = np.transpose([m_kp2[m_match.trainIdx].pt[0], m_kp2[m_match.trainIdx].pt[1], 1])

    m_transformed_xp = m_H @ m_xp
    m_transformed_xp[2] = 1

    m_transformed_xpp = np.linalg.inv(m_H) @ m_xpp
    m_transformed_xpp[2] = 1

    e1 = (np.linalg.norm(m_transformed_xp - m_xpp) * 0.01) ** 2
    e2 = (np.linalg.norm(m_transformed_xpp - m_xp) * 0.01) ** 2

    # print(f"1 {m_xp} -> {m_transformed_xp} vs {m_xpp} ; e: {e1}")
    # print(f"2 {m_xpp} -> {m_transformed_xpp} vs {m_xp} ; e: {e2}")

    return e1 + e2


left_video = cv2.VideoCapture("videos/runway/gauche.mp4")  # query
right_video = cv2.VideoCapture("videos/runway/droite.mp4")  # train

left_calibration_points = np.array(
    [(735, 280), (219, 788), (797, 698), (1054, 906), (182, 1041)]) * 2
right_calibration_points = np.array(
    [(1486, 270), (948, 688), (1491, 702), (1682, 945), (848, 886)]) * 2

# H * [[xi], [yi], [1]] ~ si * [[xi'], [yi'], [1]]
H, _ = cv2.findHomography(left_calibration_points, right_calibration_points)

sum = 0
for i in range(len(left_calibration_points)):
    p1 = left_calibration_points[i]
    p2 = right_calibration_points[i]
    sum += np.linalg.norm((H @ np.transpose([p1[0], p1[1], 1]))[:2] - np.transpose(p2))
    sum += np.linalg.norm((np.linalg.inv(H) @ np.transpose([p2[0], p2[1], 1]))[:2] - np.transpose(p1))

print(f"Calibration error: {sum/len(left_calibration_points)}")

while True:
    left_check, left_image = left_video.read()
    right_check, right_image = right_video.read()

    if not (left_check and right_check):
        break

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(left_image, None)
    kp2, des2 = orb.detectAndCompute(right_image, None)

    matches = get_flann_matches(H, left_image, right_image, kp1, kp2, des1, des2)
    # (matches, img) = get_bf_matches(H, left_image, right_image, kp1, kp2, des1, des2)

    for i in range(len(left_calibration_points)):
        kp1.append(cv2.KeyPoint(left_calibration_points[i][0], left_calibration_points[i][1], -1))
        kp2.append(cv2.KeyPoint(right_calibration_points[i][0], right_calibration_points[i][1], -1))
        matches.append(cv2.DMatch(len(kp1) - 1, len(kp2) - 1, -1))

    img = cv2.drawMatches(left_image, kp1, right_image, kp2, matches, None, flags=2)

    left_points = np.asarray([kp1[matches[i].queryIdx].pt for i in range(len(matches))])
    right_points = np.asarray([kp2[matches[i].trainIdx].pt for i in range(len(matches))])

    H2, _ = cv2.findHomography(left_points, right_points)

    w, h = left_image.shape[:2]
    # img = cv2.warpPerspective(left_image, H2, (h, w))

    # cv2.namedWindow("right", cv2.WINDOW_NORMAL)
    # cv2.imshow("right", right_image)

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", img)

    if cv2.waitKey(1) == ord('q'):
        break

    # time.sleep(3)

cv2.destroyAllWindows()
left_video.release()
right_video.release()
