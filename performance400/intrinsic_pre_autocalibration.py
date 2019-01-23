import cv2
import numpy as np


def autocalibrate(left_targets, right_targets, width, height):
    # critère d'arrêt pour la recherche de mire
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Traitement pour la camera de gauche
    # creation artificelle des points objet de la mire
    object_point = np.zeros((width * height, 3), np.float32)
    object_point[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    # Listes pour stocker les points objet et image que l'on parviendra à détecter
    object_points = []  # 3d point in real world space
    image_points = []  # 2d points in image plane.

    count = 0
    gray = None
    for file_name in left_targets:
        img = cv2.imread(file_name)
        img = cv2.threshold(img, 200, 255, cv2.THRESH_TRUNC)[1]  # si image floue on permet une detection mais moins
        # precise
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
        print(ret)

        # Si on trouve des points, on affine leur posistion et on les stock
        if ret:
            count += 1
            object_points.append(object_point)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            image_points.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            cv2.namedWindow("mire cali gauche", cv2.WINDOW_NORMAL)
            cv2.imshow("mire cali gauche", img)
            cv2.waitKey(0)

    # calcul des paramètres intrinsèques à partir des points trouvés
    (_, intrinsic_left_camera_matrix, intrinsic_left_distortion_vector, _, _) = cv2.calibrateCamera(object_points,
                                                                                                    image_points,
                                                                                                    gray.shape[::-1],
                                                                                                    None, None)
    # Même traitement pour la caméra de droite
    object_point = np.zeros((width * height, 3), np.float32)
    object_point[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    object_points = []
    image_points = []

    count = 0
    for file_name in right_targets:
        img = cv2.imread(file_name)
        img = cv2.threshold(img, 200, 255, cv2.THRESH_TRUNC)[1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
        print(ret)

        if ret:
            count += 1
            object_points.append(object_point)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            image_points.append(corners2)

            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            cv2.namedWindow("mire cali droite", cv2.WINDOW_NORMAL)
            cv2.imshow("mire cali droite", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    (_, intrinsic_right_camera_matrix, intrinsic_right_distortion_vector, _, _) = cv2.calibrateCamera(object_points,
                                                                                                      image_points,
                                                                                                      gray.shape[::-1],
                                                                                                      None, None)

    np.savetxt('matrices/camera_matrices/intrinsic/left', intrinsic_left_camera_matrix)
    np.savetxt('matrices/distortion_vectors/intrinsic/left',
               intrinsic_left_distortion_vector)
    np.savetxt('matrices/camera_matrices/intrinsic/right', intrinsic_right_camera_matrix)
    np.savetxt('matrices/distortion_vectors/intrinsic/right',
               intrinsic_right_distortion_vector)


def extract_targets(video, nbr, right_camera):
    prefix = 'right' if right_camera else 'left'
    video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_ids = None
    if video.isOpened() and video_length > 0:
        if video_length >= 4:
            frame_ids = np.arange(0, video_length, round(video_length / nbr))
        count = 0
        success, image = video.read()
        while success:
            if count in frame_ids:
                print('ok '+prefix, count)
                cv2.imwrite('images/targets/'+prefix+'/'+str(round(count*nbr/video_length))+'.jpg', image)
            success, image = video.read()
            count += 1


def get_intrinsic_parameters():
    intrinsic_left_camera_matrix = np.loadtxt('matrices/camera_matrices/intrinsic/left')
    intrinsic_left_distortion_vector = np.loadtxt(
        'matrices/distortion_vectors/intrinsic/left')
    intrinsic_right_camera_matrix = np.loadtxt('matrices/camera_matrices/intrinsic/right')
    intrinsic_right_distortion_vector = np.loadtxt(
        'matrices/distortion_vectors/intrinsic/right')

    return (intrinsic_left_camera_matrix, intrinsic_left_distortion_vector,
            intrinsic_right_camera_matrix, intrinsic_right_distortion_vector)
