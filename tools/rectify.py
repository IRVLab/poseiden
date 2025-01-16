import cv2
import yaml
import numpy as np


class Camera:
    def __init__(self,
                 width=None,
                 height=None,
                 cam_matrix=np.eye(3),
                 dist=np.zeros(5),
                 rvec=np.eye(9),
                 tvec=np.zeros(3),
                 rect_matrix=None,
                 proj_matrix=None):

        self.width = width
        self.height = height
        self.cam_matrix = cam_matrix
        self.dist = dist
        self.rvec = rvec
        self.tvec = tvec
        self.rect_matrix = rect_matrix
        self.proj_matrix = proj_matrix


class Rectificator:
    def __init__(self, camera_left, camera_right):
        assert isinstance(camera_left, Camera) and \
            isinstance(camera_right, Camera), \
            'camera_left and camera_right must be instances of Camera'

        self.meta, self.cam_params = self.run(camera_left, camera_right)

    @staticmethod
    def load_camera_parameters(yaml_file):
        with open(yaml_file, 'r') as file:
            camera_params = yaml.safe_load(file)

        width = camera_params['image_width']
        height = camera_params['image_height']

        camera_matrix = np.array(
            camera_params['camera_matrix']['data']).reshape(3, 3)
        dist_coeffs = np.array(
            camera_params['distortion_coefficients']['data']).reshape(1, 5)
        rect_matrix = np.array(
            camera_params['rectification_matrix']['data']).reshape(3, 3)
        proj_matrix = np.array(
            camera_params['projection_matrix']['data']).reshape(3, 4)

        camera = Camera(
            width=width,
            height=height,
            cam_matrix=camera_matrix,
            dist=dist_coeffs,
            rect_matrix=rect_matrix,
            proj_matrix=proj_matrix)

        return camera

    @staticmethod
    def parse_calibration_data(calibration_file_path):
        def gen_intrinsic(camera_intrinsic):
            K = np.array([
                [camera_intrinsic[0], 0, camera_intrinsic[2]],
                [0, camera_intrinsic[1], camera_intrinsic[3]],
                [0, 0, 1]
            ], dtype=np.float32)

            return K

        # Load calibration data
        with open(calibration_file_path, 'r') as f:
            calib_data = yaml.safe_load(f)

        # Extract camera parameters
        left_camera_intrinsic = \
            gen_intrinsic(calib_data['cam0']['intrinsics'])
        left_dist_coeffs = np.array(calib_data['cam0']['distortion_coeffs'])
        right_camera_intrinsic = \
            gen_intrinsic(calib_data['cam1']['intrinsics'])
        right_dist_coeffs = np.array(calib_data['cam1']['distortion_coeffs'])
        trans_mat = np.array(calib_data['cam1']['T_cn_cnm1']).reshape(4, 4)

        camera_left = Camera(
            width=calib_data['cam0']['resolution'][0],
            height=calib_data['cam0']['resolution'][1],
            cam_matrix=left_camera_intrinsic,
            dist=left_dist_coeffs,
            rvec=np.eye(3),
            tvec=np.zeros(3))

        # trans_mat[:3, 3:] * 1000 -> is to convert the translation
        # from meters to mm
        camera_right = Camera(
            width=calib_data['cam1']['resolution'][0],
            height=calib_data['cam1']['resolution'][1],
            cam_matrix=right_camera_intrinsic,
            dist=right_dist_coeffs,
            rvec=trans_mat[:3, :3],
            tvec=trans_mat[:3, 3:] * 1000)

        return camera_left, camera_right

    def run(self, cam_left, cam_right):
        assert cam_left.width == cam_right.width and \
            cam_left.height == cam_right.height, \
            'Image size of left and right camera must be the same'

        width = cam_left.width
        height = cam_left.height

        # Prepare rectify parameters
        if cam_left.rect_matrix is not None and \
            cam_right.rect_matrix is not None and \
                cam_left.proj_matrix is not None and \
                cam_right.proj_matrix is not None:

            R1 = cam_left.rect_matrix
            R2 = cam_right.rect_matrix
            P1 = cam_left.proj_matrix
            P2 = cam_right.proj_matrix
        else:

            # R-> Rotation matrix between the coordinate systems of the first
            # and the second cameras
            # T-> Translation vector between the coordinate systems of the
            # cameras

            assert np.all(cam_left.rvec == np.eye(3)) and \
                np.all(cam_left.tvec == np.zeros(3)), \
                'Rotation and translation vectors of left camera must be ' \
                'identity and zero, respectively, representing the world ' \
                'coordinate system.'

            assert np.any(cam_right.tvec != np.zeros(3)), \
                'Translation vector of right camera must not be zero.'

            R = cam_right.rvec
            T = cam_right.tvec

            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                cam_left.cam_matrix, cam_left.dist,
                cam_right.cam_matrix, cam_right.dist,
                (width, height), R, T)

        map1_left, map2_left = cv2.initUndistortRectifyMap(
            cam_left.cam_matrix, cam_left.dist,
            R1, P1, (width, height), cv2.CV_32FC1)
        map1_right, map2_right = cv2.initUndistortRectifyMap(
            cam_right.cam_matrix, cam_right.dist,
            R2, P2, (width, height), cv2.CV_32FC1)

        cam_params = {
            # rectified intrinsic matrix of left camera
            'K1': P1[:, :3].tolist(),
            # rectified intrinsic matrix of right camera
            'K2': P2[:, :3].tolist(),
            # projection matrix that projects points given in the rectified
            # first camera coordinate system into the rectified first
            # camera's image
            'P1': P1.tolist(),
            # projection matrix that projects points given in the rectified
            # first camera coordinate system into the rectified second
            # camera's image.
            'P2': P2.tolist(),
            # baseline (translation between the two cameras)
            'baseline': -P2[0][-1] / P1[0][0],
            # transformation matrix that takes points in left camera coordinate
            # to right camera coordinate
            'T_cr_cl': [[1, 0, 0, P2[0][-1] / P1[0][0]],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]]
        }

        meta = {
            'left': {
                'camera_intrinsic': cam_left.cam_matrix,
                'dist_coeffs': cam_left.dist,
                'R': R1,
                'P': P1,
                'map1': map1_left,
                'map2': map2_left
            },
            'right': {
                'camera_intrinsic': cam_right.cam_matrix,
                'dist_coeffs': cam_right.dist,
                'R': R2,
                'P': P2,
                'map1': map1_right,
                'map2': map2_right
            },
        }

        return meta, cam_params

    def rectify_images(self, side, img):
        """
        Rectify the input image

        Args:
            side (str): 'left' or 'right'
            img (np.ndarray): input image

        Returns:
            np.ndarray: rectified image
        """
        map1 = self.meta[side]['map1']
        map2 = self.meta[side]['map2']
        img_rectified = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

        return img_rectified

    def rectify_annots(self, side, kpts, bbox):
        """
        Rectify the annotations so that they are consistent with the
        rectified image

        Args:
            side (str): 'left' or 'right'
            kpts (np.ndarray [N, 3]): keypoints
            bbox (np.ndarray (4,)): bounding box

        Returns:
            np.ndarray: rectified keypoints
            np.ndarray: rectified bounding box
        """
        assert kpts.ndim == 2, 'kpts must have 2 dimensions'
        assert kpts.shape[1] == 3, 'kpts must have 3 columns'
        assert bbox.shape == (4,), 'bbox must be (x, y, w, h)'

        intrinsics = self.meta[side]['camera_intrinsic']
        dist_coeffs = self.meta[side]['dist_coeffs']
        R = self.meta[side]['R']
        P = self.meta[side]['P']

        kpts_rectified = cv2.undistortPoints(
            kpts[:, :2].reshape((-1, 1, 2)), intrinsics, dist_coeffs, R=R, P=P)
        kpts_rectified = kpts_rectified.reshape(-1, 2)
        kpts_rectified = np.hstack([kpts_rectified, kpts[:, 2:]])

        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
        bbox = np.array([[x1, y1], [x2, y2]])
        bbox_rectified = cv2.undistortPoints(
            bbox.reshape((2, 1, 2)), intrinsics, dist_coeffs, R=R, P=P)
        x1, y1, x2, y2 = bbox_rectified.flatten()
        bbox_rectified = np.array([x1, y1, x2 - x1, y2 - y1])

        return kpts_rectified, bbox_rectified

    def get_cam_params(self):
        return self.cam_params
