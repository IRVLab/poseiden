import copy
import cv2
import random
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from tools.augmentations import fliplr, flipud, hsv, hidenseek
from tools.utils import to_torch
from tools.common import (
    get_warp_matrix, fix_aspect_ratio, triangulation, gen_reproj_matrix
)


class StereoBaseDataset(Dataset):
    def __init__(self, image_set, cfg):
        self.flip_pairs = []

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.image_set = image_set
        self.num_joints = cfg.dataset.get('num_joints')
        self.image_size = cfg.get('image_size')

        self.scale_factor = cfg.dataset.aug.get('scale_factor', 0)
        self.translate_factor = cfg.dataset.aug.get('translate_factor', 0)
        self.horizontal_flip = cfg.dataset.aug.get('horizontal_flip', False)
        self.vertical_flip = cfg.dataset.aug.get('vertical_flip', False)
        self.color_jitter = cfg.dataset.aug.get('color_jitter', False)
        self.hide_and_seek = cfg.dataset.aug.get('hide_and_seek', False)

    def __len__(self,):
        assert len(self.flip_pairs) > 0, 'Flip pairs are not defined!'
        assert len(self.db) > 0, f'Data not found in {self.data_path}'
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        img_left = cv2.imread(db_rec['image_left'], cv2.IMREAD_COLOR)
        img_right = cv2.imread(db_rec['image_right'], cv2.IMREAD_COLOR)

        if img_left is None:
            raise ValueError('Fail to read {}'.format(db_rec['image_left']))
        if img_right is None:
            raise ValueError('Fail to read {}'.format(db_rec['image_right']))

        # get camera projection matrix
        P_left = db_rec['project_left']
        P_right = db_rec['project_right']

        # get 2D annotations
        pose_left = db_rec['pose_left']
        pose_right = db_rec['pose_right']

        # get array that indicates visibility of the keypoints
        vis_left = db_rec['pose_left_vis']
        vis_right = db_rec['pose_right_vis']

        # get 3D annotations if available
        pose_3d = db_rec.get('pose_3d', None)

        h, w = img_left.shape[:2]
        center = db_rec.get('bbox_center', np.array([w / 2, h / 2]))
        scale = db_rec.get('bbox_scale', np.array([min(h, w), min(h, w)]))
        scale = fix_aspect_ratio(scale, aspect_ratio=1)
        shift = [0, 0]

        # perform data augmentation if in training mode
        if self.image_set == 'train':
            sf = self.scale_factor
            tf = self.translate_factor
            scale *= np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            shift[0] = np.random.uniform(-tf, tf)
            shift[1] = np.random.uniform(-tf, tf)

            if self.vertical_flip and random.random() <= 0.5:
                img_left, pose_left, vis_left = flipud(
                    img_left, pose_left, vis_left,
                    img_left.shape[0], self.flip_pairs)
                img_right, pose_right, vis_right = flipud(
                    img_right, pose_right, vis_right,
                    img_right.shape[0], self.flip_pairs)

                center[1] = img_left.shape[0] - center[1] - 1

            if self.horizontal_flip and random.random() <= 0.5:
                img_left, pose_left, vis_left = fliplr(
                    img_left, pose_left, vis_left,
                    img_left.shape[1], self.flip_pairs)
                img_right, pose_right, vis_right = fliplr(
                    img_right, pose_right, vis_right,
                    img_right.shape[1], self.flip_pairs)

                center[0] = img_left.shape[1] - center[0] - 1

                img_left, img_right = img_right.copy(), img_left.copy()
                pose_left, pose_right = pose_right, pose_left.copy()
                vis_left, vis_right = vis_right, vis_left.copy()

            # triangulate new 3D pose after performing flipping
            # and affine transformation
            if pose_3d is not None:
                pose_3d = triangulation(
                    P_left, P_right, pose_left, pose_right)

            if self.color_jitter and random.random() <= 0.3:
                img_left = hsv(img_left)
            if self.color_jitter and random.random() <= 0.3:
                img_right = hsv(img_right)

            if self.hide_and_seek and random.random() <= 0.3:
                img_left = hidenseek(img_left, pose_left, vis_left)
            if self.hide_and_seek and random.random() <= 0.3:
                img_right = hidenseek(img_right, pose_right, vis_right)

        trans = get_warp_matrix(center, scale, 0, self.image_size, shift=shift)

        # crop and transform image and annotations
        img_left, pose_left, P_left = self.preprocess(
            img_left, pose_left, P_left, trans)
        img_right, pose_right, P_right = self.preprocess(
            img_right, pose_right, P_right, trans)

        # check if the joints are within the image boundary after
        # data augmentation
        vis_left = self._check_boundary(pose_left, vis_left)
        vis_right = self._check_boundary(pose_right, vis_right)

        joints_vis = np.logical_and(vis_left, vis_right)
        joints_vis = np.logical_and.reduce(joints_vis, axis=1, keepdims=True)

        # convert images to torch.tensor and normalize it
        img_left = self.transform(img_left)
        img_right = self.transform(img_right)

        P_left = np.vstack((P_left, np.array([0, 0, 0, 1])))
        P_right = np.vstack((P_right, np.array([0, 0, 0, 1])))

        # create disparity-to-depth mapping matrix
        Q = gen_reproj_matrix(P_left, P_right, db_rec['baseline'])

        meta = {
            'image_left': db_rec['image_left'],
            'image_right': db_rec['image_right'],
            'pose_left': to_torch(pose_left),
            'pose_right': to_torch(pose_right),
            'project_left': to_torch(P_left),
            'project_right': to_torch(P_right),
            'baseline': to_torch(db_rec['baseline']),
            'Q': to_torch(Q),
            'joints_vis': to_torch(joints_vis),
        }

        if pose_3d is not None:
            meta.update({'pose_3d': to_torch(pose_3d)})

        return img_left, img_right, meta

    def _get_db(self):
        raise NotImplementedError

    def _check_boundary(self, joints, joints_vis):
        pos_valid = (
            (joints[:, 0] >= 0) & (joints[:, 0] < self.image_size[1])
            & (joints[:, 1] >= 0) & (joints[:, 1] < self.image_size[0])
        )

        joints_vis[~pos_valid, :] = False

        return joints_vis

    def preprocess(self, image, joints, P, trans):
        """
        Resize images and joints accordingly for model training.
        If in training stage, random flip, scale, and rotation will be applied.

        Args:
            image: input image
            joints: ground truth keypoints: [num_joints, 2 or 3]
            P: camera projection matrix
            trans: transformation matrix
        Returns:
            image, joints, joints_vis, P (after preprocessing)
        """

        image = cv2.warpAffine(
            image,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        joints[:, :2] = cv2.transform(joints[None, :, :2], trans)[0]

        T = np.eye(3)
        T[:2, :3] = trans
        P = T @ P

        return image, joints, P
