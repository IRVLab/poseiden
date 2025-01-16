import copy
import cv2
import random
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from itertools import product

from tools.augmentations import fliplr, hsv
from tools.utils import to_torch
from tools.common import get_warp_matrix


class MonocularBaseDataset(Dataset):
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

        self.sigma = cfg.get('sigma')
        self.heatmap_size = cfg.get('heatmap_size')

        self.scale_factor = cfg.dataset.aug.get('scale_factor', 0)
        self.rotation_factor = cfg.dataset.aug.get('rotation_factor', 0)
        self.translate_factor = cfg.dataset.aug.get('translate_factor', 0)
        self.horizontal_flip = cfg.dataset.aug.get('horizontal_flip', False)
        self.color_jitter = cfg.dataset.aug.get('color_jitter', False)
        self.hide_and_seek = cfg.dataset.aug.get('hide_and_seek', False)

    def __len__(self,):
        assert len(self.db) > 0, f'Data not found in {self.data_path}'
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']

        data_numpy = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if data_numpy is None:
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d'][:, :2]
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.color_jitter and random.random() <= 0.5:
            data_numpy = hsv(data_numpy)

        input, joints, joints_vis = self.preprocess(
            data_numpy, joints, joints_vis, c, s, r, 200)

        # convert images to torch.tensor and normalize it
        input = self.transform(input)

        # convert 2d keypoints to heatmaps
        target, target_weight = \
            self.gen_gaussian_heatmaps(joints[None, :], joints_vis[None, :, 0])
        target_weight = target_weight[0]

        target = to_torch(target)
        target_weight = to_torch(target_weight)

        target_weight = target_weight.squeeze(-1)

        meta = {
            'image': image_file,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        return input, target, target_weight, meta

    def _get_db(self):
        raise NotImplementedError

    def preprocess(self, image, joints, joints_vis, c, s, r, origin_size):
        """
        Resize images and joints accordingly for model training.
        If in training stage, random flip, scale, and rotation will be applied.

        Args:
            image: input image
            joints: ground truth keypoints: [num_joints, 3]
            joints_vis: visibility of the keypoints: [num_joints, 3],
                        (1: visible, 0: invisible)
            c: center point of the cropped region
            s: scale factor
            r: degree of rotation
            origin_size: original size of the cropped region
        Returns:
            image, joints, joints_vis (after preprocessing)
        """
        if self.image_set == 'train':
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0

            if self.horizontal_flip and random.random() <= 0.5:
                image, joints, joints_vis = fliplr(
                    image, joints, joints_vis, image.shape[1], self.flip_pairs)
                c[0] = image.shape[1] - c[0] - 1

        trans = get_warp_matrix(c, s * 200, r, self.image_size)
        image = cv2.warpAffine(
            image,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        joints[:, :2] = cv2.transform(joints[None, :, :2], trans)[0]

        return image, joints, joints_vis

    def gen_gaussian_heatmaps(self, keypoints, keypoints_visible):
        """Generate unbiased gaussian heatmaps of keypoints using
           Dark Pose: https://arxiv.org/abs/1910.06278

        Args:
            keypoints (np.ndarray):
                Keypoint coordinates in shape (N, K, 2)
            keypoints_visible (np.ndarray):
                Keypoint visibilities in shape (N, K)

        Returns:
            - heatmaps (np.ndarray):
                The generated heatmap in shape (K, H, W) where [W, H] is the
                `heatmap_size`
            - keypoint_weights (np.ndarray):
                The target weights in shape (N, K)
        """
        N, K, _ = keypoints.shape
        W, H = self.heatmap_size

        heatmaps = np.zeros((K, H, W), dtype=np.float32)
        keypoint_weights = keypoints_visible.copy()

        # 3-sigma rule
        radius = self.sigma * 3

        # xy grid
        x = np.arange(0, W, 1, dtype=np.float32)
        y = np.arange(0, H, 1, dtype=np.float32)[:, None]

        for n, k in product(range(N), range(K)):
            # skip unlabled keypoints
            if keypoints_visible[n, k] < 0.5:
                continue

            mu = keypoints[n, k] / (self.image_size[0] / self.heatmap_size[0])
            # check that the gaussian has in-bounds part
            left, top = mu - radius
            right, bottom = mu + radius + 1

            if left >= W or top >= H or right < 0 or bottom < 0:
                keypoint_weights[n, k] = 0
                continue

            gaussian = np.exp(
                -((x - mu[0])**2 + (y - mu[1])**2) / (2 * self.sigma**2))

            _ = np.maximum(gaussian, heatmaps[k], out=heatmaps[k])

        return heatmaps, keypoint_weights
