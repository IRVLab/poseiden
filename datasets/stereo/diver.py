import os
import json
import glob
import numpy as np

from .stereobase import StereoBaseDataset


class DiverDataset(StereoBaseDataset):
    def __init__(self, data_dir, image_set, cfg):
        super().__init__(image_set, cfg)
        if image_set == "train":
            self.data_path = os.path.join(data_dir, "train")
        elif image_set == "val":
            # there is no validation data
            self.data_path = os.path.join(data_dir, "train")
        else:
            raise ValueError("Invalid image set")
        self.flip_pairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]
        self.db = self._get_db()

    def _get_db(self):
        l_img_paths = sorted(glob.glob(
            os.path.join(self.data_path, "**", "left", "*_left.png")))
        r_img_paths = sorted(glob.glob(
            os.path.join(self.data_path, "**", "right", "*_right.png")))
        label_paths = sorted(glob.glob(
            os.path.join(self.data_path, "**", "annots", "*.json")))

        assert len(l_img_paths) == len(r_img_paths) == len(label_paths), \
            "Number of images and ground truths must match, " \
            "left: {}, right: {}, gt: {}".format(
                len(l_img_paths), len(r_img_paths), len(label_paths))

        gt_db = []
        for i in range(len(l_img_paths)):
            with open(label_paths[i], 'r') as f:
                data = json.load(f)

                pose_left = np.array(data['keypoints_left']).reshape(-1, 3)
                pose_right = np.array(data['keypoints_right']).reshape(-1, 3)
                cam_params = data['cam_params']

            # The third idx indicates visibility of the keypoints
            # v=0: not labeled (in which case x=y=0)
            # v=1: labeled but not visible
            # v=2: labeled and visible
            vis_left = (pose_left[:, 2:] == 2).astype(np.bool_)
            vis_right = (pose_right[:, 2:] == 2).astype(np.bool_)

            x1 = (pose_left[:, 0].min() + pose_right[:, 0].min()) / 2
            x2 = (pose_left[:, 0].max() + pose_right[:, 0].max()) / 2
            y1 = (pose_left[:, 1].min() + pose_right[:, 1].min()) / 2
            y2 = (pose_left[:, 1].max() + pose_right[:, 1].max()) / 2
            bbox_center = np.asarray([(x1 + x2) / 2, (y1 + y2) / 2])
            bbox_scale = np.array([(x2 - x1) * 2.0, (y2 - y1) * 2.2])

            # Only use the images from right camera for 2d human pose training
            gt_db.append({
                'image_left': l_img_paths[i],
                'image_right': r_img_paths[i],
                'pose_left': pose_left[:, :2],
                'pose_right': pose_right[:, :2],
                'pose_left_vis': vis_left,
                'pose_right_vis': vis_right,
                'baseline': np.asarray([cam_params['baseline']]),
                'project_left': np.asarray(cam_params['P1']),
                'project_right': np.asarray(cam_params['P2']),
                'bbox_center': bbox_center,
                'bbox_scale': bbox_scale,
                'K_left': np.asarray(cam_params['K1']),
                'K_right': np.asarray(cam_params['K2']),
                'T_cr_cl': np.asarray(cam_params['T_cr_cl'])
            })

        return gt_db
