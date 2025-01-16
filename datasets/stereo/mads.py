import os
import json
import glob
import numpy as np

from .stereobase import StereoBaseDataset


class MADSDataset(StereoBaseDataset):
    def __init__(self, data_dir, image_set, cfg):
        super().__init__(image_set, cfg)
        if image_set == "train":
            self.data_path = os.path.join(data_dir, "train")
        elif image_set == "val":
            self.data_path = os.path.join(data_dir, "valid")
        else:
            raise ValueError("Invalid image set")
        self.flip_pairs = [[2, 6], [3, 7], [4, 8], [5, 9], [10, 14],
                           [11, 15], [12, 16], [13, 17]]
        self.db = self._get_db()

    def _transform(self, pose_3d, T):
        pose_3d = T @ np.vstack((pose_3d.T, np.ones((1, pose_3d.shape[0]))))

        return pose_3d.T

    def _project_3d_to_2d(self, pose_3d, P):
        pose_2d = P @ np.vstack((pose_3d.T, np.ones((1, pose_3d.shape[0]))))
        pose_2d = pose_2d.T[:, :3]
        pose_2d[:, :2] /= pose_2d[:, 2:]

        return pose_2d[:, :2]

    def _get_db(self):
        left_img_paths = sorted(glob.glob(
            os.path.join(self.data_path, "**/**/left/*.jpg")))
        right_img_paths = sorted(glob.glob(
            os.path.join(self.data_path, "**/**/right/*.jpg")))
        gt_pose_paths = sorted(glob.glob(
            os.path.join(self.data_path, "**/**/pose/*.json")))
        depth_map_paths = sorted(glob.glob(
            os.path.join(self.data_path, "**/**/depth_map/*.npy")))

        print(f"Found {len(left_img_paths)} left images")
        print(f"Found {len(right_img_paths)} right images")
        print(f"Found {len(gt_pose_paths)} ground truths")
        print(f"Found {len(depth_map_paths)} depth maps")

        assert len(left_img_paths) == len(right_img_paths) \
            == len(gt_pose_paths) == len(depth_map_paths), \
            "Number of images and ground truths must match"

        gt_db = []
        for i in range(len(right_img_paths)):
            with open(gt_pose_paths[i], 'r') as f:
                data = json.load(f)

                calibs_info = data['calibs_info']
                pose_3d = np.array(data['pose_3d'])

            # set the value of joints that have NaN values to 0
            mask = np.isnan(pose_3d)
            pose_3d[mask] = 0

            # set the visibility of joints that have NaN values to 0
            vis_left = np.bool_(~mask[:, :1])
            vis_right = np.bool_(~mask[:, :1])

            intrins_left = np.array(calibs_info['cam_left']['intrinsics'])
            intrins_right = np.array(calibs_info['cam_right']['intrinsics'])

            extrins_left = np.hstack(
                (calibs_info['cam_left']['rotation'],
                 calibs_info['cam_left']['translation'])
            )

            # transform 3D pose to the coordinate system of the left camera
            pose_3d = self._transform(pose_3d, extrins_left)

            baseline = 160

            # Since the image is already rectified, the extrinsic matrix of the
            # left camera is the identity matrix, and the extrinsic matrix
            # of the right camera has translation in the x-axis equal to the
            # baseline distance between the cameras.
            extrins_left = np.eye(4)[:3]
            extrins_right = np.eye(4)[:3]
            extrins_right[0, -1] = -baseline

            project_left = intrins_left @ extrins_left
            project_right = intrins_right @ extrins_right

            pose_left = self._project_3d_to_2d(pose_3d, project_left)
            pose_right = self._project_3d_to_2d(pose_3d, project_right)

            # get the center and the scale of the bounding box
            x1 = (pose_left[vis_left[:, 0], 0].min()
                  + pose_right[vis_right[:, 0], 0].min()) / 2
            x2 = (pose_left[vis_left[:, 0], 0].max()
                  + pose_right[vis_right[:, 0], 0].max()) / 2
            y1 = (pose_left[vis_left[:, 0], 1].min()
                  + pose_right[vis_right[:, 0], 1].min()) / 2
            y2 = (pose_left[vis_left[:, 0], 1].max()
                  + pose_right[vis_right[:, 0], 1].max()) / 2
            bbox_center = np.asarray([(x1 + x2) / 2, (y1 + y2) / 2])
            bbox_scale = np.array([x2 - x1, y2 - y1]) * 1.2

            # Only use the images from right camera for 2d human pose training
            gt_db.append({
                'image_left': left_img_paths[i],
                'image_right': right_img_paths[i],
                'depth_map': depth_map_paths[i],
                'pose_left': pose_left,
                'pose_right': pose_right,
                'pose_left_vis': vis_left,
                'pose_right_vis': vis_right,
                'bbox_center': bbox_center,
                'bbox_scale': bbox_scale,
                'pose_3d': pose_3d,
                'baseline': np.asarray([baseline]),
                'project_left': project_left,
                'project_right': project_right,
            })

        return gt_db
