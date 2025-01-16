import os
import cv2
import glob
from pathlib import Path
import numpy as np
import torchvision.transforms as transforms
import torch
import hydra
from omegaconf import OmegaConf, DictConfig

from tools.utils import (
    to_numpy, plot_joints, plot_pose_3d, analyze_accuracy,
    compute_diver_body_frame
)
from tools.common import (
    reproject_tensor, get_warp_matrix, fix_aspect_ratio, gen_reproj_matrix
)
from tools.rectify import Rectificator
from tools.yolo import YOLO
from models import build_model


def save_to_video(frames, save_path, filename):
    assert len(frames) > 0, "No frames to save"

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        os.path.join(save_path, f'{filename}.mp4'), fourcc, 10.0,
        (frames[0].shape[1], frames[0].shape[0]))

    # Write each frame to the video file
    for frame in frames:
        out.write(frame)

    print(f"Video saved to {os.path.join(save_path, f'{filename}.mp4')}")

    # Release the video writer and close the video file
    out.release()


def plot(img_left, img_right, kps_2d, kps_3d, axis=None):
    img_left = plot_joints(img_left, kps_2d[0])
    img_right = plot_joints(img_right, kps_2d[1])

    img_3d = plot_pose_3d(None, kps_3d,
                          axis=axis,
                          xlim=[-1000, 1000],
                          ylim=[0, 8000],
                          zlim=[0, 1500],
                          diver=True)

    # Display the image with keypoints and depth value
    size = img_3d.shape[0]
    img_left = cv2.resize(img_left, (size, size))
    img_right = cv2.resize(img_right, (size, size))

    img_3d = cv2.cvtColor(img_3d, cv2.COLOR_RGB2BGR)
    img = np.hstack((img_left, img_right, img_3d))

    return img


class Inferencer:
    def __init__(self, model, device, image_size, yolo_weight=""):
        self.device = torch.device(
            'cuda:{}'.format(device) if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.image_size = image_size

        self.yolo = None
        if len(yolo_weight):
            self.yolo = YOLO(yolo_weight)
            print(f"Weight found at: {yolo_weight}. YOLO model loaded.")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def preprocess(self, img_left, img_right, PL, PR, baseline):
        """
        Preprocess the input images and camera parameters.

        The images are cropped and resized to match the model input size based
        on the 2D bounding box. If the 2D bounding box is unavailable, the
        images are cropped from the center.

        Args:
            img_left (np.ndarray): Left image.
            img_right (np.ndarray): Right image.
            PL (np.ndarray): Projection matrix of the left camera.
            PR (np.ndarray): Projection matrix of the right camera.
            baseline (float): Baseline distance between the two cameras.

        Returns:
            img_left (np.ndarray): Preprocessed left image.
            img_right (np.ndarray): Preprocessed right image.
            Q (np.ndarray): Disparity-to-depth mapping matrix.
        """

        h, w = img_left.shape[:2]
        center = np.array([w / 2, h / 2])
        scale = np.array([min(h, w), min(h, w)])

        if self.yolo is not None:
            det_bbox_left, _ = self.yolo.inference(img_left)
            det_bbox_right, _ = self.yolo.inference(img_right)

            if len(det_bbox_left) > 0 and len(det_bbox_right) > 0:
                x1L, y1L, x2L, y2L = det_bbox_left[0]
                x1R, y1R, x2R, y2R = det_bbox_right[0]

                x1, y1 = (x1L + x1R) / 2, (y1L + y1R) / 2
                x2, y2 = (x2L + x2R) / 2, (y2L + y2R) / 2

                center = np.asarray([(x1 + x2) / 2, (y1 + y2) / 2])
                scale = np.array([x2 - x1, y2 - y1]) * 1.5

        scale = fix_aspect_ratio(scale, aspect_ratio=1)
        shift = [0, 0]

        trans = get_warp_matrix(center, scale, 0, self.image_size, shift=shift)

        # crop and resize images to match the model input size
        img_left = cv2.warpAffine(
            img_left,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        img_right = cv2.warpAffine(
            img_right,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        T = np.eye(3)
        T[:2, :] = trans

        PL = np.vstack((T @ PL, np.array([0, 0, 0, 1])))
        PR = np.vstack((T @ PR, np.array([0, 0, 0, 1])))

        # Create a disparity-to-depth mapping matrix.
        # Identical to the Q matrix from cv2.stereoRectify()
        Q = gen_reproj_matrix(PL, PR, baseline)

        return img_left, img_right, Q

    def estimate(self, img_left, img_right, PL, PR, baseline):
        """Estimate the 3D keypoints of the diver.

        Args:
            img_left (np.ndarray): Left image.
            img_right (np.ndarray): Right image.
            PL (np.ndarray): Projection matrix of the left camera.
            PR (np.ndarray): Projection matrix of the right camera.
            baseline (float): Baseline distance between the two cameras.

        Returns:
            kps_2d (list): 2D keypoints of the diver.
            kps_3d (np.ndarray): 3D keypoints of the diver.
            axis (list): Coordinate of the diver's body frame.
        """

        img_left, img_right, Q = \
            self.preprocess(img_left, img_right, PL, PR, baseline)

        img_left_tensor = self.transform(img_left.copy()).unsqueeze(0)
        img_left_tensor = img_left_tensor.to(self.device)

        img_right_tensor = self.transform(img_right.copy()).unsqueeze(0)
        img_right_tensor = img_right_tensor.to(self.device)

        Q = torch.tensor(Q, dtype=torch.float32).unsqueeze(0).to(self.device)

        kps_2d, disp, _ = self.model(img_left_tensor, img_right_tensor)
        kps_3d = reproject_tensor(kps_2d[0], disp, Q)

        # Convert kps_2d to numpy array
        for i in range(2):
            kps_2d[i] = to_numpy(kps_2d[i].squeeze(0))
        kps_3d = to_numpy(kps_3d.squeeze(0))
        c, x, y, z = compute_diver_body_frame(kps_3d)

        img = plot(img_left, img_right, kps_2d, kps_3d, [c, x, y, z])
        cv2.imshow("img", img)
        cv2.waitKey(1)

        return kps_2d, kps_3d, [c, x, y, z]


class Validator:
    def __init__(self, calib_file, inferencer):
        cam_left, cam_right = Rectificator.parse_calibration_data(
            os.path.join(calib_file))
        self.rectify = Rectificator(cam_left, cam_right)
        self.cam_params = self.rectify.get_cam_params()

        self.inferencer = inferencer

    def get_image_path(self, img_folder):
        left_img_paths = sorted(
            glob.glob(os.path.join(img_folder, "*_left.png")))
        right_img_paths = sorted(
            glob.glob(os.path.join(img_folder, "*_right.png")))

        assert len(left_img_paths) == len(right_img_paths), \
            "Number of images must match, left: {}, right: {}".format(
                len(left_img_paths), len(right_img_paths))

        metadata = []
        for i in range(len(left_img_paths)):
            metadata.append({
                'left_img_path': left_img_paths[i],
                'right_img_path': right_img_paths[i],
            })

        return metadata

    def map_depth(self, c):
        depth = c[2] / 1000

        if 2 <= depth < 3:
            pred = 2
        elif 3 <= depth < 4:
            pred = 3
        elif 4 <= depth < 5:
            pred = 4
        elif 5 <= depth < 6:
            pred = 5
        elif 6 <= depth < 7:
            pred = 6
        elif 7 <= depth < 8:
            pred = 7
        else:
            pred = -1

        return pred

    def map_orientation(self, z):
        degree = np.arctan2(z[2], z[0]) * 180 / np.pi

        if -112.5 <= degree < -67.5:
            pred = 1
        elif -157.5 <= degree < -112.5:
            pred = 2
        elif 157.5 <= degree or degree < -157.5:
            pred = 3
        elif 112.5 <= degree < 157.5:
            pred = 4
        elif 67.5 <= degree < 112.5:
            pred = 5
        elif 22.5 <= degree < 67.5:
            pred = 6
        elif -22.5 <= degree < 22.5:
            pred = 7
        elif -67.5 <= degree < -22.5:
            pred = 8

        return pred

    def map_pose(self, pose3d, target):
        body_depth = pose3d[[0, 1, 6, 7], 2].mean()

        if "arm_left" in target:
            if pose3d[4, 2] > body_depth:
                pred = "arm_left_back"
            else:
                pred = "arm_left_front"
        elif "arm_right" in target:
            if pose3d[5, 2] > body_depth:
                pred = "arm_right_back"
            else:
                pred = "arm_right_front"
        elif "leg_left" in target:
            if pose3d[10, 2] > body_depth:
                pred = "leg_left_back"
            else:
                pred = "leg_left_front"
        elif "leg_right" in target:
            if pose3d[11, 2] > body_depth:
                pred = "leg_right_back"
            else:
                pred = "leg_right_front"
        else:
            raise ValueError("Invalid target")

        return pred

    def val(self, root_path, test_set, output_dir):
        images_folder = glob.glob(os.path.join(root_path, test_set, "**"))
        targ_list, pred_list = [], []

        for folder in images_folder:
            metadata = self.get_image_path(folder)

            for i, meta in enumerate(metadata):
                img_left = cv2.imread(meta['left_img_path'])
                img_right = cv2.imread(meta['right_img_path'])

                img_left = self.rectify.rectify_images("left", img_left)
                img_right = self.rectify.rectify_images("right", img_right)

                P_left = np.array(self.cam_params['P1'])
                P_right = np.array(self.cam_params['P2'])
                baseline = np.array([self.cam_params['baseline']])

                kps_2d, kps_3d, axis = self.inferencer.estimate(
                    img_left, img_right, P_left, P_right, baseline)

                if test_set == "depth":
                    target = int(Path(folder).stem)
                    pred = self.map_depth(axis[0])
                elif test_set == "orientation":
                    target = int(Path(folder).stem)
                    pred = self.map_orientation(axis[3])
                elif test_set == "pose":
                    target = Path(folder).stem
                    pred = self.map_pose(kps_3d, target)
                else:
                    raise ValueError("Invalid test set")

                targ_list.append(target)
                pred_list.append(pred)

        precision, recall, conf_m = analyze_accuracy(targ_list, pred_list)
        conf_m = cv2.cvtColor(conf_m, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"{test_set}.jpg"), conf_m)
        print("Absolute Result-> "
              "precision: {:.4f}, recall: {:.4f}".format(precision, recall))


@hydra.main(config_path="conf", config_name="", version_base="1.3")
def run(cfg: DictConfig):
    print(cfg)

    # This will throw an error if any required fields (marked with ???)
    # are missing
    OmegaConf.to_container(cfg, throw_on_missing=True)
    assert cfg.image_size[0] == cfg.image_size[1], "Image size must be square"

    model = build_model(cfg)
    # Load model weight
    model_weight = cfg.get("model_weight")
    state_dict = torch.load(model_weight)['state_dict']
    new_state_dict = {
        k.replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)

    inferencer = Inferencer(
        model=model.to(cfg.device),
        device=cfg.device,
        image_size=cfg.image_size,
        yolo_weight=cfg.yolo_weight
    )
    validator = Validator(
        calib_file=os.path.join(cfg.data_path, "calibs.yaml"),
        inferencer=inferencer
    )

    output_dir = os.path.join(cfg.output_dir, Path(cfg.model_weight).parts[1])
    os.makedirs(output_dir, exist_ok=True)
    validator.val(
        root_path=cfg.data_path,
        test_set="depth",
        output_dir=output_dir)
    validator.val(
        root_path=cfg.data_path,
        test_set="orientation",
        output_dir=output_dir)
    validator.val(
        root_path=cfg.data_path,
        test_set="pose",
        output_dir=output_dir)


if __name__ == "__main__":
    run()
