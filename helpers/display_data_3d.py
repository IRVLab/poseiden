import tqdm
import cv2
import numpy as np
import hydra
from omegaconf import DictConfig
import sys
sys.path.append("./")

from datasets import build_dataset  # noqa
from tools.utils import plot_joints, plot_pose_3d  # noqa
from tools.common import triangulation, reproject, project  # noqa


def unnorm_image(image):
    image[:, 0] = image[:, 0] * 0.229 + 0.485
    image[:, 1] = image[:, 1] * 0.224 + 0.456
    image[:, 2] = image[:, 2] * 0.225 + 0.406
    image = (image * 255.0).numpy().transpose(0, 2, 3, 1)
    return image.astype(np.uint8)


@hydra.main(config_path="../conf", config_name="", version_base="1.3")
def run(cfg: DictConfig):

    train_loader = build_dataset(cfg, "train")
    disp_list = []

    for i, (image_left, image_right, meta) \
            in enumerate(tqdm.tqdm(train_loader)):

        image_left = unnorm_image(image_left)
        image_right = unnorm_image(image_right)

        target_left = meta['pose_left'].numpy()
        target_right = meta['pose_right'].numpy()

        P_left = meta["project_left"].numpy()
        P_right = meta["project_right"].numpy()

        target_3d = meta.get("pose_3d", None)
        if target_3d is not None:
            target_3d = target_3d.numpy()

        baseline = meta['baseline'].numpy()
        Q = meta['Q'].numpy()

        joints_vis = meta["joints_vis"].numpy()

        for j in range(image_left.shape[0]):
            display_left = np.ascontiguousarray(image_left[j])
            display_right = np.ascontiguousarray(image_right[j])

            reproject_3d = triangulation(
                P_left[j], P_right[j], target_left[j], target_right[j])

            vis = joints_vis[j].astype(np.int32)

            t_left = target_left[j]
            t_right = target_right[j]

            disp = t_left[:, :1] - t_right[:, :1]

            for d, v in zip(disp, vis):
                if v > 0:
                    disp_list.append(d)

            kps_3d = reproject(
                t_left[None, :, :2],
                disp[None, :], Q[j:j + 1, :])
            kps_3d = kps_3d.reshape(len(t_left), 3)

            T = np.eye(4)
            T[0, 3] = -baseline[j]

            reproject_2d = project(kps_3d, P_right[j])

            plot_joints(display_left, t_left, vis)
            plot_joints(display_right, t_right, vis)
            plot_joints(display_right, reproject_2d, vis, c=(255, 255, 0))

            display = np.concatenate((display_left, display_right), axis=1)
            cv2.imshow("img", display)

            if target_3d is not None:
                img_3d = plot_pose_3d(
                    target_3d[j], reproject_3d, visibility=vis)
            else:
                img_3d = plot_pose_3d(
                    kps_3d, reproject_3d, visibility=vis,
                    xlim=[-1500, 1500],
                    ylim=[2000, 8000],
                    zlim=[-1500, 1500],
                    diver=True)
            img_3d = cv2.cvtColor(img_3d, cv2.COLOR_RGB2BGR)
            cv2.imshow("img_3d", img_3d)

            key = cv2.waitKey(0)
            if key == ord('q'):
                print("quit display")
                exit(1)

    d = np.stack(disp_list)
    print(np.mean(d), np.percentile(d, 5), np.percentile(d, 95),
          np.amax(d), np.amin(d))


if __name__ == "__main__":
    run()
