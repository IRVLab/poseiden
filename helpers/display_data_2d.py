import tqdm
import cv2
import numpy as np
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
import sys
sys.path.append("./")

from datasets import build_dataset  # noqa
from tools.utils import plot_joints  # noqa


@hydra.main(config_path="../conf", config_name="", version_base="1.3")
def run(cfg: DictConfig):

    train_loader = build_dataset(cfg, "train")

    for i, (input, targets, target_weights, meta) \
            in enumerate(tqdm.tqdm(train_loader)):
        input[:, 0] = input[:, 0] * 0.229 + 0.485
        input[:, 1] = input[:, 1] * 0.224 + 0.456
        input[:, 2] = input[:, 2] * 0.225 + 0.406
        input = input * 255.0

        targets = F.interpolate(
            targets, scale_factor=4, mode='bilinear', align_corners=True)
        joints = meta['joints']

        for j in range(input.shape[0]):
            img = input[j].numpy().transpose(1, 2, 0).astype(np.uint8)
            joint = joints[j].numpy()
            target = targets[j].numpy()
            target_weight = target_weights[j].numpy()

            img = img.copy()

            img = plot_joints(img, joint, target_weight)

            for i in range(target.shape[0]):
                heatmap = target[i, :, :]

                heatmap = cv2.normalize(
                    heatmap, None, alpha=0, beta=255,
                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                display = img * 0.8 + heatmap * 0.2
                cv2.imshow("img", display.astype(np.uint8))
                key = cv2.waitKey(0)
                if key == ord('q'):
                    print("quit display")
                    exit(1)


if __name__ == "__main__":
    run()
