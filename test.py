import cv2
import os
import torch
import glob
import subprocess
import numpy as np
import matplotlib
from pathlib import Path
from lightning.pytorch import Trainer
import hydra
from omegaconf import OmegaConf, DictConfig

from datasets import build_dataset
from models import build_model
from engine import build_module
from tools.utils import plot_pose_2d, plot_pose_3d, plot_error
from tools.metrics import mpjpe

torch.set_float32_matmul_precision("high")
matplotlib.use("Agg")


def unnorm_image(image):
    image[:, 0] = image[:, 0] * 0.229 + 0.485
    image[:, 1] = image[:, 1] * 0.224 + 0.456
    image[:, 2] = image[:, 2] * 0.225 + 0.406
    image = (image * 255.0).numpy().transpose(0, 2, 3, 1)
    return image.astype(np.uint8)


def plot(img_left, img_right, target_left, target_right,
         pred_left, pred_right, target_3d, pred_3d, visibility):

    img_2d = plot_pose_2d((target_left, target_right),
                          (pred_left, pred_right),
                          (img_left.copy(), img_right.copy()),
                          visibility)
    img_2d = cv2.cvtColor(img_2d, cv2.COLOR_BGR2RGB)

    img_3d = plot_pose_3d(target_3d, pred_3d, visibility=visibility)

    return img_2d, img_3d


@hydra.main(config_path="conf", config_name="", version_base="1.3")
def run(cfg: DictConfig):
    # This will throw an error if any required fields (marked with ???)
    # are missing
    OmegaConf.to_container(cfg, throw_on_missing=True)
    assert cfg.image_size[0] == cfg.image_size[1], "Image size must be square"

    val_loader = build_dataset(cfg, "val")
    model = build_model(cfg)

    # Load model weight
    model_weight = cfg.get("model_weight")
    state_dict = torch.load(model_weight)['state_dict']
    new_state_dict = {
        k.replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)

    module = build_module(cfg, model, "")
    module.eval()

    output_dir = os.path.join(Path(model_weight).parents[1], "result")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    trainer = Trainer(accelerator="gpu",
                      devices=[cfg.device],
                      precision=32,
                      logger=False)
    outputs = trainer.predict(module, val_loader)

    errors = []
    count = 0

    for preds_2d, preds_3d, batchs in outputs:
        imagesL, imagesR, meta = batchs

        imagesL = unnorm_image(imagesL.clone())
        imagesR = unnorm_image(imagesR.clone())
        preds_2d = [preds_2d[0].numpy(), preds_2d[1].numpy()]
        preds_3d = preds_3d.numpy()
        targets_2dL = meta["pose_left"].numpy()
        targets_2dR = meta["pose_right"].numpy()
        targets_3d = meta["pose_3d"].numpy()
        visibilities = meta["joints_vis"].numpy().astype(np.int32)

        for i in range(len(preds_3d)):
            error = mpjpe(preds_3d[i], targets_3d[i], visibilities[i])
            errors.append(error)

            if cfg.visualize:
                img_2d, img_3d = plot(imagesL[i], imagesR[i],
                                      targets_2dL[i], targets_2dR[i],
                                      preds_2d[0][i], preds_2d[1][i],
                                      targets_3d[i], preds_3d[i],
                                      visibilities[i])

                error_plot = plot_error(
                    errors, len(val_loader) * cfg.batch_size)

                pose_img = np.hstack(
                    (np.vstack((img_2d, error_plot)), img_3d))
                pose_img = cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR)

                cv2.imwrite(
                    os.path.join(output_dir, f"{count:05d}.jpg"), pose_img)
                cv2.imshow("Pose Estimation", pose_img)
                cv2.waitKey(1)

            count += 1

    if cfg.visualize:
        cv2.destroyAllWindows()
        os.chdir(output_dir)
        subprocess.call([
            'ffmpeg', '-framerate', '20', '-i', '%05d.jpg', '-r', '30',
            '-pix_fmt', 'yuv420p',
            '-c:v', 'libx264',
            '-b:v', '3000k',
            '-crf', '28',
            'result.mp4'
        ])
        # remove result images
        for file_name in glob.glob("*.jpg"):
            os.remove(file_name)

        os.chdir("../")

    print(f"Mean MPJPE: {np.mean(errors)}")


if __name__ == "__main__":
    run()
