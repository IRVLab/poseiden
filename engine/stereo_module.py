import os
import torch
from lightning.pytorch import LightningModule

from tools.common import reproject_tensor
from tools.loss import JointsL1Loss
from tools.metrics import mpjpe
from tools.vis import save_stereo_attention_map


class StereoNetModule(LightningModule):
    def __init__(self, cfg, model, output_dir=""):
        super().__init__()

        self.batch_size = cfg.get("batch_size")
        self.lr = cfg.get("learning_rate")
        if cfg.get("scheduler") is not None:
            s = cfg.get("scheduler")
            self.lr_step = s["milestones"]
            self.lr_factor = s["gamma"]

        if len(output_dir) > 0:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)

        self.model = model
        self.criterion = JointsL1Loss(use_target_weight=True)

        self.save_hyperparameters(ignore=['model'])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), self.lr)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, self.lr_step, self.lr_factor)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, batch, batch_idx):
        image_left, image_right, meta = batch

        target_left = meta['pose_left']
        target_right = meta['pose_right']
        target_weight = meta["joints_vis"]

        kps_2d, disp, attn = self.model(image_left, image_right)
        kps_3d = reproject_tensor(kps_2d[0], disp, meta['Q'])

        keypoints_loss = \
            self.criterion(kps_2d[0], target_left, target_weight) \
            + self.criterion(kps_2d[1], target_right, target_weight)

        loss = {"total_loss": keypoints_loss}

        # calculate error
        target_3d = meta.get("pose_3d", None)
        if target_3d is not None:
            error = mpjpe(kps_3d.detach().cpu(),
                          target_3d.detach().cpu(),
                          target_weight.detach().cpu())
        else:
            error = torch.zeros(1, device=self.device)

        return loss, error, kps_2d, kps_3d, attn

    def training_step(self, batch, batch_idx):
        loss, error, kps_2d, _, attn = self.forward(batch, batch_idx)

        log = {}
        for key, value in loss.items():
            log[f"train/{key}"] = value
        log.update({"train/mpjpe": error})

        self.log_dict(
            log,
            logger=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size)

        return {"loss": loss["total_loss"], "pred": kps_2d, "attn": attn}

    def validation_step(self, batch, batch_idx):
        loss, error, kps_2d, _, attn = self.forward(batch, batch_idx)

        log = {}
        for key, value in loss.items():
            log[f"val/{key}"] = value
        log.update({"val/mpjpe": error})

        self.log_dict(
            log,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size)

        return {"loss": loss["total_loss"], "pred": kps_2d, "attn": attn}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        _, _, kps_2d, kps_3d, _ = self.forward(batch, batch_idx)
        return kps_2d, kps_3d, batch

    def on_validation_batch_end(self, out, batch, batch_idx):
        if batch_idx % 100 == 0:
            image_left, image_right, meta = batch

            pred, attn = out["pred"], out["attn"]
            prefix = '{}_{}'.format(
                os.path.join(self.output_dir, 'val'), batch_idx)
            save_stereo_attention_map(
                image_left, image_right, pred[0], pred[1], attn,
                '{}_attn_map.jpg'.format(prefix)
            )
