import os
import torch
from lightning.pytorch import LightningModule

from tools.loss import JointsMSELoss
from tools.metrics import accuracy
from tools.vis import save_debug_images, save_mono_attention_map


class MonoNetModule(LightningModule):
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
        self.criterion = JointsMSELoss(use_target_weight=True)

        self.save_hyperparameters(ignore=['model'])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), self.lr)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, self.lr_step, self.lr_factor)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, batch, batch_idx):
        image, target, target_weight, _ = batch

        output, attn = self.model(image)

        loss = self.criterion(output, target, target_weight)
        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())

        return loss, avg_acc, cnt, output, pred, attn

    def training_step(self, batch, batch_idx):
        loss, avg_acc, cnt, output, pred, attn = self.forward(batch, batch_idx)
        self.train_count += cnt
        self.train_total_acc += avg_acc * cnt

        self.log_dict(
            {
                'train/total_loss': loss,
                'train/acc': self.train_total_acc / self.train_count
            },
            logger=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size)

        return {"loss": loss, "output": output, "pred": pred, "attn": attn}

    def validation_step(self, batch, batch_idx):
        loss, avg_acc, cnt, output, pred, attn = self.forward(batch, batch_idx)
        self.val_count += cnt
        self.val_total_acc += avg_acc * cnt

        self.log_dict(
            {
                'val/total_loss': loss,
                'val/acc': self.val_total_acc / self.val_count
            },
            logger=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size)

        return {"loss": loss, "output": output, "pred": pred, "attn": attn}

    def on_train_epoch_start(self):
        self.train_count = 0
        self.train_total_acc = 0

    def on_validation_epoch_start(self):
        self.val_count = 0
        self.val_total_acc = 0

    def on_train_batch_end(self, out, batch, batch_idx):
        if batch_idx % 100 == 0:
            img, target, _, meta = batch

            output, pred = out["output"], out["pred"]
            prefix = '{}_{}'.format(
                os.path.join(self.output_dir, 'train'), batch_idx)
            save_debug_images(img, meta, target, pred*4, output, prefix)

    def on_validation_batch_end(self, out, batch, batch_idx):
        if batch_idx % 100 == 0:
            img, target, target_weight, meta = batch

            output, pred, attn = out["output"], out["pred"], out["attn"]
            prefix = '{}_{}'.format(
                os.path.join(self.output_dir, 'val'), batch_idx)

            save_debug_images(img, meta, target, pred*4, output, prefix)
            save_mono_attention_map(
                img, pred*4, attn, '{}_attn_map.jpg'.format(prefix))
