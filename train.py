import os
import torch
import hydra
from omegaconf import OmegaConf, DictConfig
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from datasets import build_dataset
from models import build_model
from engine import build_module

torch.set_float32_matmul_precision("high")
# sets seeds for numpy, torch and python.random.
seed_everything(42, workers=True)


@hydra.main(config_path="conf", config_name="", version_base="1.3")
def run(cfg: DictConfig):
    # This will throw an error if any required fields (marked with ???)
    # are missing
    OmegaConf.to_container(cfg, throw_on_missing=True)
    assert cfg.image_size[0] == cfg.image_size[1], "Image size must be square"

    model_name = (
        f"{cfg.name}_{cfg.model.name}_{cfg.model.backbone}"
        f"_{cfg.dataset.name}_{cfg.image_size[0]}x{cfg.image_size[1]}")
    save_path = os.path.join(cfg.path.output_dir, model_name)

    train_loader = build_dataset(cfg, "train")
    val_loader = build_dataset(cfg, "val")
    model = build_model(cfg)

    pretrained = cfg.model.get("pretrained", "")
    if len(pretrained) > 0:
        model.init_weights(pretrained)
        print(f"Loaded pretrained weights from {pretrained}")

    module = build_module(cfg, model, save_path)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(save_path, "weight"),
        filename="best",
        monitor='val/total_loss',
        mode='min',
        save_top_k=1,
        save_last=True)
    callbacks = [lr_monitor, ckpt_cb]

    logger = TensorBoardLogger(
        save_dir=cfg.path.log_dir,
        name=model_name)

    trainer = Trainer(accelerator='gpu',
                      devices=[cfg.gpu],
                      precision=32,
                      max_epochs=cfg.num_epochs,
                      deterministic=True,
                      num_sanity_val_steps=1,
                      logger=logger,
                      callbacks=callbacks)

    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    run()
