import torch

from .mono.coco import COCODataset
from .stereo.mads import MADSDataset
from .stereo.diver import DiverDataset
from .stereo.stereobj import StereoObjDataset


def build_dataset(cfg, image_set):
    batch_size = cfg.get("batch_size")
    num_workers = cfg.get("num_workers")

    data_dir = cfg.dataset.get('root')

    dataset = {
        "coco": COCODataset,
        "mads": MADSDataset,
        "diver": DiverDataset,
        "stereobj": StereoObjDataset
    }

    try:
        dataset_class = dataset[cfg.dataset.get('name')]
    except KeyError:
        raise NotImplementedError(
            f"Dataset {cfg.dataset.get('name')} not implemented")

    dataset = dataset_class(data_dir, image_set, cfg)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(image_set == "train"),
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader
