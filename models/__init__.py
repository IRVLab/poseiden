from .mononet import build_mono_net
from .stereonet import build_stereo_net


def build_model(cfg):
    if cfg.model.name == "mono":
        model = build_mono_net(cfg)
    elif cfg.model.name == "stereo":
        model = build_stereo_net(cfg)
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}, "
                         "expected 'mono' or 'stereo'.")

    return model
