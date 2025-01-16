from .mono_module import MonoNetModule
from .stereo_module import StereoNetModule


def build_module(cfg, model, output_dir):
    if cfg.model.name == "mono":
        module = MonoNetModule(cfg, model, output_dir)
    elif cfg.model.name == "stereo":
        module = StereoNetModule(cfg, model, output_dir)
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}, "
                         "expected 'mono' or 'stereo.")

    return module
