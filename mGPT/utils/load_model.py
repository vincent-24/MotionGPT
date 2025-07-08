from pathlib import Path
import torch
import pytorch_lightning as pl
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model

def load_model(cfg):
    cfg.FOLDER = 'cache'
    output_dir = Path(cfg.FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(cfg.SEED_VALUE)
    if cfg.ACCELERATOR == "gpu":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    datamodule = build_data(cfg, phase="test")
    model = build_model(cfg, datamodule)
    state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)

    return cfg, model, device