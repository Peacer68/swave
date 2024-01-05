"""
Generate more correct match with k-reflow-model to train (k+1)-reflow-model
"""
import os
import argparse
import json
import numpy as np
import re
import pickle

import torch
import torchaudio
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from model.warmup import GradualWarmupScheduler

from datetime import datetime
from tqdm import tqdm

from logger import Logger
from model import WaveGrad
from data import AudioDataset, MelSpectrogramFixed
from benchmark import compute_rtf
from utils import ConfigWrapper, show_message, str2bool

def batch_dump(y_0, y_1, y_0_pred, file_name):
    y_0, y_1, y_0_pred = y_0.detach().cpu(), y_1.detach().cpu(), y_0_pred.detach().cpu()
    with open(file_name, "wb") as f:
        pickle.dump({"y_0":y_0, "y_1":y_1, "y_0_pred":y_0_pred}, f)



with open("./configs/match.json") as f:
    config = ConfigWrapper(**json.load(f))

cuda = config.training_config.cuda
torch.cuda.set_device(f"cuda:{cuda}")

seed = config.training_config.seed+68*cuda
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

train_dataset = AudioDataset(config, training=True)
train_dataloader = DataLoader(
        train_dataset, batch_size=config.training_config.batch_size, drop_last=True
    )
teacher_model = WaveGrad(config)
teacher_model.load_state_dict(torch.load(config.training_config.teacher_ckpt_path, map_location=torch.device("cpu"))["model"])
teacher_model.cuda()
teacher_model.eval()

mel_fn = MelSpectrogramFixed(
    sample_rate=config.data_config.sample_rate,
    n_fft=config.data_config.n_fft,
    win_length=config.data_config.win_length,
    hop_length=config.data_config.hop_length,
    f_min=config.data_config.f_min,
    f_max=config.data_config.f_max,
    n_mels=config.data_config.n_mels,
    window_fn=torch.hann_window
).cuda()

re_comp = re.compile(r"(?<=checkpoint\_).*(?=\.pt)")
iter_num = re_comp.findall(config.training_config.teacher_ckpt_path)[0]
save_dir = os.path.join(config.data_config.save_dir, f"3_checkpoint_{iter_num}")
os.makedirs(save_dir, exist_ok=True)

method = config.training_config.method
if method == "model":
    method = f"model{config.training_config.test_noise_schedule.n_iter}"

with torch.no_grad():
    for epoch in range(config.training_config.n_epoch):
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch-{epoch}: Distilling")):
            y_0 = batch.cuda()
            mel = mel_fn(y_0)
            
            if method[:5] == "model":
                y_1, y_0_pred = teacher_model.forward_reflow(
                    mel, store_intermediate_states=False,return_noise=True
                )
            elif method == "ode":
                y_1, y_0_pred = teacher_model.maybe_optimize_forward(
                    mel, store_intermediate_states=False
                )
            else:
                raise NotImplementedError(f"{method} is not implemented!")

            file_name = os.path.join(save_dir, f"{method}_seed{seed}_epoch{epoch}_idx{i}_cuda{cuda}.pkl")
            batch_dump(y_0, y_1, y_0_pred, file_name)
        


