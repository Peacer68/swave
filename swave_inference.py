import json
import numpy as np
import os

import torchaudio
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from datetime import datetime
from tqdm import tqdm

from logger import Logger
from model import WaveGrad
from data import AudioDataset, MelSpectrogramFixed, load_audio_to_torch
from benchmark import compute_rtf
from utils import ConfigWrapper, show_message, str2bool
import torch.nn.functional as F

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.cuda.set_device("cuda:7")

with open("./configs/inference.json") as f:
    config = ConfigWrapper(**json.load(f))

if "wav_save_dir" not in config.training_config or config.training_config["wav_save_dir"]=="":
    # config.training_config["wav_save_dir"] = "./reflow_gen_wav"
    config.training_config["wav_save_dir"] = "-".join(config.resume_checkpoint.split("/")[-2:]).replace(".pt","")
os.makedirs(config.training_config.wav_save_dir, exist_ok=True)
show_message("Loading data loaders finished!")

test_dataset = AudioDataset(config, training=False)
test_dataloader = DataLoader(test_dataset, batch_size=1)
test_batch = test_dataset.sample_test_batch(
    config.training_config.n_samples_to_test
)
show_message("Loading test batch finished!")

model = WaveGrad(config)
# model.load_state_dict(torch.load(config.resume_checkpoint, map_location=torch.device("cpu"))["epoch"].state_dict())
model.load_state_dict(torch.load(config.resume_checkpoint, map_location=torch.device("cpu"))["model"])
model = model.eval()
model = model.cuda()

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

with torch.no_grad():
    tol_straightness = []
    test_mse_losses = []
    test_mse_spec_losses = []
    average_rtfs = []

    for i in tqdm(range(1)): # repeat 1 times
        # Restore random batch from test dataset
        audios = {}
        specs = {}
        test_mse_loss = 0
        test_mse_spec_loss = 0
        average_rtf = 0
        single_straightness = []
        for index, test_sample in enumerate(test_batch):
            # test_sample, sr = load_audio_to_torch(audio_path="./data/LJSpeech-1.1/wavs/LJ009-0121.wav")
            test_sample = test_sample.unsqueeze(0).cuda()
            test_mel = mel_fn(test_sample)

            # print(test_mel.shape)

            start = datetime.now()
            y_0_hat, straightness = model.forward_reflow(
                test_mel, store_intermediate_states=False, test_sample=test_sample, return_straightness=True
            )
            print("straightness", straightness)
            end = datetime.now()
            # y_0_hat, test_sample, test_mel = y_0_hat[:,60000:], test_sample[:,60000:], mel_fn(test_sample[:,60000:])
            single_straightness.append(straightness)
            y_0_hat_mel = mel_fn(y_0_hat)
            generation_time = (end - start).total_seconds()
            average_rtf += compute_rtf(
                y_0_hat, generation_time, config.data_config.sample_rate
            )
            
            single_test_mse_loss, single_test_mse_spec_loss = torch.nn.MSELoss()(y_0_hat, test_sample).item(), \
            torch.nn.MSELoss()(y_0_hat_mel, test_mel).item()
            test_mse_loss += single_test_mse_loss
            test_mse_spec_loss += single_test_mse_spec_loss
            print("wav mse loss", single_test_mse_loss, "spec mse loss: ", single_test_mse_spec_loss)

            audios[f'audio_{index}/predicted'] = y_0_hat.cpu()
            specs[f'mel_{index}/predicted'] = y_0_hat_mel.cpu().squeeze()
            audio_save_path = f"{config.training_config.wav_save_dir}/audio_{index}_step{config.training_config.test_noise_schedule.n_iter}.wav"
            spec_save_path = f"{config.training_config.wav_save_dir}/spec_{index}.png"
            
            if i == 0:
                torchaudio.save(
                    audio_save_path, audios[f'audio_{index}/predicted'], sample_rate=config.data_config.sample_rate
                    )
                torchaudio.save(
                    f"{config.training_config.wav_save_dir}/audio_{index}_gt.wav", test_sample.cpu(), sample_rate=config.data_config.sample_rate
                    )
            
            fig, ax = plt.subplots(figsize=(12,6))
            im = ax.imshow(specs[f'mel_{index}/predicted'].flip(dims=(0,)))
            # im = ax.imshow(test_mel.squeeze(0).cpu().flip(dims=(0,)))
            fig.colorbar(im, ax=ax)
            fig.savefig(spec_save_path)
            plt.close()
        test_mse_loss /= len(test_batch)
        test_mse_spec_loss /= len(test_batch)
        average_rtf /= len(test_batch)

        test_mse_losses.append(test_mse_loss)
        test_mse_spec_losses.append(test_mse_spec_loss)
        average_rtfs.append(average_rtf)
        tol_straightness.append(single_straightness)

        print(test_mse_loss, test_mse_spec_loss)
        show_message(f'Device: GPU. average_rtf={average_rtf}', verbose=True)

    with open(f"{config.training_config.wav_save_dir}/statistics.json", "w") as f:
        stat = {
            "test_mse_losses": test_mse_losses,
            "test_mse_spec_losses": test_mse_spec_losses,
            "average_rtfs": average_rtfs,
            "tol_straightness": tol_straightness,
            "ckpt":config.resume_checkpoint
        }
        json.dump(stat, f)

