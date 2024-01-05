import sys, os
sys.path.append('..')

import pickle
import numpy as np
import torch.nn.functional as F
import torch

from model.base import BaseModule
from model.nn import WaveGradNN
from tqdm import tqdm

from reflow_utils.step_sample import create_named_schedule_sampler
from scipy.integrate import solve_ivp

from model.utils import from_flattened_numpy, to_flattened_numpy

from .mul_stft_loss import MultiResolutionSTFTLoss

from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation



class WaveGrad(BaseModule):
    """
    WaveGrad diffusion process as described in WaveGrad paper
    (link: https://arxiv.org/pdf/2009.00713.pdf).
    Implementation adopted from `Denoising Diffusion Probabilistic Models`
    repository (link: https://github.com/hojonathanho/diffusion,
    paper: https://arxiv.org/pdf/2006.11239.pdf).
    """
    def __init__(self, config):
        super(WaveGrad, self).__init__()
        # Setup noise schedule
        # self.noise_schedule_is_set = False
        self.noise_schedule_is_set = True # In Reflow Process, this is always True

        if "mstft" in config.training_config and config.training_config.mstft:
            self.mstft = True
        else:
            self.mstft = False
        
        self.iters = config.training_config.training_noise_schedule.n_iter # for compute_loss
        self.gen_steps = config.training_config.test_noise_schedule.n_iter # for forward
        # Backbone neural network to model noise
        self.total_factor = np.product(config.model_config.factors)
        assert self.total_factor == config.data_config.hop_length, \
            """Total factor-product should be equal to the hop length of STFT."""
        self.nn = WaveGradNN(config)
        self.t_ignore = config.training_config.t_ignore

        # Setup step sampler for training
        if "step_sampler" not in config.training_config or config.training_config.step_sampler:
            self.step_sampler = create_named_schedule_sampler("lossaware", self.iters)
        else:
            self.step_sampler = None
        
        if "phase_loss_weight" in config.training_config:
            self.phase_loss_weight = config.training_config.phase_loss_weight
        else:
            self.phase_loss_weight = 0

        if "mag_loss_weight" in config.training_config:
            self.mag_loss_weight = config.training_config.mag_loss_weight
        else:
            self.mag_loss_weight = 0

        # if "one_step" in config.training_config:
        #     self.one_step = config.training_config.one_step
        # else:
        #     self.one_step = False
        if "distill_step" in config.training_config:
            self.distill_step = config.training_config.distill_step
        else:
            self.distill_step = self.iters
        


    def set_new_noise_schedule(
        self,
        init=torch.linspace,
        init_kwargs={'steps': 50, 'start': 1e-6, 'end': 1e-2}
    ):
        """
        Sets sampling noise schedule. Authors in the paper showed
        that WaveGrad supports variable noise schedules during inference.
        Thanks to the continuous noise level conditioning.
        :param init (callable function, optional): function which initializes betas
        :param init_kwargs (dict, optional): dict of arguments to be pushed to `init` function.
            Should always contain the key `steps` corresponding to the number of iterations to be done by the model.
            This is done so because `torch.linspace` has this argument named as `steps`.
        """
        assert 'steps' in list(init_kwargs.keys()), \
            '`init_kwargs` should always contain the key `steps` corresponding to the number of iterations to be done by the model.'
        n_iter = init_kwargs['steps']

        betas = init(**init_kwargs)
        alphas = 1 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat([torch.FloatTensor([1]), alphas_cumprod[:-1]])
        alphas_cumprod_prev_with_last = torch.cat([torch.FloatTensor([1]), alphas_cumprod])
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Calculations for posterior q(y_n|y_0)
        sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        # For WaveGrad special continuous noise level conditioning
        self.sqrt_alphas_cumprod_prev = alphas_cumprod_prev_with_last.sqrt().numpy()
        sqrt_recip_alphas_cumprod = (1 / alphas_cumprod).sqrt()
        sqrt_alphas_cumprod_m1 = (1 - alphas_cumprod).sqrt() * sqrt_recip_alphas_cumprod
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod_m1', sqrt_alphas_cumprod_m1)

        # Calculations for posterior q(y_{t-1} | y_t, y_0)
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        posterior_variance = torch.stack([posterior_variance, torch.FloatTensor([1e-20] * n_iter)])
        posterior_log_variance_clipped = posterior_variance.max(dim=0).values.log()
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        posterior_mean_coef1 = betas * alphas_cumprod_prev.sqrt() / (1 - alphas_cumprod)
        posterior_mean_coef2 = (1 - alphas_cumprod_prev) * alphas.sqrt() / (1 - alphas_cumprod)
        self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)
        
        self.n_iter = n_iter
        self.noise_schedule_kwargs = {'init': init, 'init_kwargs': init_kwargs}
        self.noise_schedule_is_set = True

    def sample_continuous_noise_level(self, batch_size, device):
        """
        Samples continuous noise level sqrt(alpha_cumprod).
        This is what makes WaveGrad different from other Denoising Diffusion Probabilistic Models.
        """
        s = np.random.choice(range(1, self.iters + 1), size=batch_size)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[s-1],
                self.sqrt_alphas_cumprod_prev[s],
                size=batch_size
            )
        ).to(device)
        return continuous_sqrt_alpha_cumprod.unsqueeze(-1)
    
    def q_sample(self, y_0, continuous_sqrt_alpha_cumprod=None, eps=None):
        """
        Efficiently computes diffusion version y_t from y_0 using a closed form expression:
            y_t = sqrt(alpha_cumprod)_t * y_0 + sqrt(1 - alpha_cumprod_t) * eps,
            where eps is sampled from a standard Gaussian.
        """
        batch_size = y_0.shape[0]
        continuous_sqrt_alpha_cumprod \
            = self.sample_continuous_noise_level(batch_size, device=y_0.device) \
                if isinstance(eps, type(None)) else continuous_sqrt_alpha_cumprod
        if isinstance(eps, type(None)):
            eps = torch.randn_like(y_0)
        # Closed form signal diffusion
        outputs = continuous_sqrt_alpha_cumprod * y_0 + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * eps
        return outputs

    def q_posterior(self, y_start, y, t):
        """
        Computes reverse (denoising) process posterior q(y_{t-1}|y_0, y_t, x)
        parameters: mean and variance.
        """
        posterior_mean = self.posterior_mean_coef1[t] * y_start + self.posterior_mean_coef2[t] * y
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def predict_start_from_noise(self, y, t, eps):
        """
        Computes y_0 from given y_t and reconstructed noise.
        Is needed to reconstruct the reverse (denoising)
        process posterior q(y_{t-1}|y_0, y_t, x).
        """
        return self.sqrt_recip_alphas_cumprod[t] * y - self.sqrt_alphas_cumprod_m1[t] * eps

    def p_mean_variance(self, mels, y, t, clip_denoised: bool):
        """
        Computes Gaussian transitions of Markov chain at step t
        for further computation of y_{t-1} given current state y_t and features.
        """
        batch_size = mels.shape[0]
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(mels)
        eps_recon = self.nn(mels, y, noise_level)
        y_recon = self.predict_start_from_noise(y, t, eps_recon)

        if clip_denoised:
            y_recon.clamp_(-1.0, 1.0)
        
        model_mean, posterior_log_variance = self.q_posterior(y_start=y_recon, y=y, t=t)
        return model_mean, posterior_log_variance

    def compute_inverse_dynamics(self, mels, y, t, clip_denoised=True):
        """
        Computes reverse (denoising) process dynamics. Closely related to the idea of Langevin dynamics.
        :param mels (torch.Tensor): mel-spectrograms acoustic features of shape [B, n_mels, T//hop_length]
        :param y (torch.Tensor): previous state from dynamics trajectory
        :param clip_denoised (bool, optional): clip signal to [-1, 1]
        :return (torch.Tensor): next state
        """
        model_mean, model_log_variance = self.p_mean_variance(mels, y, t, clip_denoised)
        eps = torch.randn_like(y) if t > 0 else torch.zeros_like(y)
        return model_mean + eps * (0.5 * model_log_variance).exp()

    def sample(self, mels, store_intermediate_states=False):
        """
        Samples speech waveform via progressive denoising of white noise with guidance of mels-epctrogram.
        :param mels (torch.Tensor): mel-spectrograms acoustic features of shape [B, n_mels, T//hop_length]
        :param store_intermediate_states (bool, optional): whether to store dynamics trajectory or not
        :return ys (list of torch.Tensor) (if store_intermediate_states=True)
            or y_0 (torch.Tensor): predicted signals on every dynamics iteration of shape [B, T]
        """
        with torch.no_grad():
            device = next(self.parameters()).device
            batch_size, T = mels.shape[0], mels.shape[-1]
            ys = [torch.randn(batch_size, T*self.total_factor, dtype=torch.float32).to(device)]
            t = self.n_iter - 1
            while t >= 0:
                y_t = self.compute_inverse_dynamics(mels, y=ys[-1], t=t)
                ys.append(y_t)
                t -= 1
            return ys if store_intermediate_states else ys[-1]

    def compute_loss(self, mels, y_0):
        """
        Computes loss between GT Gaussian noise and reconstructed noise by model from diffusion process.
        :param mels (torch.Tensor): mel-spectrograms acoustic features of shape [B, n_mels, T//hop_length]
        :param y_0 (torch.Tensor): GT speech signals
        :return loss (torch.Tensor): loss of diffusion model
        """
        self._verify_noise_schedule_existence()

        # Sample continuous noise level
        batch_size = y_0.shape[0]
        continuous_sqrt_alpha_cumprod \
            = self.sample_continuous_noise_level(batch_size, device=y_0.device)
        eps = torch.randn_like(y_0)

        # Diffuse the signal
        y_noisy = self.q_sample(y_0, continuous_sqrt_alpha_cumprod, eps)

        # Reconstruct the added noise
        eps_recon = self.nn(mels, y_noisy, continuous_sqrt_alpha_cumprod)
        loss = torch.nn.L1Loss()(eps_recon, eps)
        return loss
    
    def compute_loss_reflow(self, mels, y_0, y_1=None, phase_fn=None, mel_fn=None, y_0_pred=None, train_mode=True):
        """
        Computes loss between GT Gaussian noise and reconstructed noise by model from diffusion process.
        :param mels (torch.Tensor): mel-spectrograms acoustic features of shape [B, n_mels, T//hop_length]
        :param y_0 (torch.Tensor): GT speech signals
        :return loss (torch.Tensor): loss of diffusion model
        """
        # self._verify_noise_schedule_existence()

        # Sample continuous noise level
        batch_size = y_0.shape[0]
        
        if self.step_sampler != None:
            t, weights = self.step_sampler.sample(batch_size=batch_size, device=y_0.device)
        else:
            t = self.iters-self.iters/self.distill_step*torch.tensor(np.random.choice(self.distill_step, batch_size)).to(y_0.device)
    
        rescale_t = (t / self.iters).unsqueeze(-1) # 1~1000 --> 0.001~1
        if y_1 == None:
            y_1 = torch.randn_like(y_0)
        # Diffuse the signal
        y_noisy = (1-rescale_t)*y_0 + rescale_t*y_1
        if self.t_ignore:
            tgt = y_noisy - y_0
        else:
            tgt = y_1 - y_0

        # Predict target
        pred = self.nn(mels=mels, yn=y_noisy, noise_level=rescale_t) # conditioned with rescale_t instead of noise_level

        loss = 0

        if self.mstft:
            final_y_0_hat = y_noisy-rescale_t*pred
            # print(sum(mask))
            sc_loss, mag_loss = MultiResolutionSTFTLoss().cuda()(final_y_0_hat, y_0)
            if y_0_pred != None:
                prior_loss = torch.mean(torch.nn.MSELoss(reduction="none")(y_0, y_0_pred), dim=(-1))
                mask = torch.zeros_like(prior_loss)
                mask[prior_loss<0.01] = 1
            else:
                mask = torch.ones_like(sc_loss)
            loss += ((sc_loss + mag_loss)*mask).mean()
            # print(loss.item())
            # if y_0_pred != None:
            #     loss += torch.nn.MSELoss()(final_y_0_hat, y_0_pred)
            #     print(torch.nn.MSELoss()(final_y_0_hat, y_0_pred).item())
            return loss


        if self.phase_loss_weight > 0:
            loss += self.phase_loss_weight*torch.nn.MSELoss()(phase_fn(y_0), phase_fn(y_noisy-rescale_t*pred))
            print(loss.item())
        if self.mag_loss_weight > 0:
            loss += self.mag_loss_weight*torch.nn.MSELoss()(mels, mel_fn(y_noisy-rescale_t*pred))
            print(loss.item())
        if train_mode and self.step_sampler != None:
            loss = torch.nn.MSELoss(reduction="none")(tgt, pred).mean(dim=-1)
            
            self.step_sampler.update_with_local_losses(t.detach().clone(), loss.detach().clone())

            loss = (weights*loss).mean()
        else:
            loss += torch.nn.MSELoss(reduction="mean")(tgt, pred)
        print("--", loss.item())
        return loss
    
    def maybe_optimize_forward(self, mels, store_intermediate_states=False):
        """
        Sample a point in Gaussian Distribution randomly and generate the speech based on this point and mel spectrum
        """
        self.device = next(self.parameters()).device
        batch_size, T = mels.shape[0], mels.shape[-1]
        self.shape = (batch_size, T*self.total_factor)
        self.mels = mels
        def get_slope(t, y):
            y = from_flattened_numpy(y, self.shape).float().to(self.device)
            slope = self.nn(mels=self.mels, yn=y, noise_level=t*torch.ones(self.shape[0], 1).to(self.device))
            return to_flattened_numpy(slope)
        with torch.no_grad():
            y_1 = torch.randn(batch_size, T*self.total_factor, dtype=torch.float32)
            sol = solve_ivp(get_slope, t_span=(1, 1/1000), y0=to_flattened_numpy(y_1), method="RK45", atol=1e-8, rtol=1e-8)
            y_0_pred = from_flattened_numpy(sol.y[:,-1], self.shape).float().to(self.device)
            print(len(sol.t))
            return y_1.to(self.device), y_0_pred.to(self.device)


    def forward(self, mels, store_intermediate_states=False):
        """
        Generates speech from given mel-spectrogram.
        :param mels (torch.Tensor): mel-spectrogram tensor of shape [1, n_mels, T//hop_length]
        :param store_intermediate_states (bool, optional):
            flag to set return tensor to be a set of all states of denoising process 
        """
        self._verify_noise_schedule_existence()
        
        return self.sample(
            mels, store_intermediate_states
        )
    
    def forward_reflow(self, mels, y_1=None, store_intermediate_states=False, test_sample=None, return_noise=False, return_straightness=False):
        """
        Generates speech from given mel-spectrogram.
        :param mels (torch.Tensor): mel-spectrogram tensor of shape [1, n_mels, T//hop_length]
        :param store_intermediate_states (bool, optional):
            flag to set return tensor to be a set of all states of denoising process 
        """
        # self._verify_noise_schedule_existence()


        with torch.no_grad():
            device = next(self.parameters()).device
            batch_size, T = mels.shape[0], mels.shape[-1]
            ys = [torch.randn(batch_size, T*self.total_factor, dtype=torch.float32).to(device)]
            if y_1 is not None:
                ys[0] = y_1.to(device)
            if self.distill_step != self.iters:
                ts = np.linspace(1, 0, self.distill_step+1)
            else:
                ts = np.linspace(1, 0, self.gen_steps+1)
            # pred_old = torch.zeros_like(ys[-1])
            preds = []
            for i, t in enumerate(ts[:-1]):
                y = ys[-1]
                pred = self.nn(mels=mels, yn=y, noise_level=t*torch.ones(batch_size,1).to(device))
                preds.append(pred)
                if self.t_ignore:
                    pred = pred / t
                dt = t - ts[i+1]
                ys.append(y-pred*dt)
            
            straightness = torch.tensor(0)
            if return_straightness:
                straightness = F.mse_loss((ys[0]-ys[-1]).repeat(len(preds), 1), torch.concat(preds, dim=0))

            if return_noise:
                return (ys[0], ys, straightness) if store_intermediate_states else (ys[0], ys[-1], straightness.item())
            else:
                return (ys, straightness) if store_intermediate_states else (ys[-1], straightness.item())


        
    def forward_reflow_record(self, mels, store_intermediate_states=False, test_sample=None, return_noise=None, \
                              return_straightness=None):
        """
        Generates speech from given mel-spectrogram.
        :param mels (torch.Tensor): mel-spectrogram tensor of shape [1, n_mels, T//hop_length]
        :param store_intermediate_states (bool, optional):
            flag to set return tensor to be a set of all states of denoising process 
        """
        # self._verify_noise_schedule_existence()

        os.makedirs("./reflow_gen_wav", exist_ok=True)

        with torch.no_grad():
            device = next(self.parameters()).device
            batch_size, T = mels.shape[0], mels.shape[-1]
            for i in range(1):
                # torch.manual_seed(i)
                ys = [torch.randn(batch_size, T*self.total_factor, dtype=torch.float32).to(device)]
                ts = np.linspace(1, 0, self.gen_steps+1)
                # pred_old = torch.zeros_like(ys[-1])

                fig, ax = plt.subplots()
                imgs = [] # plot gif to record the change of waveform
                start = 2000
                p1 = 1000
                p2 = 2000
                
                point = []
                im = ax.plot(ys[-1][0][start:start+1000].detach().cpu().numpy(), c="b")
                imgs.append(im)
                
                ts = [round(i,5) for i in ts]
                preds = []
                for i, t in enumerate(ts[:-1]):
                    y = ys[-1]
                    point.append(y[0, [p1, p2]])
                    pred = self.nn(mels=mels, yn=y, noise_level=t*torch.ones(batch_size,1).to(device))
                    if self.t_ignore:
                        pred = pred / t
                    preds.append(pred)
                    dt = t - ts[i+1]
                    ys.append(y-pred*dt)
                    # print("path:", (pred-pred_old).mean(dim=-1), "-"*20)
                    # print("path:", torch.cosine_similarity(pred, pred_old, dim=-1).mean(), "-"*20)
                    
                    if test_sample != None:
                        # print("path:", t*(pred - (y - test_sample)/t).norm(dim=-1).mean(), "-"*20)
                        print(t, "path:", torch.cosine_similarity(pred, ys[0]-test_sample, dim=-1).mean(),\
                            pred.norm(dim=-1)-(ys[0]-test_sample).norm(dim=-1), "-"*20)
                        # print("single:", pred.norm(dim=-1), (ys[0]-test_sample).norm(dim=-1), (ys[0]-test_sample-pred).norm(dim=-1))
                        print(((y-t*pred)/(y-t*pred).var(dim=-1)*test_sample.var(dim=-1)-test_sample).norm(dim=-1).mean())
                        print(((y-t*pred)-test_sample).norm(dim=-1).mean())
                        ax.plot(test_sample[0][start:start+1000].detach().cpu().numpy(), c='red', linestyle="--")
                        ax.set_title(f"t={t}")
                        im = ax.plot((y-t*pred)[0][start:start+1000].detach().cpu().numpy(), c="b")
                        ax.set_ylim(-1,1)
                        imgs.append(im)
                    # pred_old = pred
                fig.close()
                # ani = ArtistAnimation(fig, imgs, interval=500)
                # ani.save("./reflow_gen_wav/gen.gif", writer="pillow")
                # print("-"*50)
                # exit(0)
                point.append(ys[-1][0, [p1, p2]])
                point.append(test_sample[0, [p1, p2]])
                point = np.array([i.detach().cpu().numpy() for i in point])
                with open("./reflow_gen_wav/point.txt", "a") as f:
                    f.write(str([list(i) for i in point])+"\n")
                fig, ax = plt.subplots()
                ax.scatter(point[0, 0], point[0, 1], c="blue", linewidths=0.05)
                ax.scatter(point[1:-1, 0], point[1:-1, 1], c="red", linewidths=0.05)
                ax.plot(point[:-1, 0], point[:-1, 1], c="red")
                ax.scatter(point[-1, 0], point[-1, 1], c="orange", linewidths=0.05)
                fig.savefig("../reflow_gen_wav/path.png")
                print(test_sample.var(dim=-1), ys[-1].var(dim=-1))

                straightness = F.mse_loss((ys[0]-ys[-1]).repeat(len(preds), 1), torch.concat(preds, dim=0))
                print("Straightness: ", straightness)

            return ys if store_intermediate_states else (ys[-1], straightness.item())

    def _verify_noise_schedule_existence(self):
        if not self.noise_schedule_is_set:
            raise RuntimeError(
                'No noise schedule is found. Specify your noise schedule '
                'by pushing arguments into `set_new_noise_schedule(...)` method. '
                'For example: '
                "`wavegrad.set_new_noise_level(init=torch.linspace, init_kwargs=\{'steps': 50, 'start': 1e-6, 'end': 1e-2\})`."
            )
