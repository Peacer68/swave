import numpy as np

import torch
from .criterion import cal_si_snr_with_pit
from .mul_stft_loss import MultiResolutionSTFTLoss

from model.base import BaseModule
from model.layers import Conv1dWithInitialization
from model.upsampling import UpsamplingBlock as UBlock
from model.downsampling import DownsamplingBlock as DBlock
from model.linear_modulation import FeatureWiseLinearModulation as FiLM
import torch.nn.functional as F

class LVCBlock(torch.nn.Module):
    ''' location-variable convolutions
    '''
    def __init__(self,
                 config=None,
                 in_channels=1,
                 cond_channels=80,
                 conv_layers=4,
                 conv_kernel_size=3,
                 cond_hop_length=300,
                 kpnet_hidden_channels=64,
                 kpnet_conv_size=3,
                 kpnet_dropout=0.0,
                 ):
        super().__init__()

        if "mstft" in config.training_config and config.training_config.mstft:
            self.mstft = True
            self.mstft_loss = MultiResolutionSTFTLoss()
        else:
            self.mstft = False

        if "snr_loss_weight" in config.training_config:
            self.snr_loss_weight = config.training_config.snr_loss_weight
        else:
            self.snr_loss_weight = 0

        if "phase_loss_weight" in config.training_config:
            self.phase_loss_weight = config.training_config.phase_loss_weight
        else:
            self.phase_loss_weight = 0

        if "mag_loss_weight" in config.training_config:
            self.mag_loss_weight = config.training_config.mag_loss_weight
        else:
            self.mag_loss_weight = 0

        self.cond_hop_length = cond_hop_length
        self.conv_layers = conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.convs = torch.nn.ModuleList()



        self.kernel_predictor = KernelPredictor(
            cond_channels=cond_channels,
            conv_in_channels=in_channels,
            conv_out_channels=2 * in_channels,
            conv_layers=conv_layers,
            conv_kernel_size=conv_kernel_size,
            kpnet_hidden_channels=kpnet_hidden_channels,
            kpnet_conv_size=kpnet_conv_size,
            kpnet_dropout=kpnet_dropout
        )

        for i in range(conv_layers):
            padding = (3 ** i) * int((conv_kernel_size - 1) / 2)
            conv = torch.nn.Conv1d(in_channels, in_channels, kernel_size=conv_kernel_size, padding=padding, dilation=3 ** i)

            self.convs.append(conv)
    def location_variable_convolution(self, x, kernel, bias, dilation, hop_size):
        ''' perform location-variable convolution operation on the input sequence (x) using the local convolution kernl.
        Time: 414 μs ± 309 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each), test on NVIDIA V100.
        Args:
            x (Tensor): the input sequence (batch, in_channels, in_length).
            kernel (Tensor): the local convolution kernel (batch, in_channel, out_channels, kernel_size, kernel_length)
            bias (Tensor): the bias for the local convolution (batch, out_channels, kernel_length)
            dilation (int): the dilation of convolution.
            hop_size (int): the hop_size of the conditioning sequence.
        Returns:
            (Tensor): the output sequence after performing local convolution. (batch, out_channels, in_length).
        '''
        batch, in_channels, in_length = x.shape
        batch, in_channels, out_channels, kernel_size, kernel_length = kernel.shape


        assert in_length == (kernel_length * hop_size), "length of (x, kernel) is not matched"

        padding = dilation * int((kernel_size - 1) / 2)
        x = F.pad(x, (padding, padding), 'constant', 0)  # (batch, in_channels, in_length + 2*padding)
        x = x.unfold(2, hop_size + 2 * padding, hop_size)  # (batch, in_channels, kernel_length, hop_size + 2*padding)

        if hop_size < dilation:
            x = F.pad(x, (0, dilation), 'constant', 0)
        x = x.unfold(3, dilation,
                     dilation)  # (batch, in_channels, kernel_length, (hop_size + 2*padding)/dilation, dilation)
        x = x[:, :, :, :, :hop_size]
        x = x.transpose(3, 4)  # (batch, in_channels, kernel_length, dilation, (hop_size + 2*padding)/dilation)
        x = x.unfold(4, kernel_size, 1)  # (batch, in_channels, kernel_length, dilation, _, kernel_size)

        o = torch.einsum('bildsk,biokl->bolsd', x, kernel)
        o = o + bias.unsqueeze(-1).unsqueeze(-1)
        o = o.contiguous().view(batch, out_channels, -1)
        return o


    def forward(self, y_0, mel):
        ''' forward propagation of the time-aware location-variable convolutions.
        Args:
            x (Tensor): the input sequence (batch, in_channels, in_length)
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)

        Returns:
            Tensor: the output sequence (batch, in_channels, in_length)
        '''
        x, condition= y_0, mel
        batch, in_channels, in_length = x.shape

        kernels, bias = self.kernel_predictor(condition)

        for i in range(self.conv_layers):
            y = F.leaky_relu(x, 0.2)
            y = self.convs[i](y)
            y = F.leaky_relu(y, 0.2)

            k = kernels[:, i, :, :, :, :]
            b = bias[:, i, :, :]
            y = self.location_variable_convolution(y, k, b, 1, self.cond_hop_length)
            # x = x + torch.sigmoid(y[:, :in_channels, :]) * torch.tanh(y[:, in_channels:, :])
            x = torch.sigmoid(y[:, :in_channels, :]) * torch.tanh(y[:, in_channels:, :])
        return x.squeeze(1)
    def compute_loss_reflow(self, y_0, y_0_hat, mels, mel_fn=None, phase_fn=None):
        loss = 0
        final_y_0_hat = self.forward(y_0_hat.unsqueeze(1), mels)
        if self.mstft:
            sc_loss, mag_loss = self.mstft_loss(final_y_0_hat, y_0)
            loss += (sc_loss + mag_loss)
            return loss
        if self.snr_loss_weight > 0:
            max_snr, _, _ = cal_si_snr_with_pit(y_0.unsqueeze(1), final_y_0_hat.unsqueeze(1), source_lengths=(torch.ones(y_0.size(0))*y_0.size(-1)).long().to(y_0.device))
            loss += -self.snr_loss_weight*torch.mean(max_snr)
        if self.phase_loss_weight > 0:
            loss += self.phase_loss_weight*torch.nn.MSELoss()(phase_fn(y_0), phase_fn(final_y_0_hat))
            # print(loss.item())
        if self.mag_loss_weight > 0:
            loss += self.mag_loss_weight*torch.nn.L1Loss()(mels, mel_fn(final_y_0_hat))
            # print(loss.item())
        # loss += torch.nn.MSELoss()(final_y_0_hat, y_0)
            
        return loss

class KernelPredictor(torch.nn.Module):
    ''' Kernel predictor for the time-aware location-variable convolutions
    '''

    def __init__(self,
                 cond_channels,
                 conv_in_channels,
                 conv_out_channels,
                 conv_layers,
                 conv_kernel_size=3,
                 kpnet_hidden_channels=64,
                 kpnet_conv_size=3,
                 kpnet_dropout=0.0,
                 kpnet_nonlinear_activation="LeakyReLU",
                 kpnet_nonlinear_activation_params={"negative_slope": 0.1}
                 ):
        '''
        Args:
            cond_channels (int): number of channel for the conditioning sequence,
            conv_in_channels (int): number of channel for the input sequence,
            conv_out_channels (int): number of channel for the output sequence,
            conv_layers (int):
            kpnet_
        '''
        super().__init__()

        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_layers = conv_layers

        l_w = conv_in_channels * conv_out_channels * conv_kernel_size * conv_layers
        l_b = conv_out_channels * conv_layers

        padding = (kpnet_conv_size - 1) // 2
        self.input_conv = torch.nn.Sequential(
            torch.nn.Conv1d(cond_channels, kpnet_hidden_channels, 5, padding=(5 - 1) // 2, bias=True),
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
        )

        self.residual_conv = torch.nn.Sequential(
            torch.nn.Dropout(kpnet_dropout),
            torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True),
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
            torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True),
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
            torch.nn.Dropout(kpnet_dropout),
            torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True),
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
            torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True),
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
            torch.nn.Dropout(kpnet_dropout),
            torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True),
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
            torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True),
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
        )

        self.kernel_conv = torch.nn.Conv1d(kpnet_hidden_channels, l_w, kpnet_conv_size,
                                           padding=padding, bias=True)
        self.bias_conv = torch.nn.Conv1d(kpnet_hidden_channels, l_b, kpnet_conv_size, padding=padding,
                                         bias=True)

    def forward(self, c):
        '''
        Args:
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)
        Returns:
        '''
        batch, cond_channels, cond_length = c.shape

        c = self.input_conv(c)
        c = c + self.residual_conv(c)
        k = self.kernel_conv(c)
        b = self.bias_conv(c)

        kernels = k.contiguous().view(batch,
                                      self.conv_layers,
                                      self.conv_in_channels,
                                      self.conv_out_channels,
                                      self.conv_kernel_size,
                                      cond_length)
        bias = b.contiguous().view(batch,
                                   self.conv_layers,
                                   self.conv_out_channels,
                                   cond_length)
        return kernels, bias

class FinetuneNN(BaseModule):
    """Finetune the output of WaveGradNN"""
    def __init__(self, config, T=7200):
        """
        T: the length of waveform
        """
        super().__init__()
        # self.layernorm = torch.nn.LayerNorm(T)
        self.relu = torch.nn.LeakyReLU(0.2)
        if "kernel_size" in config.training_config:
            kernel_size = config.training_config.kernel_size
            padding = int((kernel_size-1)/2)
        else:
            kernel_size, padding = 3, 1
        if "hidden_channels" in config.training_config:
            hidden_channels = config.training_config.hidden_channels
        else:
            hidden_channels = 1
        self.postconv1 = torch.nn.Conv1d(in_channels=1, out_channels=hidden_channels, kernel_size=kernel_size,\
                                        stride=1, padding=padding)
        self.postconv2 = torch.nn.Conv1d(in_channels=hidden_channels, out_channels=1, kernel_size=kernel_size,\
                                        stride=1, padding=padding)
        self.config = config
        torch.nn.init.normal_(self.postconv1.weight.data)
        torch.nn.init.normal_(self.postconv2.weight.data)
        
        if "snr_loss_weight" in config.training_config:
            self.snr_loss_weight = config.training_config.snr_loss_weight
        else:
            self.snr_loss_weight = 0

        if "phase_loss_weight" in config.training_config:
            self.phase_loss_weight = config.training_config.phase_loss_weight
        else:
            self.phase_loss_weight = 0

        if "mag_loss_weight" in config.training_config:
            self.mag_loss_weight = config.training_config.mag_loss_weight
        else:
            self.mag_loss_weight = 0
        if "layernorm" in config.training_config:
            self.ln = config.training_config.layernorm
        else:
            self.ln = False

    def forward(self, input, mel=None):
        assert len(input.shape) == 3
        if self.ln:
            input = torch.nn.LayerNorm(input.size(-1), device=input.device)(input)
        # output = self.postconv(self.relu(self.layernorm(input)))
        output = self.postconv2(self.relu(self.postconv1(input)))
        # output = output/output.var(dim=-1).sqrt().unsqueeze(-1)+output.mean(dim=-1).unsqueeze(-1)
        return output.squeeze(1)
    def compute_loss_reflow(self, y_0, y_0_hat, mels, mel_fn=None, phase_fn=None):
        loss = 0
        final_y_0_hat = self.forward(y_0_hat.unsqueeze(1), mels)
        
        if self.snr_loss_weight > 0:
            max_snr, _, _ = cal_si_snr_with_pit(y_0.unsqueeze(1), final_y_0_hat.unsqueeze(1), source_lengths=(torch.ones(y_0.size(0))*y_0.size(-1)).long().to(y_0.device))
            loss += -self.snr_loss_weight*torch.mean(max_snr)
        if self.phase_loss_weight > 0:
            loss += self.phase_loss_weight*torch.nn.MSELoss()(phase_fn(y_0), phase_fn(final_y_0_hat))
            # print(loss.item())
        if self.mag_loss_weight > 0:
            loss += self.mag_loss_weight*torch.nn.L1Loss()(mels, mel_fn(final_y_0_hat))
            # print(loss.item())
        loss += torch.nn.MSELoss()(final_y_0_hat, y_0)
            
        return loss



class WaveGradNN(BaseModule):
    """
    WaveGrad is a fully-convolutional mel-spectrogram conditional
    vocoder model for waveform generation introduced in
    "WaveGrad: Estimating Gradients for Waveform Generation" paper (link: https://arxiv.org/pdf/2009.00713.pdf).
    The concept is built on the prior work on score matching and diffusion probabilistic models.
    Current implementation follows described architecture in the paper.
    """
    def __init__(self, config):
        super(WaveGradNN, self).__init__()
        self.t_ignore = config.training_config.t_ignore
        # Building upsampling branch (mels -> signal)
        self.ublock_preconv = Conv1dWithInitialization(
            in_channels=config.data_config.n_mels,
            out_channels=config.model_config.upsampling_preconv_out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        upsampling_in_sizes = [config.model_config.upsampling_preconv_out_channels] \
            + config.model_config.upsampling_out_channels[:-1]
        self.ublocks = torch.nn.ModuleList([
            UBlock(
                in_channels=in_size,
                out_channels=out_size,
                factor=factor,
                dilations=dilations
            ) for in_size, out_size, factor, dilations in zip(
                upsampling_in_sizes,
                config.model_config.upsampling_out_channels,
                config.model_config.factors,
                config.model_config.upsampling_dilations
            )
        ])
        self.ublock_postconv = Conv1dWithInitialization(
            in_channels=config.model_config.upsampling_out_channels[-1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Building downsampling branch (starting from signal)
        self.dblock_preconv = Conv1dWithInitialization(
            in_channels=1,
            out_channels=config.model_config.downsampling_preconv_out_channels,
            kernel_size=5,
            stride=1,
            padding=2
        )
        downsampling_in_sizes = [config.model_config.downsampling_preconv_out_channels] \
            + config.model_config.downsampling_out_channels[:-1]
        self.dblocks = torch.nn.ModuleList([
            DBlock(
                in_channels=in_size,
                out_channels=out_size,
                factor=factor,
                dilations=dilations
            ) for in_size, out_size, factor, dilations in zip(
                downsampling_in_sizes,
                config.model_config.downsampling_out_channels,
                config.model_config.factors[1:][::-1],
                config.model_config.downsampling_dilations
            )
        ])

        # Building FiLM connections (in order of downscaling stream)
        film_in_sizes = [32] + config.model_config.downsampling_out_channels
        film_out_sizes = config.model_config.upsampling_out_channels[::-1]
        film_factors = [1] + config.model_config.factors[1:][::-1]
        self.films = torch.nn.ModuleList([
            FiLM(
                in_channels=in_size,
                out_channels=out_size,
                linear_scale=config.training_config.training_noise_schedule.n_iter,
                input_dscaled_by=np.product(film_factors[:i+1])  # for proper positional encodings initialization
            ) for i, (in_size, out_size) in enumerate(
                zip(film_in_sizes, film_out_sizes)
            )
        ])

    def forward(self, mels, yn, noise_level):
        """
        Computes forward pass of neural network.
        :param mels (torch.Tensor): mel-spectrogram acoustic features of shape [B, n_mels, T//hop_length]
        :param yn (torch.Tensor): noised signal `y_n` of shape [B, T]
        :param noise_level (float): level of noise added by diffusion
        :return (torch.Tensor): epsilon noise
        """
        # Prepare inputs
        if self.t_ignore:
            noise_level = torch.ones_like(noise_level)
        assert len(mels.shape) == 3  # B, n_mels, T
        yn = yn.unsqueeze(1)
        assert len(yn.shape) == 3  # B, 1, T

        # Downsampling stream + Linear Modulation statistics calculation
        statistics = []
        dblock_outputs = self.dblock_preconv(yn)
        scale, shift = self.films[0](x=dblock_outputs, noise_level=noise_level)
        statistics.append([scale, shift])
        for dblock, film in zip(self.dblocks, self.films[1:]):
            dblock_outputs = dblock(dblock_outputs)
            scale, shift = film(x=dblock_outputs, noise_level=noise_level)
            statistics.append([scale, shift])
        statistics = statistics[::-1]
        
        # Upsampling stream
        ublock_outputs = self.ublock_preconv(mels)
        for i, ublock in enumerate(self.ublocks):
            scale, shift = statistics[i]
            ublock_outputs = ublock(x=ublock_outputs, scale=scale, shift=shift)
        outputs = self.ublock_postconv(ublock_outputs)
        return outputs.squeeze(1)
