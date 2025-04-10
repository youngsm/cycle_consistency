import math
from functools import partial
from typing import Callable, Dict, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os


def print_grad(name):
    def hook(grad):
        print(f"Gradient for {name}: {grad}")

    return hook


T = TypeVar("T")


class Config(Dict[str, T]):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getattr__(self, name: str) -> T:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: T) -> None:
        if isinstance(value, str):
            value = eval(value, {}, {"uniform": np.random.uniform})
        self[name] = value


class BatchedLightSimulation(nn.Module):
    def __init__(
        self,
        cfg: str = os.path.join(
            os.path.dirname(__file__), "../templates/waveform_sim.yaml"
        ),
        verbose: bool = False,
    ):
        super().__init__()

        if isinstance(cfg, str):
            cfg_txt = open(cfg, "r").readlines()
            print("BatchedLightSimulation Config:\n\t%s" % "\t".join(cfg_txt))
            cfg = yaml.safe_load("".join(cfg_txt))
            cfg = Config(**cfg)

        self.cfg = cfg

        # Load parameters directly from config
        self.singlet_fraction = cfg.SINGLET_FRACTION
        self.tau_s = cfg.TAU_S
        self.tau_t = cfg.TAU_T
        self.light_oscillation_period = cfg.LIGHT_OSCILLATION_PERIOD
        self.light_response_time = cfg.LIGHT_RESPONSE_TIME
        self.light_gain = cfg.LIGHT_GAIN

        # Constants
        self.light_tick_size = cfg.LIGHT_TICK_SIZE
        self.light_window = cfg.LIGHT_WINDOW

        self.conv_ticks = math.ceil(
            (self.light_window[1] - self.light_window[0]) / self.light_tick_size
        )

        self.time_ticks = torch.arange(self.conv_ticks)

        self.k = 100

        if verbose:
            self.register_grad_hook()

    def to(self, device):
        self.time_ticks = self.time_ticks.to(device)
        return super().to(device)

    def cuda(self):
        return self.to("cuda")

    @property
    def device(self):
        return self.time_ticks.device

    def register_grad_hook(self):
        for name, p in self.named_parameters():
            p.register_hook(print_grad(name))

    def scintillation_model(
        self, time_tick: torch.Tensor, relax_cut: bool = True
    ) -> torch.Tensor:
        """
        Calculates the fraction of scintillation photons emitted
        during time interval `time_tick` to `time_tick + 1`

        Args:
            time_tick (torch.Tensor): time tick relative to t0
            relax_cut (bool): whether to apply the relaxing cut for differentiability

        Returns:
            torch.Tensor: fraction of scintillation photons
        """
        t = time_tick * self.light_tick_size

        p1 = (
            self.singlet_fraction
            * torch.exp(-t / self.tau_s)
            * (1 - torch.exp(-self.light_tick_size / self.tau_s))
        )
        p3 = (
            (1 - self.singlet_fraction)
            * torch.exp(-t / self.tau_t)
            * (1 - torch.exp(-self.light_tick_size / self.tau_t))
        )

        if relax_cut:
            return (p1 + p3) / (1 + torch.exp(-self.k * t))

        return (p1 + p3) * (t >= 0).float()

    def sipm_response_model(self, time_tick, relax_cut=True) -> torch.Tensor:
        """
        Calculates the SiPM response from a PE at `time_tick` relative to the PE time

        Args:
            time_tick (torch.Tensor): time tick relative to t0
            relax_cut (bool): whether to apply the relaxing cut for differentiability

        Returns:
            torch.Tensor: response
        """
        t = time_tick * self.light_tick_size

        impulse = torch.exp(-t / self.light_response_time) * torch.sin(
            t / self.light_oscillation_period
        )
        if relax_cut:
            impulse = impulse / (1 + torch.exp(-self.k * time_tick))
        else:
            impulse = impulse * (time_tick >= 0).float()

        impulse /= self.light_oscillation_period * self.light_response_time**2
        impulse *= self.light_oscillation_period**2 + self.light_response_time**2
        return impulse * self.light_tick_size

    def fft_conv(self, light_sample_inc: torch.Tensor, model: Callable) -> torch.Tensor:
        """
        Performs a Fast Fourier Transform (FFT) convolution on the input light sample.

        Args:
            light_sample_inc (torch.Tensor): Light incident on each detector.
                Shape: (ninput, ndet, ntick)
            model (callable): Function that generates the convolution kernel.

        Returns:
            torch.Tensor: Convolved light sample.
                Shape: (ninput, ndet, ntick)

        This method applies the following steps:
        1. Pads the input tensor
        2. Computes the convolution kernel using the provided model
        3. Performs FFT on both the input and the kernel
        4. Multiplies the FFTs in the frequency domain
        5. Performs inverse FFT to get the convolved result
        6. Reshapes and trims the output to match the input shape
        """
        ninput, ndet, ntick = light_sample_inc.shape

        # Pad the input tensor
        pad_size = self.conv_ticks - 1
        padded_input = F.pad(light_sample_inc, (0, pad_size))

        # Compute kernel
        scintillation_kernel = model(self.time_ticks)
        kernel_fft = torch.fft.rfft(scintillation_kernel, n=ntick + pad_size)

        # Reshape for batched FFT convolution
        padded_input = padded_input.reshape(ninput * ndet, ntick + pad_size)

        # Perform FFT convolution
        input_fft = torch.fft.rfft(padded_input)
        output_fft = input_fft * kernel_fft.unsqueeze(0)
        output = torch.fft.irfft(output_fft, n=ntick + pad_size)

        # Reshape and trim the result to match the input shape
        output = output.reshape(ninput, ndet, -1)  # [:, :, :ntick]

        return output

    def downsample_waveform(
        self, waveform: torch.Tensor, ns_per_tick: int = 16
    ) -> torch.Tensor:
        """
        Downsample the input waveform by summing over groups of ticks.
        This effectively compresses the waveform in the time dimension while preserving the total signal.

        Args:
            waveform (torch.Tensor): Input waveform tensor of shape (ninput, ndet, ntick), where each tick corresponds to 1 ns.
            ns_per_tick (int, optional): Number of nanoseconds per tick in the downsampled waveform. Defaults to 16.

        Returns:
            torch.Tensor: Downsampled waveform of shape (ninput, ndet, ntick_down).
        """
        ninput, ndet, ntick = waveform.shape
        ntick_down = ntick // ns_per_tick
        downsample = waveform.view(ninput, ndet, ntick_down, ns_per_tick).sum(dim=3)
        return downsample

    def forward(
        self,
        timing_dist: torch.Tensor,
        relax_cut: bool = True,
        return_intermediates: bool = False,
        downscale_factor: int = 1,
    ):
        reshaped = False
        if timing_dist.ndim == 1:  # ndet=1, ninput=1; (ntick) -> (1, 1, ntick)
            timing_dist = timing_dist[None, None, :]
            reshaped = True
        elif (
            timing_dist.ndim == 2
        ):  # ndet>1, ninput=1; (ndet, ntick) -> (1, ndet, ntick)
            timing_dist = timing_dist[None, :, :]
            reshaped = True

        intermediates = {}
        intermediates["input"] = timing_dist.detach()

        x = self.fft_conv(
            timing_dist, partial(self.scintillation_model, relax_cut=relax_cut)
        )
        intermediates["after_scintillation"] = x.detach()

        x = self.fft_conv(x, partial(self.sipm_response_model, relax_cut=relax_cut))
        intermediates["after_sipm_response"] = x.detach()

        x = self.light_gain * x
        intermediates["after_gain"] = x.detach()

        x = self.downsample_waveform(x, downscale_factor)
        intermediates["after_downsample"] = x.detach()

        if reshaped:
            result = x.squeeze(0).squeeze(0)
            if return_intermediates:
                return result, intermediates
            return result

        if return_intermediates:
            return x, intermediates
        return x
