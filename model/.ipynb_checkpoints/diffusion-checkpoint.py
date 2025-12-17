import torch
import torch.nn.functional as F
import numpy as np


class DiffusionScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps

        # Linearly spaced noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise=None):
        """
        Adds noise to the input x_start at timestep t using the forward diffusion process.

        x_start: [batch_size, seq_len, embed_dim]
        t:       [batch_size]  (each example gets a different t)
        noise:   optional, usually sampled as Gaussian

        Returns:
        - x_t: noised version of x_start
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Grab sqrt(alpha_bar_t) and sqrt(1 - alpha_bar_t)
        sqrt_alpha_bar = self.alpha_bars[t].sqrt().unsqueeze(-1).unsqueeze(-1)
        sqrt_one_minus = (1. - self.alpha_bars[t]).sqrt().unsqueeze(-1).unsqueeze(-1)

        return sqrt_alpha_bar * x_start + sqrt_one_minus * noise
