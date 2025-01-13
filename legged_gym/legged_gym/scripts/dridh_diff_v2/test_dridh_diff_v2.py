import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F

import matplotlib.pyplot as plt
import numpy as np

from .positional_embeddings import PositionalEmbedding
from tqdm.auto import tqdm

class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))

class MLP_dridh(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 6, time_emb_size: int = 128,obs_emb_size: int = 128,
                 time_emb: str = "sinusoidal", obs_emb: str = "learnable_multi",obs_size=None,action_size=None):
        super().__init__()

        self.time_mlp = PositionalEmbedding(time_emb_size, time_emb)
        self.obs_mlp1 = PositionalEmbedding(obs_emb_size,  type=obs_emb,input_size=obs_size)
        # self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        concat_size = action_size + len(self.time_mlp.layer) + \
            len(self.obs_mlp1.layer)
        
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))

        layers.append(nn.Linear(hidden_size, action_size))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t, obs):
        # x - (batch_size,action_size)
        obs_emb = self.obs_mlp1(obs)
        t_emb = self.time_mlp(t)
        x = torch.cat((x, obs_emb, t_emb), dim=-1)
        # x - (batch_size,action_size+obs_emb_size+time_emb_size)
        x = self.joint_mlp(x)
        # x - (batch_size,action_size)
        return x

class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear"):

        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps

class diff_policy_v2():
    def __init__(self,action_size = 6,obs_size = 30*5,path = "exps/dridh_base/model.pth",eval_batch_size = 1,num_diff_timesteps = 60):
    

        self.model = MLP_dridh(
            hidden_size=256,
            hidden_layers=7,
            obs_emb_size=48,
            time_emb="sinusoidal",
            obs_emb="learnable_multi",
            obs_size=obs_size,
            action_size=action_size)


        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        self.noise_scheduler = NoiseScheduler(num_timesteps=num_diff_timesteps)
        self.eval_batch_size = eval_batch_size
        self.action_size = action_size
        self.num_diff_timesteps = num_diff_timesteps
        
    def sample_action(self,obs):

        ###########
        obs = torch.tensor(obs).unsqueeze(0)
        ###########

        sample = torch.randn(self.eval_batch_size, self.action_size)
        timesteps = list(range(self.num_diff_timesteps))[::-1]
        for i, t in enumerate(timesteps):
            t = torch.from_numpy(np.repeat(t, self.eval_batch_size)).long()
            with torch.no_grad():
                residual = self.model(sample, t, obs)
            sample = self.noise_scheduler.step(residual, t[0], sample)

        return  sample


