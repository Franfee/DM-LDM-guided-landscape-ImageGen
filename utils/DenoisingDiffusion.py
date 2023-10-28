# -*- coding: utf-8 -*-
# @Time    : 2023/6/23 12:33
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12

import torch
from torch import nn
from tqdm.auto import tqdm


# normalization functions
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, beta_scheduler='sigmoid', timesteps=1000, sampling_timesteps=50, ddim_sampling_eta=0,
                 schedule_fn_kwargs=dict()):
        super().__init__()

        if beta_scheduler == 'linear':
            from utils.Beta_Scheduler import linear_beta_schedule
            beta_schedule_fn = linear_beta_schedule
        elif beta_scheduler == 'cosine':
            from utils.Beta_Scheduler import cosine_beta_schedule
            beta_schedule_fn = cosine_beta_schedule
        elif beta_scheduler == 'sigmoid':
            from utils.Beta_Scheduler import sigmoid_beta_schedule
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_scheduler}')

        # linear : img_noise=sqrt(betas)*N + sqrt(1-betas)*X
        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # cumulative product, 累积乘法

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = sampling_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        self.unnormalize = unnormalize_to_zero_to_one

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_sample(self, x_start, t, noise):
        """
        noise sample for xt
        """
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @torch.no_grad()
    def ddim_sample(self, model, shape, mask_condition=None, img=None, return_all_timesteps=False):
        # shape = (batch_size, channels, image_size, image_size)
        batch, device = shape[0], self.betas.device
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        if img is None:
            img = torch.randn(shape, device=device)
        imgs = [img]

        # start ddim sampling loop
        for time_step, time_next in tqdm(time_pairs, desc='ddim sampling loop time step\n'):
            time_cond = torch.full((batch,), time_step, device=device, dtype=torch.long)

            # model_out = model(out_vae, mask_condition, time_step)
            if mask_condition is not None:
                pred_noise = model(img, mask_condition, time_cond)
            else:
                pred_noise = model(img, time_cond)
            x_start = self.predict_start_from_noise(img, time_cond, pred_noise)
            x_start = torch.clamp(x_start, min=-1., max=1.)
            pred_noise = self.predict_noise_from_start(img, time_cond, x_start)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time_step]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)
        return ret

    def forward(self, x_start, t, noise=None):
        """
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return: x_noised : add noise to x
                noise    : norm noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_start = x_start.cuda()
        noise = noise.cuda()
        t = t.cuda()

        x_noised = self.q_sample(x_start=x_start, t=t, noise=noise)
        return x_noised, noise


# if __name__ == '__main__':
#     noise_helper = GaussianDiffusion().cuda()
#     b = 2
#     vae_out = torch.zeros(b, 4, 48, 64)
#     vae_out = vae_out.cuda()
#     noise_one = torch.ones(b, 4, 48, 64)
#     noise_one = noise_one.cuda()
#     noise_step = torch.randint(0, 1, (b,)).long()
#     noise_step = noise_step.cuda()
#     x_noise, n_noise = noise_helper(vae_out, noise_step, noise=noise_one)
#     print("step 1")
#     print(x_noise[0, 0, 0, :10])
#     print(n_noise[0, 0, 0, :10])
#
#     noise_step = torch.randint(0, 500, (b,)).long()
#     x_noise, n_noise = noise_helper(vae_out, noise_step, noise=noise_one)
#     print("step 500")
#     print(x_noise[0, 0, 0, :10])
#     print(n_noise[0, 0, 0, :10])
#
#     noise_step = torch.randint(0, 999, (b,)).long()
#     x_noise, n_noise = noise_helper(vae_out, noise_step, noise=noise_one)
#     print("step 999")
#     print(x_noise[0, 0, 0, :10])
#     print(n_noise[0, 0, 0, :10])
#
#     noise_step = torch.randint(0, 1000, (b,)).long()
#     x_noise, n_noise = noise_helper(vae_out, noise_step, noise=noise_one)
#     print("step 1000")
#     print(x_noise[0, 0, 0, :10])
#     print(n_noise[0, 0, 0, :10])

# from nets.UNet import UNet
#
# unet = UNet()
# mask_cond = torch.ones(1, 1, 384, 512)
# rets = noise_helper.ddim_sample(model=unet, shape=vae_out.size(), mask_condition=mask_cond)
#
# print(rets.shape)
