import os
# 指定显卡可见性
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
# 制图
from torchvision.utils import make_grid, save_image

from nets.UNet_v2 import UNet
from nets.VAE import VAE
from utils.DenoisingDiffusion import GaussianDiffusion


def unet_sample_images(img_gen, batches_done, dir_output="images"):
    """Saves a generated sample"""
    b = img_gen.size()[0]
    
    # Arrange images along x-axis
    img_gen = make_grid(img_gen, nrow=b, normalize=True)

    save_image(img_gen, os.path.join("result", dir_output, "unet-test_%d.png" % batches_done), normalize=False)


vae1 = VAE()
unet1 = UNet()

noise_helper = GaussianDiffusion()
noise_helper = noise_helper.cuda()
noise_helper.eval()

vae1.load_state_dict(torch.load(os.path.join("result", "models", "vae.ckpt"), map_location='cpu'))
unet1.load_state_dict(torch.load(os.path.join("result", "models", "unet-200.ckpt"), map_location='cpu'), strict=False)

vae1 = vae1.cuda()
vae1.eval()
unet1 = unet1.cuda()
unet1.eval()

with torch.no_grad():
    for batch in range(5):
        # ddim阶段 unet从完全的噪声中预测
        latent_gen = noise_helper.ddim_sample(model=unet1, shape=(1,4,48,64))
        
        # 从压缩图恢复成图片
        vae_seed = latent_gen / 0.18215

        # [1, 4, 48, 64] -> [1, 3, 384, 512]
        img_gen = vae1.decoder(vae_seed)
        unet_sample_images(img_gen=img_gen, batches_done=batch)