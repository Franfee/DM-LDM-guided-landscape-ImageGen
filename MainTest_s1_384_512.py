import datetime
import time
import os
# 指定显卡可见性
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch

# 制图
from torchvision.utils import make_grid, save_image

from nets.VAE import VAE
from utils.lr_scheduler import exp_lr_scheduler
from utils.get_all_parsar import *
from utils.MyDataLoader import get_dataloader

def vae_sample_images(img_ref, img_rec, batches_done, dir_output="images"):
    """Saves a generated sample"""
    b = img_ref.size()[0]

    # Arrange images along x-axis
    img_ref = make_grid(img_ref, nrow=b, normalize=True)
    img_rec = make_grid(img_rec, nrow=b, normalize=True)
    # Arrange images along y-axis
    image_grid = torch.cat((img_ref, img_rec), 1)
    save_image(image_grid, os.path.join("result", dir_output, "vae-test_%d.png" % batches_done), normalize=False)


dataLoader = get_dataloader(muti=False)
vae1 = VAE()
vae1.load_state_dict(torch.load(os.path.join("result", "models", "vae.ckpt"), map_location='cpu'))
vae1 = vae1.cuda()
vae1 = vae1.eval()

with torch.no_grad():
    for idx, data in enumerate(dataLoader):
        print("idx:", idx)
        img_ref = data['img_ref']
        img_ref = img_ref.cuda()

        recover_img = vae1(img_ref)

        vae_sample_images(img_ref, recover_img, idx)

        if idx > 5:
            break