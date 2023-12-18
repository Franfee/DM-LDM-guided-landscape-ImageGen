# -*- coding: utf-8 -*-
# @Time    : 2023/6/22 11:02
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12
import datetime
import time
import os
# 指定显卡可见性
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch

# 自动混合精度
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

# 制图
from torchvision.utils import make_grid, save_image

from nets.VAE import VAE
from utils.lr_scheduler import exp_lr_scheduler
from utils.get_all_parsar import *


def vae_sample_images(img_ref, img_rec, batches_done, dir_output="images"):
    """Saves a generated sample"""
    b = img_ref.size()[0]

    # Arrange images along x-axis
    img_ref = make_grid(img_ref, nrow=b, normalize=True)
    img_rec = make_grid(img_rec, nrow=b, normalize=True)
    # Arrange images along y-axis
    image_grid = torch.cat((img_ref, img_rec), 1)
    save_image(image_grid, os.path.join("result", dir_output, "vae-step_%d.png" % batches_done), normalize=False)


def Stage1_Train_VAE():
    from utils.MyDataLoader import get_dataloader
    dataLoader = get_dataloader(muti=False)
    vae1 = VAE()

    if LOAD_CHECK_POINT_VAE:
        vae1.load_state_dict(torch.load(os.path.join("result", "models", f"vae-{LOAD_VAE_IDX}.ckpt"), map_location='cpu'))
        # vae1.load_state_dict(torch.load(os.path.join("result", "models", "vae.ckpt"), map_location='cpu'))
        print("loaded!")
    else:
        # init
        pass

    optimizer = torch.optim.AdamW(vae1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=0.01, eps=1e-8)
    criterion_recover = torch.nn.L1Loss()
    # GradScaler对象用于自动混合精度
    scaler = GradScaler()

    # --------GPU-----------
    vae1.cuda()
    criterion_recover.cuda()

    # --------train---------
    prev_time = time.time()
    for epoch in range(opt.s1start_epoch, opt.s1_epochs):
        vae1.train()
        # adjust learning rate
        optimizer = exp_lr_scheduler(optimizer, epoch, opt.lr, opt.lrd)
        for idx, data in enumerate(dataLoader):
            img_ref = data['img_ref']
            img_ref = img_ref.cuda()

            # 前向过程(model + loss)开启 autocast
            with autocast():
                recover_img = vae1(img_ref)
                recover_loss = criterion_recover(img_ref, recover_img) / opt.graccbatch_size

            # Scales loss，这是因为半精度的数值范围有限，因此需要用它放大,否则报错
            scaler.scale(recover_loss).backward()
            # recover_loss.backward()

            if (idx + 1) % opt.graccbatch_size == 0:
                torch.nn.utils.clip_grad_norm_(vae1.parameters(), 1.0)
                # optimizer.step()
                # optimizer.zero_grad()
                scaler.step(optimizer)
                # 查看是否要动态调整scaler的大小scaler
                scaler.update()
                optimizer.zero_grad()

            # --------Log Progress--------
            # Determine approximate time left
            batches_done = epoch * len(dataLoader) + idx
            batches_left = opt.s1_epochs * len(dataLoader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            print("[Epoch %d/%d] [Batch %d/%d] [recover_loss: %f] ETA: %s" %
                  (epoch + 1, opt.s1_epochs, idx + 1, len(dataLoader), recover_loss.item(), time_left))

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                vae_sample_images(img_ref, recover_img, batches_done)

            prev_time = time.time()
            # end one batch
        # end one epoch checkpoint
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            torch.save(vae1.state_dict(), os.path.join("result", "models", "vae-%d.ckpt" % epoch))
    # end all epochs, train done
    torch.save(vae1.state_dict(), os.path.join("result", "models", "vae.pth"))


if __name__ == '__main__':
    os.makedirs("result", exist_ok=True)
    os.makedirs(os.path.join("result", "images"), exist_ok=True)
    os.makedirs(os.path.join("result", "models"), exist_ok=True)

    Stage1_Train_VAE()