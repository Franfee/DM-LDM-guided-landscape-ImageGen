# -*- coding: utf-8 -*-
# @Time    : 2023/6/22 11:02
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12
import datetime
import time

# 自动混合精度
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from torchvision.utils import make_grid, save_image

from nets.UNet import UNet
from nets.VAE import VAE

from utils.lr_scheduler import exp_lr_scheduler
from utils.MyDataLoader import *
from utils.DenoisingDiffusion import GaussianDiffusion
from utils.get_all_parsar import LOAD_CHECK_POINT_VAE, LOAD_CHECK_POINT_UNET, LOAD_VAE_IDX, LOAD_UNET_IDX


DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


dataLoader = get_dataloader()


def vae_sample_images(img_ref, img_rec, batches_done, dir_output="images"):
    """Saves a generated sample"""
    b = img_ref.size()[0]

    # Arrange images along x-axis
    img_ref = make_grid(img_ref, nrow=b, normalize=True)
    img_rec = make_grid(img_rec, nrow=b, normalize=True)
    # Arrange images along y-axis
    image_grid = torch.cat((img_ref, img_rec), 1)
    save_image(image_grid, os.path.join("result", dir_output, "vae-step_%d.png" % batches_done), normalize=False)


def unet_sample_images(img_ref, img_msk, img_gen, batches_done, dir_output="images"):
    """Saves a generated sample"""
    b = img_ref.size()[0]
    img_msk = torch.cat((img_msk * 3, img_msk * 3, img_msk * 3), dim=1)

    # Arrange images along x-axis
    img_ref = make_grid(img_ref, nrow=b, normalize=True)
    img_msk = make_grid(img_msk, nrow=b, normalize=True)
    img_gen = make_grid(img_gen, nrow=b, normalize=True)

    # Arrange images along y-axis
    image_grid = torch.cat((img_ref, img_msk, img_gen), 1)
    save_image(image_grid, os.path.join("result", dir_output, "unet-step_%d.png" % batches_done), normalize=False)


def Stage1_Train_VAE():
    vae1 = VAE()

    if LOAD_CHECK_POINT_VAE:
        vae1.load_state_dict(torch.load(os.path.join("result", "models", f"vae-{LOAD_VAE_IDX}.ckpt"), map_location=DEVICE))
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


def Stage2_Train_UNet():
    vae1 = VAE()
    vae1.eval()
    noise_helper = GaussianDiffusion()
    noise_helper.eval()

    unet1 = UNet()

    vae1.load_state_dict(torch.load(os.path.join("result", "models", "vae.ckpt"), map_location=DEVICE))
    
    if LOAD_CHECK_POINT_UNET:
        unet1.load_state_dict(torch.load(os.path.join("result", "models", f"unet-{LOAD_UNET_IDX}.ckpt"), map_location=DEVICE))
    else:
        pass

    optimizer = torch.optim.AdamW(unet1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=0.01, eps=1e-8)
    criterion_l1 = torch.nn.L1Loss()
    # GradScaler对象用于自动混合精度
    scaler = GradScaler()

    # --------GPU-----------
    vae1.cuda()
    noise_helper.cuda()
    unet1.cuda()
    criterion_l1.cuda()

    # --------train---------
    prev_time = time.time()
    for epoch in range(opt.s2start_epoch, opt.s2_epochs):
        unet1.train()
        # adjust learning rate
        optimizer = exp_lr_scheduler(optimizer, epoch, opt.lr, opt.lrd)
        for idx, data in enumerate(dataLoader):
            img_ref, img_msk = data['img_ref'], data['img_msk']
            img_ref = img_ref.cuda()
            img_msk = img_msk.cuda()

            # 风格ref图像, vae_latent_space特征图
            vae_out = vae1.encoder(img_ref)
            vae_out = vae1.sample(vae_out)
            # 0.18215 = vae.config.scaling_factor
            vae_out = vae_out * 0.18215

            # 往vae_out隐空间中添加噪声
            noise_step = torch.randint(0, 1000, (opt.batch_size,)).long()
            noise_step = noise_step.cuda()
            x_noised, noise = noise_helper(vae_out, noise_step)

            # 前向过程(model + loss)开启 autocast
            with autocast():
                # 根据mask语义信息,把特征图中的噪声计算出来
                noise_pred = unet1(x_noised, img_msk, noise_step)

                # 计算mse loss [1, 4, 64, 64],[1, 4, 64, 64]
                pred_loss = criterion_l1(noise_pred, noise) / 4

            # pred_loss.backward()
            # Scales loss，这是因为半精度的数值范围有限，因此需要用它放大,否则报错
            scaler.scale(pred_loss).backward()

            # if (idx + 1) % 4 == 0:
            torch.nn.utils.clip_grad_norm_(unet1.parameters(), 1.0)
            # optimizer.step()
            # optimizer.zero_grad()
            scaler.step(optimizer)
            # 查看是否要动态调整scaler的大小scaler
            scaler.update()
            optimizer.zero_grad()

            # --------Log Progress--------
            # Determine approximate time left
            batches_done = epoch * len(dataLoader) + idx
            batches_left = opt.s2_epochs * len(dataLoader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            print("[Epoch %d/%d] [Batch %d/%d] [pred_loss: %f] ETA: %s" %
                  (epoch + 1, opt.s2_epochs, idx + 1, len(dataLoader), pred_loss.item(), time_left))

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                # ddim阶段 unet从完全的噪声中预测
                latent_gen = noise_helper.ddim_sample(model=unet1, shape=vae_out.size(), mask_condition=img_msk)
                # 从压缩图恢复成图片
                vae_seed = 1 / 0.18215 * latent_gen
                # [1, 4, 64, 64] -> [1, 3, 512, 512]
                img_gen = vae1.decoder(vae_seed)
                # 保存照片
                unet_sample_images(img_ref=img_ref, img_msk=img_msk, img_gen=img_gen, batches_done=batches_done)

            prev_time = time.time()
            # end one batch
        # end one epoch checkpoint
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            torch.save(unet1.state_dict(), os.path.join("result", "models", "unet-%d.ckpt" % epoch))
    # end all epochs, train done
    torch.save(unet1.state_dict(), os.path.join("result", "models", "unet.pth"))


if __name__ == '__main__':
    os.makedirs("result", exist_ok=True)
    os.makedirs(os.path.join("result", "images"), exist_ok=True)
    os.makedirs(os.path.join("result", "models"), exist_ok=True)

    Stage1_Train_VAE()
    Stage2_Train_UNet()
