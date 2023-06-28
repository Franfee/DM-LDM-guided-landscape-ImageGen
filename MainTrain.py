# -*- coding: utf-8 -*-
# @Time    : 2023/6/22 11:02
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12
import datetime
import time

from torchvision.utils import make_grid, save_image

from nets.UNet import UNet
from nets.VAE import VAE
from utils.MyDataLoader import *
from utils.DenoisingDiffusion import GaussianDiffusion

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

LOAD_CHECK_POINT_VAE = False
LOAD_CHECK_POINT_UNET = False


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
        vae1.load_state_dict(torch.load(os.path.join("result", "models", "vae.ckpt"), map_location=DEVICE))
    else:
        # init
        pass

    train_lr = 1e-4
    adam_betas = (0.9, 0.999)
    optimizer = torch.optim.AdamW(vae1.parameters(), lr=train_lr, betas=adam_betas, weight_decay=0.01, eps=1e-8)
    criterion_recover = torch.nn.L1Loss()

    # --------GPU-----------
    vae1.cuda()
    criterion_recover.cuda()

    # --------train---------
    prev_time = time.time()
    for epoch in range(opt.start_epoch, opt.n_epochs):
        vae1.train()
        for idx, data in enumerate(dataLoader):
            img_ref = data['img_ref']
            img_ref = img_ref.cuda()

            recover_img = vae1(img_ref)

            recover_loss = criterion_recover(img_ref, recover_img) / 4
            # -----------debug---------------
            if torch.any(torch.isnan(recover_loss)):
                os.makedirs(os.path.join("result", "debug"), exist_ok=True)
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                vae_sample_images(img_ref, recover_img, idx + 100000 * epoch, dir_output="debug")
                torch.save(vae1.state_dict(), os.path.join("result", "debug",
                                                           "vae-break-%d.ckpt" % (idx + 100000 * epoch)))
                return
            # ---------------------------
            recover_loss.backward()

            if (idx + 1) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(vae1.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            # --------Log Progress--------
            # Determine approximate time left
            batches_done = epoch * len(dataLoader) + idx
            batches_left = opt.n_epochs * len(dataLoader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            print("[Epoch %d/%d] [Batch %d/%d] [recover_loss: %f] ETA: %s" %
                  (epoch + 1, opt.n_epochs, idx + 1, len(dataLoader), recover_loss.item(), time_left))

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

    if LOAD_CHECK_POINT_VAE:
        vae1.load_state_dict(torch.load(os.path.join("result", "models", "vae.ckpt"), map_location=DEVICE))
    else:
        pass

    if LOAD_CHECK_POINT_UNET:
        unet1.load_state_dict(torch.load(os.path.join("result", "models", "unet.ckpt"), map_location=DEVICE))
    else:
        pass

    train_lr = 1e-4
    adam_betas = (0.8, 0.999)
    optimizer = torch.optim.AdamW(unet1.parameters(), lr=train_lr, betas=adam_betas, weight_decay=0.01, eps=1e-8)
    criterion_l2 = torch.nn.MSELoss()

    # --------GPU-----------
    vae1.cuda()
    noise_helper.cuda()
    unet1.cuda()
    criterion_l2.cuda()

    # --------train---------
    prev_time = time.time()
    for epoch in range(opt.start_epoch, opt.n_epochs):
        unet1.train()
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

            # 根据mask语义信息,把特征图中的噪声计算出来
            noise_pred = unet1(x_noised, img_msk, noise_step)

            # 计算mse loss [1, 4, 64, 64],[1, 4, 64, 64]
            pred_loss = criterion_l2(noise_pred, noise) / 4
            pred_loss.backward()

            if (idx + 1) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(unet1.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            # --------Log Progress--------
            # Determine approximate time left
            batches_done = epoch * len(dataLoader) + idx
            batches_left = opt.n_epochs * len(dataLoader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            print("[Epoch %d/%d] [Batch %d/%d] [pred_loss: %f] ETA: %s" %
                  (epoch + 1, opt.n_epochs, idx + 1, len(dataLoader), pred_loss.item(), time_left))

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                # unet不从完全的噪声中预测,从混合大量噪声的参考图中采样
                low_steps, high_steps = 750, 850
                noise_step = torch.randint(low_steps, high_steps, (opt.batch_size,)).long()
                noise_step = noise_step.cuda()

                ref_vae_out, _noise = noise_helper(vae_out, noise_step)
                latent_gen = noise_helper.ddim_sample(model=unet1, shape=vae_out.size(),
                                                      mask_condition=img_msk, img=ref_vae_out)
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


def Stage3_Train_Com():
    # ---------------------------
    vae1 = VAE()
    unet1 = UNet()
    noise_helper = GaussianDiffusion()
    noise_helper.eval()

    # -------------------------------
    if LOAD_CHECK_POINT_VAE:
        vae1.load_state_dict(torch.load(os.path.join("result", "models", "vae.ckpt"), map_location=DEVICE))
    else:
        pass

    if LOAD_CHECK_POINT_UNET:
        unet1.load_state_dict(torch.load(os.path.join("result", "models", "unet.ckpt"), map_location=DEVICE))
    else:
        pass

    # -----------------------------------
    optimizer_vae = torch.optim.AdamW(vae1.parameters(), lr=1e-5, betas=(0.5, 0.999), weight_decay=0.01, eps=1e-8)
    optimizer_unet = torch.optim.AdamW(unet1.parameters(), lr=1e-5, betas=(0.5, 0.999), weight_decay=0.01, eps=1e-8)
    criterion_l1 = torch.nn.L1Loss()
    criterion_l2 = torch.nn.MSELoss()

    # -----------------------------------
    vae1.cuda()
    unet1.cuda()
    noise_helper.cuda()
    criterion_l1.cuda()
    criterion_l2.cuda()

    # -------------------train--------------------
    prev_time = time.time()
    for epoch in range(opt.start_epoch, opt.n_epochs):
        vae1.train()
        unet1.train()
        for idx, data in enumerate(dataLoader):
            img_ref, img_msk = data['img_ref'], data['img_msk']
            img_ref = img_ref.cuda()
            img_msk = img_msk.cuda()

            # 风格ref图像, vae_latent_space特征图
            vae_out = vae1.encoder(img_ref)
            vae_out = vae1.sample(vae_out)

            # ------part of training vae-------
            img_rec = vae1.decoder(vae_out)
            recover_loss = criterion_l1(img_ref, img_rec)
            recover_loss.backward()
            # -----------debug---------------
            if torch.any(torch.isnan(recover_loss)):
                os.makedirs(os.path.join("result", "debug"), exist_ok=True)
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                vae_sample_images(img_ref, img_rec, idx + 100000 * epoch, dir_output="debug")
                torch.save(vae1.state_dict(), os.path.join("result", "debug", "vae-break-%d.ckpt" % (idx + 100000 * epoch)))
                return
            # ------------------------------
            # ---------------------------------

            # ---------detach for training unet-------------
            vae_out = vae_out.detach().cpu().numpy()
            vae_out = torch.from_numpy(vae_out).cuda()
            # 0.18215 = vae.config.scaling_factor
            vae_out = vae_out * 0.18215

            # 往vae_out隐空间中添加噪声
            noise_step = torch.randint(0, 1000, (opt.batch_size,)).long()
            noise_step = noise_step.cuda()
            x_noised, noise = noise_helper(vae_out, noise_step)

            # 根据mask语义信息,把特征图中的噪声计算出来
            noise_pred = unet1(x_noised, img_msk, noise_step)

            # 计算mse loss [1, 4, 64, 64],[1, 4, 64, 64]
            pred_loss = criterion_l2(noise_pred, noise)
            pred_loss.backward()

            # 优化器
            torch.nn.utils.clip_grad_norm_(vae1.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(unet1.parameters(), 1.0)
            optimizer_vae.step()
            optimizer_unet.step()

            optimizer_vae.zero_grad()
            optimizer_unet.zero_grad()

            # --------Log Progress--------
            # Determine approximate time left
            batches_done = epoch * len(dataLoader) + idx
            batches_left = opt.n_epochs * len(dataLoader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            print("[Epoch %d/%d] [Batch %d/%d] [recover_loss: %f] [pred_loss: %f] ETA: %s" %
                  (epoch + 1, opt.n_epochs, idx + 1, len(dataLoader), recover_loss.item(), pred_loss.item(), time_left))

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                # unet不从完全的噪声中预测,从混合大量噪声的参考图中采样
                low_steps, high_steps = 750, 850
                noise_step = torch.randint(low_steps, high_steps, (opt.batch_size,)).long()
                noise_step = noise_step.cuda()

                ref_vae_out, _noise = noise_helper(vae_out, noise_step)
                latent_gen = noise_helper.ddim_sample(model=unet1, shape=vae_out.size(),
                                                      mask_condition=img_msk, img=ref_vae_out)
                # 从压缩图恢复成图片
                vae_seed = 1 / 0.18215 * latent_gen
                # [1, 4, 64, 64] -> [1, 3, 512, 512]
                img_gen = vae1.decoder(vae_seed)
                # 保存照片
                unet_sample_images(img_ref=img_ref, img_msk=img_msk, img_gen=img_gen, batches_done=batches_done)
                vae_sample_images(img_ref, img_rec, batches_done)

            prev_time = time.time()
            # end one batch
        # end one epoch checkpoint
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            torch.save(unet1.state_dict(), os.path.join("result", "models", "unet-%d.ckpt" % epoch))
            torch.save(vae1.state_dict(), os.path.join("result", "models", "vae-%d.ckpt" % epoch))
    # end all epochs, train done
    torch.save(unet1.state_dict(), os.path.join("result", "models", "unet.pth"))
    torch.save(vae1.state_dict(), os.path.join("result", "models", "vae.pth"))


if __name__ == '__main__':
    os.makedirs("result", exist_ok=True)
    os.makedirs(os.path.join("result", "images"), exist_ok=True)
    os.makedirs(os.path.join("result", "models"), exist_ok=True)

    # Stage1_Train_VAE()
    # Stage2_Train_UNet()
    Stage3_Train_Com()
