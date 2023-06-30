# -*- coding: utf-8 -*-
# @Time    : 2023/6/22 11:02
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12
from torchvision.utils import make_grid, save_image

from MainTrain import vae_sample_images
from utils.DenoisingDiffusion import GaussianDiffusion
from utils.MyDataLoader import *
from nets.UNet import UNet
from nets.VAE import VAE


DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(DEVICE)


def Test_VAE():
    dataLoader = get_dataloader()

    os.makedirs(os.path.join("result", "images", "vae"), exist_ok=True)

    vae_clone = VAE()
    vae_clone.cuda()

    batches_done = 0
    # --------------------------------------------
    vae_clone.load_state_dict(torch.load(os.path.join("result", "models", "vae.pth"), map_location=DEVICE))
    vae_clone.eval()
    for idx, data in enumerate(dataLoader):
        img_ref, _img_msk = data['img_ref'], data['img_msk']
        img_ref = img_ref.cuda()
        print("generate:", idx)
        recover_img = vae_clone(img_ref)

        vae_sample_images(img_ref, recover_img, batches_done, dir_output="images/vae")
        batches_done += 1

        if idx >= 20:
            break
    # -------------------------------------------
    vae_clone.load_state_dict(torch.load(os.path.join("result", "debug", "vae-break-406929.ckpt"), map_location=DEVICE))
    vae_clone.eval()
    for idx, data in enumerate(dataLoader):
        img_ref, _img_msk = data['img_ref'], data['img_msk']
        # img_ref = img_ref.cuda()
        print("generate:", idx)
        recover_img = vae_clone(img_ref)

        vae_sample_images(img_ref, recover_img, batches_done)
        batches_done += 1

        if idx >= 20:
            break


def Test():
    test_loader = get_test_dataloader()
    noise_helper = GaussianDiffusion()
    noise_helper.eval()

    vae_clone = VAE()
    UNet_clone = UNet()
    vae_clone.cuda()
    UNet_clone.cuda()
    vae_clone.load_state_dict(torch.load(os.path.join("result", "models", "vae.pth"), map_location=DEVICE))
    UNet_clone.load_state_dict(torch.load(os.path.join("result", "models", "unet.pth"), map_location=DEVICE))
    vae_clone.eval()
    UNet_clone.eval()
    for idx, data in enumerate(test_loader):
        img_ref, img_msk, save_name = data['img_ref'], data['img_msk'], data['save_name']
        img_ref = img_ref.cuda()
        img_msk = img_msk.cuda()

        # unet不从完全的噪声中预测,从混合大量噪声的参考图中采样
        low_steps, high_steps = 750, 850
        noise_step = torch.randint(low_steps, high_steps, (opt.batch_size,)).long()
        noise_step = noise_step.cuda()

        # 风格ref图像, vae_latent_space特征图
        vae_out = vae_clone.encoder(img_ref)
        vae_out = vae_clone.sample(vae_out)
        ref_vae_out, _noise = noise_helper(vae_out, noise_step)

        latent_gen = noise_helper.ddim_sample(model=UNet_clone, shape=vae_out.size(),
                                              mask_condition=img_msk, img=ref_vae_out)
        img_gen = vae_clone.decoder(latent_gen)
        img_gen = make_grid(img_gen, nrow=1, normalize=True)
        save_name = os.path.join("result", "images", save_name[0])
        save_image(img_gen, save_name, normalize=False)


if __name__ == '__main__':
    # Test_VAE()
    Test()


