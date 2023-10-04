# -*- coding: utf-8 -*-
# @Time    : 2023/6/22 11:02
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12


from torchvision.utils import make_grid, save_image

from utils.DenoisingDiffusion import GaussianDiffusion
from utils.MyDataLoader import *
from nets.UNet import UNet
from nets.VAE import VAE


DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(DEVICE)


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

        # 风格ref图像, vae_latent_space特征图
        vae_out = vae_clone.encoder(img_ref)
        vae_out = vae_clone.sample(vae_out)

        latent_gen = noise_helper.ddim_sample(model=UNet_clone, shape=vae_out.size(), mask_condition=img_msk)
        img_gen = vae_clone.decoder(latent_gen)
        img_gen = make_grid(img_gen, nrow=1, normalize=True)
        save_name = os.path.join("result", "images", save_name[0])
        save_image(img_gen, save_name, normalize=False)


if __name__ == '__main__':
    Test()


