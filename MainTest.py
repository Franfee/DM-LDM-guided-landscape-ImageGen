# -*- coding: utf-8 -*-
# @Time    : 2023/6/22 11:02
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12

from MainTrain import vae_sample_images
from utils.MyDataLoader import *
from nets.UNet import UNet
from nets.VAE import VAE

dataLoader = get_dataloader()
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(DEVICE)


def Test_VAE():
    os.makedirs(os.path.join("result", "images", "vae"), exist_ok=True)

    vae_clone = VAE()
    vae_clone.cuda()

    batches_done = 0
    # --------------------------------------------
    vae_clone.load_state_dict(torch.load(os.path.join("result", "models", "vae-0.ckpt"), map_location=DEVICE))
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


def Test_UNet():
    UNet_clone = UNet()


if __name__ == '__main__':
    Test_VAE()
