# -*- coding: utf-8 -*-
# @Time    : 2023/6/22 16:54
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12
import json
import os
import glob


import numpy as np
import torch
from PIL import Image

from torch.utils.data.distributed import DistributedSampler     # 多卡数据采样
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as f

from utils.get_all_parsar import opt


class MyImageDataset(Dataset):
    def __init__(self, root):
        self.transform = transforms.Compose(
            [
                transforms.Resize((opt.img_height, opt.img_width), f.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        # glob: Return a list of paths matching a pathname pattern.
        self.img_refs = sorted(glob.glob(os.path.join(root, "imgs") + "/*.jpg"))
        self.img_msks = sorted(glob.glob(os.path.join(root, "labels") + "/*.png"))

    def __getitem__(self, index):
        imgs_ref = Image.open(self.img_refs[index % len(self.img_refs)]).convert("RGB")
        imgs_msk = Image.open(self.img_msks[index % len(self.img_msks)]).convert("L")

        if np.random.random() < 0.5:
            np_img_A = np.array(imgs_ref)[:, ::-1, :]
            np_img_B = np.array(imgs_msk)[:, ::-1]

            imgs_ref = Image.fromarray(np_img_A, "RGB")
            imgs_msk = np_img_B
            # imgs_msk = Image.fromarray(np_img_B, "L")

        imgs_ref = self.transform(imgs_ref)
        # imgs_msk = self.transform(imgs_msk)
        imgs_msk = torch.unsqueeze(torch.from_numpy(np.array(imgs_msk).astype(np.float32) / 28.0), dim=0)

        return {"img_ref": imgs_ref, "img_msk": imgs_msk}

    def __len__(self):
        return len(self.img_refs)


class TestImageDataset(Dataset):
    def __init__(self, root):
        def get_json():
            filePath = os.path.join("datasets", "label_to_img.json")
            with open(filePath, 'r', encoding='utf8') as load_f:
                loadDict = json.load(load_f)
            return loadDict

        self.transform = transforms.Compose(
            [
                transforms.Resize((opt.img_height, opt.img_width), f.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ]
        )
        self.jsonFile = get_json()
        self.img_refs = []
        self.img_msks = []
        self.save_name = []
        for msk, ref in self.jsonFile.items():
            self.img_refs.append(os.path.join(root, "imgs", ref))
            self.img_msks.append(os.path.join(root, "..", "val_A_labels_resized", msk))
            self.save_name.append(str(msk).replace("png", "jpg"))

    def __getitem__(self, index):
        imgs_ref = Image.open(self.img_refs[index % len(self.img_refs)]).convert("RGB")
        imgs_msk = Image.open(self.img_msks[index % len(self.img_msks)]).convert("L")
        save_name = self.save_name[index % len(self.save_name)]

        if np.random.random() < 0.5:
            np_img_A = np.array(imgs_ref)[:, ::-1, :]
            np_img_B = np.array(imgs_msk)[:, ::-1]

            imgs_ref = Image.fromarray(np_img_A, "RGB")
            imgs_msk = np_img_B
            # imgs_msk = Image.fromarray(np_img_B, "L")

        imgs_ref = self.transform(imgs_ref)
        # imgs_msk = self.transform(imgs_msk)
        imgs_msk = torch.unsqueeze(torch.from_numpy(np.array(imgs_msk).astype(np.float32) / 28.0), dim=0)

        return {"img_ref": imgs_ref, "img_msk": imgs_msk, "save_name": save_name}

    def __len__(self):
        return len(self.img_refs)


def get_test_dataloader():
    test_dataloader = DataLoader(
        TestImageDataset(opt.data_root),
        batch_size=opt.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=opt.n_cpu,
        drop_last=False,
    )

    return test_dataloader


def get_dataloader(g=None, muti=True):
    if muti:
        data = MyImageDataset(opt.data_root)
        train_ddp_sampler = DistributedSampler(data, shuffle=True)
        train_dataloader = DataLoader(
            data,
            batch_size=opt.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=opt.n_cpu,
            drop_last=False,
            sampler=train_ddp_sampler,
            generator=g
        )
    else:
        train_dataloader = DataLoader(
            MyImageDataset(opt.data_root),
            data,
            batch_size=opt.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=opt.n_cpu,
            drop_last=False
        )

    return train_dataloader


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     loader = get_dataloader()
#     from torchvision.utils import make_grid, save_image
#
#     for i, batch in enumerate(loader):
#         img_ref = batch['img_ref']
#         img_msk = batch['img_msk']
#         img_ref = img_ref.cuda()
#         img_msk = img_msk.cuda()
#
#         img_msk = torch.cat((img_msk*3, img_msk*3, img_msk*3), dim=1)
#         print(img_msk.shape)
#         print(img_ref.shape)
#         # plt.show
#         # img_msk = img_msk.clamp(0, 1)
#         # img_msk = img_msk.permute(0, 2, 3, 1)
#         # plt.imshow(img_msk.numpy()[0])
#         # plt.show()
#         image_grid = make_grid(img_msk, nrow=1, normalize=False)
#         save_image(image_grid, os.path.join("..", "result", "test_%d.png" % i), normalize=False)
#         if i == 10:
#             break
