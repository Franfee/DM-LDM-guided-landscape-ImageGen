# -*- coding: utf-8 -*-
# @Time    : 2023/6/22 17:09
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12
import argparse


# load which ckpt while interrupt training
LOAD_CHECK_POINT_VAE = True
LOAD_VAE_IDX = "15"
LOAD_CHECK_POINT_UNET = False
LOAD_UNET_IDX = "15"


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1start_epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--s1_epochs", type=int, default=100, help="number of epochs of training")
    
    parser.add_argument("--s2start_epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--s2_epochs", type=int, default=100, help="number of epochs of training")

    parser.add_argument("--data_root", type=str, default="./datasets/train_resized", help="datasets path")
    parser.add_argument("--batch_size", type=int, default=3, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=384, help="size of image height")
    parser.add_argument("--img_width", type=int, default=512, help="size of image width")

    parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--lrd', type=float, default=0.0001, help='learning rate decay for epoch, default=0.0001')

    parser.add_argument("--sample_interval", type=int, default=100, help="interval of sampling images from generators")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval of model checkpoints")

    opt = parser.parse_args()
    print(opt)
    return opt
