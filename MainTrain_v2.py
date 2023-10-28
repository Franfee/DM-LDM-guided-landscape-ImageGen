# -*- coding: utf-8 -*-
# @Time    : 2023/6/22 11:02
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12
import datetime
import time
import os


# 指定显卡可见性
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import torch
# 多卡
import torch.distributed as dist    # 多卡通讯
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP    # 模型传递
from torch.nn import SyncBatchNorm  # BN层同步

# 自动混合精度
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

# 制图
from torchvision.utils import make_grid, save_image

from nets.UNet_v2 import UNet
from nets.VAE import VAE

from utils.lr_scheduler import exp_lr_scheduler
from utils.get_all_parsar import *
from utils.DenoisingDiffusion import GaussianDiffusion
from utils.my_gc import torch_gc
print(opt)

def init_ddp(local_rank):
    """
    转换device时只需要.cuda,否则使用.cuda(local_rank)
    对进程初始化，使用nccl协议后端通信，env作为初始化方法
    """
    torch.cuda.set_device(local_rank)
    os.environ['RANK'] = str(local_rank)
    dist.init_process_group(backend='nccl',init_method='env://')


def get_ddp_generator(seed=1234):
    """
    对每个进程使用不同种子
    """
    local_rank = dist.get_rank()
    g = torch.Generator()
    g.manual_seed(seed+local_rank)
    return g


def reduce_tensor(tensor):
    """
    多个进程计算结果汇总
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= dist.get_world_size()
    return rt


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
    from utils.MyDataLoader import get_dataloader
    dataLoader = get_dataloader(muti=False)
    vae1 = VAE()

    if LOAD_CHECK_POINT_VAE:
        vae1.load_state_dict(torch.load(os.path.join("result", "models", f"vae-{LOAD_VAE_IDX}.ckpt"), map_location='cpu'))
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


def Stage2_Train_UNet(local_rank, args):
    print("In Stage2_Train_UNet...")
    # 多卡部分设置
    init_ddp(local_rank)

    from utils.MyDataLoader import get_dataloader
    dataLoader = get_dataloader(g=get_ddp_generator())
    # -------------------------

    vae1 = VAE()
    vae1.eval()
    noise_helper = GaussianDiffusion()
    noise_helper.eval()

    unet1 = UNet()

    vae1.load_state_dict(torch.load(os.path.join("result", "models", "vae.ckpt"), map_location='cpu'))
    
    if LOAD_CHECK_POINT_UNET:
        unet1.load_state_dict(torch.load(os.path.join("result", "models", f"unet-{LOAD_UNET_IDX}.ckpt"), map_location='cpu'))
    else:
        pass

    optimizer = torch.optim.AdamW(unet1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=0.01, eps=1e-8)
    criterion_pred = torch.nn.MSELoss()
    # GradScaler对象用于自动混合精度
    scaler = GradScaler()

    # --------GPU-----------
    vae1.cuda()
    noise_helper.cuda()
    unet1.cuda()
    criterion_pred.cuda()

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        vae1 = SyncBatchNorm.convert_sync_batchnorm(vae1)  # BN层同步
        vae1 = DDP(vae1, device_ids=[local_rank], output_device=local_rank) # 多卡通信
        
        unet1 = SyncBatchNorm.convert_sync_batchnorm(unet1)  # BN层同步
        unet1 = DDP(unet1, device_ids=[local_rank], output_device=local_rank) # 多卡通信
        
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
            vae_out = vae1.module.encoder(img_ref)
            vae_out = vae1.module.sample(vae_out)
            # 0.18215 = vae.config.scaling_factor
            vae_out = vae_out * 0.18215

            # 往vae_out隐空间中添加噪声
            noise_step = torch.randint(0, 1000, (opt.batch_size,)).long()
            noise_step = noise_step.cuda()
            x_noised, noise = noise_helper(vae_out, noise_step)

            # 前向过程(model + loss)开启 autocast
            with autocast():
                # 根据mask语义信息,把特征图中的噪声计算出来
                # noise_pred = unet1(*(x_noised, noise_step))
                noise_pred = unet1(x_noised, noise_step)

                # 计算mse loss [1, 4, 64, 64],[1, 4, 64, 64]
                pred_loss = criterion_pred(noise_pred, noise) / opt.graccbatch_size

            # 多卡部分 多个loss求平均
            reduce_loss = reduce_tensor(pred_loss.data)

            # pred_loss.backward()
            # Scales loss，这是因为半精度的数值范围有限，因此需要用它放大,否则报错
            scaler.scale(pred_loss).backward()

            if (idx + 1) % opt.graccbatch_size == 0:
                torch.nn.utils.clip_grad_norm_(unet1.parameters(), 1.0)
                # optimizer.step()
                # optimizer.zero_grad()
                scaler.step(optimizer)
                # 查看是否要动态调整scaler的大小scaler
                scaler.update()
                optimizer.zero_grad()

            # 多卡
            if dist.get_rank() == 0:
                # --------Log Progress--------
                # Determine approximate time left
                batches_done = epoch * len(dataLoader) + idx
                batches_left = opt.s2_epochs * len(dataLoader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                
                
                print("[Epoch %d/%d] [Batch %d/%d] [pred_loss: %f] ETA: %s" %
                    (epoch + 1, opt.s2_epochs, idx + 1, len(dataLoader), reduce_loss.item(), time_left))

                # If at sample interval save image
                if batches_done % opt.sample_interval == 0:
                    # ddim阶段 unet从完全的噪声中预测
                    latent_gen = noise_helper.ddim_sample(model=unet1, shape=vae_out.size())
                    # 从压缩图恢复成图片
                    vae_seed = 1 / 0.18215 * latent_gen
                    # vae_seed = latent_gen
                    # [1, 4, 64, 64] -> [1, 3, 512, 512]
                    img_gen = vae1.module.decoder(vae_seed)
                    # 保存照片
                    unet_sample_images(img_ref=img_ref, img_msk=img_msk, img_gen=img_gen, batches_done=batches_done)
                torch_gc()
                prev_time = time.time()
                # end one batch
        # end one epoch checkpoint
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0 and dist.get_rank() == 0:
            torch.save(unet1.state_dict(), os.path.join("result", "models", "unet-%d.ckpt" % epoch))
    
    if dist.get_rank() == 0:
        # end all epochs, train done
        torch.save(unet1.state_dict(), os.path.join("result", "models", "unet.pth"))

    # 多卡消除进程组
    dist.destroy_process_group()


if __name__ == '__main__':
    os.makedirs("result", exist_ok=True)
    os.makedirs(os.path.join("result", "images"), exist_ok=True)
    os.makedirs(os.path.join("result", "models"), exist_ok=True)

    # Stage1_Train_VAE()

    # 多卡部分
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12289'

    world_size = torch.cuda.device_count()
    print("multi GPUs:", world_size)
    os.environ['WORLD_SIZE'] = str(world_size)

    mp.spawn(fn=Stage2_Train_UNet, args=(None,), nprocs=world_size)
    # Stage2_Train_UNet()
