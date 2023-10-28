# -*- coding: utf-8 -*-
# @Time    : 2023/6/23 12:25
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12


import torch

class Resnet(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.time = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.torch.nn.Linear(1280, dim_out),
            torch.nn.Unflatten(dim=1, unflattened_size=(dim_out, 1, 1)),
        )

        self.s0 = torch.nn.Sequential(
            torch.torch.nn.GroupNorm(num_groups=32, num_channels=dim_in, eps=1e-05, affine=True),
            torch.nn.SiLU(),
            torch.torch.nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
        )

        self.s1 = torch.nn.Sequential(
            torch.torch.nn.GroupNorm(num_groups=32, num_channels=dim_out, eps=1e-05, affine=True),
            torch.nn.SiLU(),
            torch.torch.nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
        )

        self.res = None
        if dim_in != dim_out:
            self.res = torch.torch.nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x, time):
        # x -> [1, 320, 48, 64]
        # time -> [1, 1280]

        res = x

        # [1, 1280] -> [1, 320, 1, 1]
        time = self.time(time)

        # [1, dim_in, 48, 64] -> [1, dim_out, 48, 64]
        s0_out = self.s0(x) + time

        # 维度不变
        # [1, dim_out, 48, 64]
        s1_out = self.s1(s0_out)

        # [1, 320, 48, 64] -> [1, 320, 48, 64]
        if self.res:
            res = self.res(res)

        # 维度不变
        # [1, 320, 48, 64]
        s1_out = res + s1_out

        return s1_out


class DownBlock(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.res0 = Resnet(dim_in, dim_out)
        self.res1 = Resnet(dim_out, dim_out)

        self.out = torch.nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=2, padding=1)

    def forward(self, out_vae, time):
        outs = []

        out_vae = self.res0(out_vae, time)
        outs.append(out_vae)

        out_vae = self.res1(out_vae, time)
        outs.append(out_vae)

        out_vae = self.out(out_vae)
        outs.append(out_vae)

        return out_vae, outs


class UpBlock(torch.nn.Module):

    def __init__(self, dim_in, dim_out, dim_prev, add_up):
        super().__init__()

        self.res0 = Resnet(dim_out + dim_prev, dim_out)
        self.res1 = Resnet(dim_out + dim_out, dim_out)
        self.res2 = Resnet(dim_in + dim_out, dim_out)

        self.out = None
        if add_up:
            self.out = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='nearest'),
                torch.nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1),
            )

    def forward(self, out_vae, time, out_down):
        out_vae = self.res0(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.res1(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.res2(torch.cat([out_vae, out_down.pop()], dim=1), time)

        if self.out:
            out_vae = self.out(out_vae)

        return out_vae


class UNet(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # --------------in_proj------------
        # [B, 4, 48, 64]->[B, 320, 48, 64]
        self.vae_latent_proj = torch.nn.Conv2d(4, 320, kernel_size=3, padding=1)
        
        # [1, 320] -> [1, 1280]
        self.in_time = torch.nn.Sequential(
            torch.nn.Linear(320, 1280),
            torch.nn.SiLU(),
            torch.nn.Linear(1280, 1280),
        )

        # ------------down-------------
        self.down_block0 = DownBlock(320, 320)
        self.down_block1 = DownBlock(320, 640)
        self.down_block2 = DownBlock(640, 1280)

        self.down_res0 = Resnet(1280, 1280)
        self.down_res1 = Resnet(1280, 1280)

        # -------------mid-------------
        self.mid_res0 = Resnet(1280, 1280)
        self.mid_res1 = Resnet(1280, 1280)

        # -------------up--------------
        self.up_res0 = Resnet(2560, 1280)
        self.up_res1 = Resnet(2560, 1280)
        self.up_res2 = Resnet(2560, 1280)

        self.up_in = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.Conv2d(1280, 1280, kernel_size=3, padding=1),
        )

        self.up_block0 = UpBlock(640, 1280, 1280, True)
        self.up_block1 = UpBlock(320, 640, 1280, True)
        self.up_block2 = UpBlock(320, 320, 640, False)
        # -------------out------------
        # [1, 320, 48, 64] -> [1, 4, 48, 64]
        self.out = torch.nn.Sequential(
            torch.nn.GroupNorm(num_channels=320, num_groups=32, eps=1e-5),
            torch.nn.SiLU(),
            torch.nn.Conv2d(320, 4, kernel_size=3, padding=1),
        )

    @staticmethod
    def get_time_embed(t):
        # -9.210340371976184 = -math.log(10000)
        e = torch.arange(160) * -9.210340371976184 / 160
        e = e.exp().to(t.device) * t

        # [160+160] -> [320] -> [1, 320]
        e = torch.cat([e.cos(), e.sin()]).unsqueeze(dim=0)

        return e

    def forward(self, out_vae, time_step):
        # out_vae -> [1, 4, 48, 64]
        # time -> [1]

        # ------------in--------------
        # [1, 4, 48, 64] -> [1, 320, 48, 64]
        out_vae = self.vae_latent_proj(out_vae)
       
        # [1] -> [1, 320]
        time_emb = self.get_time_embed(time_step)
        # [1, 320] -> [1, 1280]
        time_emb = self.in_time(time_emb)

        # ------------down-------------
        out_down = [out_vae]

        # [1, 320, 48, 64],[1, 1280] -> out_vae [1, 320, 24, 32]
        #  -> out [1, 320, 48, 64],[1, 320, 48, 64][1, 320, 24, 32]
        out_vae, out = self.down_block0(out_vae=out_vae, time=time_emb)
        out_down.extend(out)

        # [1, 320, 24, 32],[1, 1280] -> out_vae [1, 640, 12, 16]
        #  -> out [1, 640, 24, 32],[1, 640, 24, 32],[1, 640, 12, 16]
        out_vae, out = self.down_block1(out_vae=out_vae, time=time_emb)
        out_down.extend(out)

        # [1, 640, 12, 16],[1, 1280] -> out_vae [1, 1280, 6, 8]
        # out -> [1, 1280, 12, 16],[1, 1280, 12, 16],[1, 1280, 6, 8]
        out_vae, out = self.down_block2(out_vae=out_vae, time=time_emb)
        out_down.extend(out)

        # [1, 1280, 6, 8],[1, 1280] -> [1, 1280, 6, 8]
        out_vae = self.down_res0(out_vae, time_emb)
        out_down.append(out_vae)
        # print("down_block2:", out_vae.shape)

        # [1, 1280, 6, 8],[1, 1280] -> [1, 1280, 6, 8]
        out_vae = self.down_res1(out_vae, time_emb)
        out_down.append(out_vae)

        # -------------mid-------------
        # [1, 1280, 6, 8],[1, 1280] -> [1, 1280, 6, 8]
        out_vae = self.mid_res0(out_vae, time_emb)
        # [1, 1280, 6, 8] -> [1, 1280, 6, 8]

        # [1, 1280, 6, 8],[1, 1280] -> [1, 1280, 6, 8]
        out_vae = self.mid_res1(out_vae, time_emb)
        # print("mid_res1:", out_vae.shape)

        # -------------up--------------
        # [1, 1280+1280, 6, 8],[1, 1280] -> [1, 1280, 6, 8]
        out_vae = self.up_res0(torch.cat([out_vae, out_down.pop()], dim=1), time_emb)

        # [1, 1280+1280, 6, 8],[1, 1280] -> [1, 1280, 6, 8]
        out_vae = self.up_res1(torch.cat([out_vae, out_down.pop()], dim=1), time_emb)

        # [1, 1280+1280, 6, 8],[1, 1280] -> [1, 1280, 6, 8]
        out_vae = self.up_res2(torch.cat([out_vae, out_down.pop()], dim=1), time_emb)

        # [1, 1280, 6, 8] -> [1, 1280, 12, 16]
        out_vae = self.up_in(out_vae)

        # [1, 1280, 12, 16],[1, 1280] -> [1, 1280, 24, 32]
        # out_down -> [1, 640, 16, 16],[1, 1280, 16, 16],[1, 1280, 12, 16]
        out_vae = self.up_block0(out_vae=out_vae, time=time_emb, out_down=out_down)

        # [1, 1280, 24, 32],[1, 1280] -> [1, 640, 48, 64]
        # out_down -> [1, 320, 24, 32],[1, 640, 32, 32],[1, 640, 24, 32]
        out_vae = self.up_block1(out_vae=out_vae, time=time_emb, out_down=out_down)

        # [1, 640, 48, 64],[1, 1280] -> [1, 320, 48, 64]
        # out_down -> [1, 320, 48, 64],[1, 320, 48, 64],[1, 320, 48, 64]
        out_vae = self.up_block2(out_vae=out_vae, time=time_emb, out_down=out_down)

        # -------------out------------
        # [1, 320, 48, 64] -> [1, 4, 48, 64]
        noise_pred = self.out(out_vae)
        # print("unet out:", noise_pred.shape)
        return noise_pred


if __name__ == '__main__':
    print("unet test:")
    unet1 = UNet().cuda()
    
    vae_feat = torch.randn(1, 4, 48, 64).cuda()
    noise_step = torch.randint(0, 1000, (1,)).long().cuda()
    
    out = unet1(vae_feat, noise_step)
    print(out.shape)
