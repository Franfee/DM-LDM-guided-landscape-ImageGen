# -*- coding: utf-8 -*-
# @Time    : 2023/2/24 15:23
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12


def exp_lr_scheduler(optimizer, epoch, init_lr, lrd):
    """Implements torch learning rate decay """
    lr = init_lr / (1 + epoch * lrd)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
