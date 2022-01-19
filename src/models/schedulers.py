import torch


def get_scheduler(cfg,opt):
    lambda1 = lambda epoch: cfg['factor'] ** epoch
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1)