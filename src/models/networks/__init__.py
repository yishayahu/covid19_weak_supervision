from . import fcn8_vgg16, fcn8_vgg16_multiscale, unet2d, unet_resnet, fcn8_resnet, deeplab
from . import resnet_seam, infnet

from torchvision import models
import torch, os
import torch.nn as nn
import segmentation_models_pytorch


def get_network(network_name, n_classes, exp_dict):
    if network_name == 'infnet':
        model_base = infnet.InfNet(n_classes=1, loss=exp_dict['model']['loss'])

    elif network_name == 'fcn8_vgg16':
        model_base = fcn8_vgg16.FCN8VGG16(n_classes=n_classes)

    if network_name == "fcn8_vgg16_multiscale":
        model_base = fcn8_vgg16_multiscale.FCN8VGG16(n_classes=n_classes)

    if network_name == "unet_resnet":
        model_base = unet_resnet.ResNetUNet(n_class=n_classes)
    
    if network_name == "resnet_seam":
        model_base = resnet_seam.ResNetSeam()
        path_base = '/mnt/datasets/public/issam/seam'
        model_base.load_state_dict(torch.load(os.path.join(path_base, 'resnet38_SEAM.pth')))
    else:
        model_base = segmentation_models_pytorch.Unet(
            encoder_name=network_name,
            encoder_weights="imagenet",
            classes=n_classes,
            aux_params ={'classes':5}
        )

    return model_base

