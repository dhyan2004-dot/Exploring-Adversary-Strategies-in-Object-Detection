"""BigGAN-128 generator used by the NPA paper (Hu et al., ICCV 2021, github.com/aiiu-lab/Naturalistic-
Adversarial-Patch): Brock et al.'s BigGAN-PyTorch, ch96, ImageNet 128x128, EMA weights at 138k
iterations ("G_ema.pth" in the authors' pretrained_biggan.tar). Code copied from their
GANLatentDiscovery/models/BigGAN (MIT licence, see LICENSE) with only the imports changed."""

import json
from pathlib import Path

import torch
from torch import nn

from.import BigGAN,utils

HERE=Path(__file__).resolve().parent

class ConditionedBigGAN(nn.Module):
    """G(z) for a fixed ImageNet class."""
    def __init__(self,big_gan,target_class):
        super().__init__()
        self.big_gan=big_gan
        self.register_buffer("target_class",torch.tensor(int(target_class),dtype=torch.int64))
        self.dim_z=big_gan.dim_z
    def forward(self,z):
        cls=self.target_class.repeat(z.shape[0])
        return self.big_gan(z,self.big_gan.shared(cls))

def make_big_gan(weights,target_class,device="cpu"):
    config=json.load(open(HERE/"generator_config.json"))
    config["resolution"]=utils.imsize_dict[config["dataset"]]
    config["n_classes"]=utils.nclass_dict[config["dataset"]]
    config["G_activation"]=utils.activation_dict[config["G_nl"]]
    config["D_activation"]=utils.activation_dict[config["D_nl"]]
    config["skip_init"]=True
    config["no_optim"]=True
    G=BigGAN.Generator(**config)
    if weights:
        G.load_state_dict(torch.load(weights,map_location="cpu"),strict=True)
    else:
        print("WARNING: no BigGAN weights given -- random generator (tests only)")
    G=ConditionedBigGAN(G,target_class).to(device).eval()
    for p in G.parameters():
        p.requires_grad_(False)
    return G
