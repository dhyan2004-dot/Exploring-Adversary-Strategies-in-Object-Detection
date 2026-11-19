"""CPA texture generator (Generator_V2 of the CPA code): MLP 100->256->256->256->3F, BatchNorm, LeakyReLU,
tanh(0.2x)*0.5. Texture=mean over 32 noise samples +0.5 (+N(0,0.1) during training)."""

import torch
from torch import nn

class GeneratorV2(nn.Module):
    def __init__(self,n_faces,latent_dim=100,width=256):
        super().__init__()
        self.n_faces=n_faces
        self.latent_dim=latent_dim
        def block(i,o):
            return [nn.Linear(i,o),nn.BatchNorm1d(o),nn.LeakyReLU(0.2,inplace=True)]
        self.model=nn.Sequential(*block(latent_dim,width),*block(width,width),*block(width,width),
                                   nn.Linear(width,3*n_faces))
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0.0,0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm1d):
                nn.init.normal_(m.weight,1.0,0.02)
                nn.init.zeros_(m.bias)
    def forward(self,z):
        return (torch.tanh(0.2*self.model(z))*0.5).view(-1,3,self.n_faces)

def texture(gen,n_noise=32,seed=None,add_noise=0.1,device="cpu"):
    g=torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    z=torch.randn(n_noise,gen.latent_dim,generator=g).to(device)
    t=gen(z).mean(0)+0.5
    if add_noise:
        t=t+torch.randn(t.shape,generator=g).to(device)*add_noise
    return t.clamp(1e-6,0.99999)
