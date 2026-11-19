"""AdvCaT-style camouflage (Hu et al. 2023): a few palette colours in Voronoi regions defined by control
points; Gumbel-softmax colour choice during training, hard argmax for testing / printing."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

PALETTES={ # sRGB 0-1, roughly the printed colours of standard patterns
    "woodland":[[0.62,0.60,0.44],[0.30,0.36,0.20],[0.40,0.29,0.19],[0.11,0.11,0.10]],
    "desert":[[0.80,0.72,0.56],[0.66,0.56,0.41],[0.50,0.41,0.30],[0.36,0.30,0.24]],
}

def auto_palette(items,n_colors=4,max_images=200,seed=0):
    """k-means colours of the central half of the tank boxes in the training images."""
    rng=np.random.default_rng(seed)
    sel=[items[i] for i in rng.permutation(len(items))[:max_images]]
    px=[]
    for path,boxes in sel:
        img=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        H,W=img.shape[:2]
        for x0,y0,x1,y1 in boxes:
            cx,cy,w,h=(x0+x1)/2*W,(y0+y1)/2*H,(x1-x0)*W,(y1-y0)*H
            crop=img[int(max(cy-h/4,0)):int(cy+h/4),int(max(cx-w/4,0)):int(cx+w/4)]
            if crop.size:
                px.append(crop.reshape(-1,3)[rng.permutation(crop.shape[0]*crop.shape[1])[:200]])
    px=np.concatenate(px).astype(np.float32)/255
    crit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,50,1e-4)
    _,_,centres=cv2.kmeans(px,n_colors,None,crit,5,cv2.KMEANS_PP_CENTERS)
    return sorted(centres.tolist(),key= lambda c:-sum(c))

class CamoTexture(nn.Module):
    def __init__(self,palette,res=128,points_per_color=15,lam=0.7,blur=1.0,seed=0):
        super().__init__()
        g=torch.Generator().manual_seed(seed)
        C=len(palette)
        self.res,self.lam,self.blur=res,lam,blur
        self.register_buffer("colors",torch.tensor(palette,dtype=torch.float32)) # [C,3]
        self.points=nn.Parameter(torch.rand(C,points_per_color,2,generator=g)) # [C,P,2]
        self.u_train=nn.Parameter(torch.rand(res,res,C,generator=g))
        self.register_buffer("u_fixed",torch.rand(res,res,C,generator=g))
        ys,xs=torch.meshgrid(torch.arange(res),torch.arange(res),indexing="ij")
        self.register_buffer("coords",torch.stack([ys,xs],-1).float()) # [H,W,2]
        k=torch.zeros(C,C,3,3)
        for i in range(C):
            k[i,i]=1/9
        self.register_buffer("kernel",k)
    def prob(self):
        pts=self.points*self.res # [C,P,2]
        d=torch.cdist(self.coords.reshape(-1,2),pts.reshape(-1,2)) # [HW, C*P]
        w=torch.exp(-d/self.blur).reshape(self.res,self.res,*pts.shape[:2]).sum(-1) # [H,W,C]
        p=w/w.sum(-1,keepdim=True).clamp(min=1e-30)
        p=F.conv2d(p.permute(2,0,1)[None],self.kernel,padding=1)[0].permute(1,2,0)
        return p.clamp(min=1e-20)
    def forward(self,tau=0.3,hard=False):
        """-> image [3, H, W] in [0,1]"""
        u=self.lam*self.u_train.clamp(0,1)+(1-self.lam)*self.u_fixed
        g=-torch.log(-torch.log(u+1e-20)+1e-20)
        logits=torch.log(self.prob())+g
        if hard:
            onehot=F.one_hot(logits.argmax(-1),self.colors.shape[0]).float()
        else:
            onehot=F.softmax(logits/tau,-1)
        return (onehot@self.colors).permute(2,0,1)
    def ctrl_loss(self,sigma_frac=0.08):
        """their ctrl_loss: penalise same-colour control points that bunch together"""
        pts=self.points*self.res
        diff=pts[:,:,None,:]-pts[:,None,:,:]
        d2=(diff**2).sum(-1)
        sigma=sigma_frac*self.res
        return torch.exp(-d2/sigma**2).mean()-1/pts.shape[1]
    @torch.no_grad()
    def clip_(self):
        self.points.data.clamp_(0,1)
        self.u_train.data.clamp_(0,1)
