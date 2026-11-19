"""Differentiable patch corruptions for the 2-D stage: light spot, shadow, rain, mud.
Each is an alpha blend P<-P(1-aM)+aMc with a NumPy mask M, so gradients flow to the patch."""

import numpy as np
from PIL import Image,ImageDraw,ImageFilter

KINDS=["light","shadow","rain","mud"]

def _grid(size):
    ys,xs=np.mgrid[0:size,0:size].astype(np.float32)
    return xs/size,ys/size # normalised 0..1

def light_spot(size,rng):
    """Soft Gaussian glare blob that washes the patch toward warm white."""
    xs,ys=_grid(size)
    cx,cy=rng.uniform(0.0,1.0,2)
    sigma=rng.uniform(0.15,0.4)
    m=np.exp(-(((xs-cx)**2+(ys-cy)**2)/(2*sigma**2)))
    return m.astype(np.float32),(1.0,0.98,0.9),float(rng.uniform(0.4,0.9))

def shadow(size,rng):
    """Soft-edged half-plane shadow darkening one side of the patch."""
    xs,ys=_grid(size)
    theta=rng.uniform(0,2*np.pi)
    nx,ny=np.cos(theta),np.sin(theta)
    cx,cy=rng.uniform(0.3,0.7,2)
    softness=rng.uniform(0.03,0.12)
    d=(xs-cx)*nx+(ys-cy)*ny
    m=1.0/(1.0+np.exp(-d/softness))
    return m.astype(np.float32),(0.0,0.0,0.0),float(rng.uniform(0.3,0.7))

def rain(size,rng):
    """Thin, slightly tilted pale streaks."""
    canvas=Image.new("L",(size,size),0)
    draw=ImageDraw.Draw(canvas)
    tilt=rng.uniform(-0.3,0.3)
    for _ in range(int(rng.integers(25,80))):
        x0,y0=rng.uniform(0,size,2)
        length=rng.uniform(0.1,0.3)*size
        draw.line([(x0,y0),(x0+tilt*length,y0+length)],
                  fill=int(rng.uniform(150,255)),width=1 if size<200 else 2)
    canvas=canvas.filter(ImageFilter.GaussianBlur(0.6))
    m=np.clip(np.asarray(canvas,dtype=np.float32)/255.0*1.5,0.0,1.0)
    return m,(0.8,0.85,0.92),float(rng.uniform(0.5,0.85))

def mud(size,rng):
    """Irregular opaque brown blotches covering roughly 20-45% of the patch."""
    low=(rng.random((8,8))*255).astype(np.uint8)
    n=np.asarray(Image.fromarray(low).resize((size,size),Image.BICUBIC),dtype=np.float32)/255.0
    thr=np.quantile(n,rng.uniform(0.55,0.8))
    m=np.clip((n-thr)/0.08,0.0,1.0)
    return m.astype(np.float32),(0.35,0.25,0.15),float(rng.uniform(0.7,0.95))

_FUNCS={"light":light_spot,"shadow":shadow,"rain":rain,"mud":mud}

def sample_corruption(size,rng,kinds,prob=1.0,min_layers=1,max_layers=2):
    """Draw a list of (mask, colour, alpha) layers; empty list = leave the patch clean."""
    kinds=list(kinds)
    if not kinds or rng.random()>=prob:
        return []
    hi=min(max_layers,len(kinds))
    lo=min(min_layers,hi)
    n=int(rng.integers(lo,hi+1))
    chosen=rng.choice(kinds,size=n,replace=False)
    return [_FUNCS[str(k)](size,rng) for k in chosen]

def apply_layers(tile,layers):
    """Alpha-blend each layer onto `tile` ([3,S,S], values 0..1)."""
    is_torch= not isinstance(tile,np.ndarray)
    if is_torch:
        import torch
    for mask,color,alpha in layers:
        if is_torch:
            w=alpha*torch.as_tensor(mask,dtype=tile.dtype,device=tile.device)
            c=torch.as_tensor(color,dtype=tile.dtype,device=tile.device).view(3,1,1)
        else:
            w=alpha*mask
            c=np.asarray(color,dtype=tile.dtype).reshape(3,1,1)
        tile=tile*(1-w)+c*w
    return tile

def layers_for_condition(condition,size,rng):
    """Named evaluation conditions: none | light | shadow | rain | mud | mixed."""
    if condition=="none":
        return []
    if condition=="mixed":
        return sample_corruption(size,rng,KINDS,prob=1.0,min_layers=2,max_layers=2)
    return sample_corruption(size,rng,[condition],prob=1.0)
