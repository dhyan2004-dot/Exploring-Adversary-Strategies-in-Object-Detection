"""Composite the rendered sheet on a real tank at a printable size (default 1 m, from the hull width
estimated from the axis-aligned box) and hard-clip it to the region inside both possible hull outlines."""

import math

import torch
import torch.nn.functional as F

from render import EXTENT

def hull_from_box(box,aspect=1.95):
    """box xyxy (px) -> (cx, cy, Wv, L, theta_rad) of the hull rectangle (theta = long-axis angle)."""
    x0,y0,x1,y1=box
    w,h=x1-x0,y1-y0
    cx,cy=(x0+x1)/2,(y0+y1)/2
    r=w/max(h,1e-6)
    A=aspect
    if r>=A:
        th,Wv=0.0,h
    elif r<=1/A:
        th,Wv=math.pi/2,w
    else:
        th=math.atan((A-r)/(r*A-1))
        Wv=w/(A*math.cos(th)+math.sin(th))
    return cx,cy,Wv,A*Wv,th

def hull_mask(gx,gy,cx,cy,Wv,L,th,margin):
    """1 where the pixel lies inside the hull rectangle for BOTH signs of theta."""
    m=torch.ones_like(gx)
    for sgn in (1,-1):
        c,s=math.cos(sgn*th),math.sin(sgn*th)
        u=(gx-cx)*c+(gy-cy)*s
        v=-(gx-cx)*s+(gy-cy)*c
        m=m*((u.abs()<=L/2-margin)&(v.abs()<=Wv/2-margin)).float()
    return m

def patch_side_px(box,patch_m=1.0,hull_width_m=3.6,aspect=1.95):
    return patch_m/hull_width_m*hull_from_box(box,aspect)[2]

def composite(image,box,rgb,alpha,patch_m=1.0,hull_width_m=3.6,aspect=1.95,margin_frac=0.05,
              jitter=(0.0,0.0),sign=1,angle_jitter_deg=0.0):
    """image [3,H,W] in [0,1]; box xyxy px; rgb [3,R,R], alpha [R,R] from ViewBank.render."""
    _,H,W=image.shape
    cx,cy,Wv,L,th=hull_from_box(box,aspect)
    s=patch_m/hull_width_m*Wv
    margin=margin_frac*Wv
    phi=sign*th+math.radians(angle_jitter_deg)
    mx=max(0.0,(L/2-margin)-s/2-Wv*0.0)
    my=max(0.0,(Wv/2-margin)-s/2)
    ju,jv=max(-mx,min(mx,jitter[0]*Wv)),max(-my,min(my,jitter[1]*Wv))
    pcx=cx+ju*math.cos(phi)-jv*math.sin(phi)
    pcy=cy+ju*math.sin(phi)+jv*math.cos(phi)
    half=EXTENT*s/2*math.sqrt(2) # window covers any rotation
    wx0,wy0=max(int(pcx-half)-1,0),max(int(pcy-half)-1,0)
    wx1,wy1=min(int(pcx+half)+2,W),min(int(pcy+half)+2,H)
    empty=torch.zeros(H,W,device=image.device)
    if wx1<=wx0 or wy1<=wy0 or s<2:
        return image,empty
    ys=torch.arange(wy0,wy1,device=image.device,dtype=torch.float32)+0.5
    xs=torch.arange(wx0,wx1,device=image.device,dtype=torch.float32)+0.5
    gy,gx=torch.meshgrid(ys,xs,indexing="ij")
    c,sn=math.cos(phi),math.sin(phi)
    u=(gx-pcx)*c+(gy-pcy)*sn # pixel -> patch axes
    v=-(gx-pcx)*sn+(gy-pcy)*c
    cs=EXTENT*s/2 # canvas half-size in px
    grid=torch.stack([u/cs,v/cs],-1)[None]
    src=torch.cat([rgb,alpha[None]],0)[None]
    smp=F.grid_sample(src,grid,mode="bilinear",padding_mode="zeros",align_corners=False)[0]
    a=smp[3]
    col=(smp[:3]/a.clamp(min=1e-6)).clamp(0,1) # un-premultiply
    a=(a>0.5).float() # crisp printed edge
    a=a*hull_mask(gx,gy,cx,cy,Wv,L,th,margin) # HARD in-vehicle clip
    out=image.clone()
    out[:,wy0:wy1,wx0:wx1]=out[:,wy0:wy1,wx0:wx1]*(1-a)+col*a
    mask=empty.clone()
    mask[wy0:wy1,wx0:wx1]=a
    return out,mask
