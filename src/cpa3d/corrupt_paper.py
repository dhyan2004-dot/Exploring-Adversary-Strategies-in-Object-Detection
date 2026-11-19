"""Corruptions of the CPA paper ported from its code: sun-flare light spot and HLS shadow (training, p=0.5),
rain / mud spatter (test only), and the global affine (+-10 deg, scale 0.9-1.1)."""

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

def flare_layer(H,W,center,src_radius,color=(1.0,1.0,1.0)):
    out=np.zeros((H,W,3),np.float32)
    overlay=out.copy()
    num=max(int(src_radius)//2,1)
    alpha=np.linspace(0.0,0.8,num=num)
    rad=np.linspace(5,max(src_radius,5),num=num)
    pt=(int(center[0]),int(center[1]))
    for i in range(num):
        cv2.ellipse(overlay,pt,(int(rad[i]*1.2),int(rad[i]*0.9)),60,0,360,color,-1)
        alp=alpha[num-i-1]**3
        out=cv2.addWeighted(overlay,alp,out,1-alp,0)
    return out

def flare_params(patch_mask,rng):
    """Centre on a random patch pixel, radius 0.6 * min side of the patch's min-area rectangle."""
    ys,xs=np.nonzero(patch_mask)
    if len(xs)<3:
        return None
    k=rng.integers(len(xs))
    (_,_),(rw,rh),_=cv2.minAreaRect(np.stack([xs,ys],1).astype(np.float32))
    return (int(xs[k]),int(ys[k])),int(min(rw,rh)*0.6)

def add_flare(img,params):
    """img [3,H,W] tensor; additive, differentiable (identity Jacobian)."""
    if params is None:
        return img
    (cx,cy),r=params
    layer=flare_layer(img.shape[1],img.shape[2],(cx,cy),r)
    return (img+torch.from_numpy(layer).permute(2,0,1).to(img)).clamp(0,1)

def shadow_params(box,rng,dimension=4):
    x0,y0,x1,y1=[int(round(v)) for v in box]
    return np.array([[rng.integers(x0,max(x1,x0+1)+1),rng.integers(y0,max(y1,y0+1)+1)]
                     for _ in range(dimension)],np.int32)

def shadow_numpy(img_np,polys):
    """img_np HxWx3 float32 in [0,1]."""
    hls=cv2.cvtColor(img_np,cv2.COLOR_RGB2HLS)
    mask=np.zeros(img_np.shape[:2],np.uint8)
    for p in polys:
        cv2.fillPoly(mask,[p],255)
    hls[...,1][mask==255]*=0.5
    return cv2.cvtColor(hls,cv2.COLOR_HLS2RGB)

def add_shadow(img,polys):
    """Straight-through, as in the CPA code: forward = shadowed image, backward = identity."""
    if not polys:
        return img
    np_img=img.detach().permute(1,2,0).cpu().numpy().astype(np.float32)
    sh=torch.from_numpy(shadow_numpy(np_img,polys)).permute(2,0,1).to(img)
    return img-img.detach()+sh

def spatter(img_np,mode,rng,mean=0.65,std=0.3,sigma=2,cutout_threshold=0.7,intensity=0.6):
    """albumentations Spatter re-implementation."""
    H,W=img_np.shape[:2]
    liquid=rng.normal(mean,std,(H,W)).astype(np.float32)
    liquid=cv2.GaussianBlur(liquid,(0,0),sigma)
    liquid[liquid<cutout_threshold]=0
    if mode=="rain":
        l8=(liquid*255).clip(0,255).astype(np.uint8)
        dist=255-cv2.Canny(l8,50,150)
        dist=cv2.distanceTransform(dist,cv2.DIST_L2,5)
        _,dist=cv2.threshold(dist,20,20,cv2.THRESH_TRUNC)
        dist=cv2.blur(dist,(3,3)).astype(np.uint8)
        dist=cv2.equalizeHist(dist)
        ker=np.array([[-2,-1,0],[-1,1,1],[0,1,2]],np.float32)
        dist=cv2.filter2D(dist,cv2.CV_8U,ker)
        dist=cv2.blur(dist,(3,3)).astype(np.float32)
        m=liquid*dist
        m/=max(m.max(),1e-6)
        drops=m[...,None]*np.array([238,238,175],np.float32)/255*intensity
        return np.clip(img_np+drops,0,1)
    m=(liquid>cutout_threshold).astype(np.float32)
    m=cv2.GaussianBlur(m,(0,0),sigma)
    m[m<1.2*cutout_threshold]=0
    m=m[...,None]
    mud=m*np.array([20,42,63],np.float32)/255
    return np.clip(img_np*(1-m)+mud,0,1)

def add_spatter(img,mode,rng):
    np_img=img.detach().permute(1,2,0).cpu().numpy().astype(np.float32)
    return torch.from_numpy(spatter(np_img,mode,rng)).permute(2,0,1).to(img)

def random_affine(imgs,boxes_list,rng,degrees=10.0,scale=(0.9,1.1)):
    """Rotate/scale every image of the batch about its centre (myRandomAffine)."""
    B,_,H,W=imgs.shape
    out_imgs,out_boxes=[],[]
    for b in range(B):
        ang=math.radians(rng.uniform(-degrees,degrees))
        sc=rng.uniform(*scale)
        c,s=math.cos(ang),math.sin(ang)
        theta=torch.tensor([[c/sc,s/sc*H/W,0.0],
                              [-s/sc*W/H,c/sc,0.0]],dtype=imgs.dtype,device=imgs.device)
        grid=F.affine_grid(theta[None],(1,3,H,W),align_corners=False)
        out_imgs.append(F.grid_sample(imgs[b:b+1],grid,align_corners=False)[0])
        nb=[]
        for (x0,y0,x1,y1) in boxes_list[b]:
            pts=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],np.float64)-[W/2,H/2]
            fwd=np.array([[c,-s],[s,c]])*sc # input -> output
            q=pts@fwd.T+[W/2,H/2]
            nb.append((max(q[:,0].min(),0),max(q[:,1].min(),0),min(q[:,0].max(),W),min(q[:,1].max(),H)))
        out_boxes.append(nb)
    return torch.stack(out_imgs),out_boxes
