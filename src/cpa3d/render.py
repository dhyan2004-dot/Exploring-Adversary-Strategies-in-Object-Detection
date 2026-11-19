"""Differentiable rendering of the roof mesh (same model as the CPA PyTorch3D setup: FoV-45 perspective
camera, hard rasterisation, Phong shading with light along the view). Each sampled UAV view
(elevation 60-90 deg, any azimuth, +-10% distance) is rasterised once; rendering a texture is then a
differentiable gather: img=C[face_idx]*mult+spec."""

import math
import os

import numpy as np
import torch
import torch.nn.functional as F

AMBIENT,DIFFUSE,SPECULAR,SHININESS=0.5,0.3,0.2,100.0
FOV_DEG=45.0
NOMINAL_DIST=6.0

EXTENT=1.6

def vertex_normals(verts,faces):
    fn=np.cross(verts[faces[:,1]]-verts[faces[:,0]],verts[faces[:,2]]-verts[faces[:,0]])
    fn[fn[:,2]<0]*=-1 # all facets face up (+z, towards the UAV)
    vn=np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn,faces[:,k],fn)
    return vn/np.linalg.norm(vn,axis=1,keepdims=True)

def camera_basis(elev_deg,azim_deg,roll_deg,dist):
    el,az,ro=map(math.radians,(elev_deg,azim_deg,roll_deg))
    eye=dist*np.array([math.cos(el)*math.cos(az),math.cos(el)*math.sin(az),math.sin(el)])
    fwd=-eye/np.linalg.norm(eye)
    ref=np.array([0.0,0.0,1.0]) if abs(fwd[2])<0.999 else np.array([math.cos(az),math.sin(az),0.0])
    right=np.cross(fwd,ref)
    right/=np.linalg.norm(right)
    up=np.cross(right,fwd)
    xa=np.array([1.0,0.0,0.0]) # align the projected hull axis with canvas +x
    ro=ro+math.atan2(xa@up,xa@right)
    c,s=math.cos(ro),math.sin(ro) # roll about the viewing axis
    right,up=c*right+s*up,-s*right+c*up
    return eye,fwd,right,up

def rasterize_view(verts,faces,vnormals,elev,azim,roll,dist,res,light_jitter=None):
    """Hard rasterisation of one view."""
    eye,fwd,right,up=camera_basis(elev,azim,roll,dist)
    rel=verts-eye
    z=rel@fwd # depth (> 0)
    f=1.0/math.tan(math.radians(FOV_DEG)/2)
    u=(rel@right)*f/z*(NOMINAL_DIST/f)
    v=(rel@up)*f/z*(NOMINAL_DIST/f)
    px=(u/EXTENT+0.5)*res-0.5
    py=(0.5-v/EXTENT)*res-0.5 # image y grows downwards
    face_idx=-np.ones((res,res),np.int64)
    zbuf=np.full((res,res),np.inf)
    bary=np.zeros((res,res,3))
    for fi,(a,b,c) in enumerate(faces):
        xs,ys=px[[a,b,c]],py[[a,b,c]]
        x0,x1=max(int(np.floor(xs.min())),0),min(int(np.ceil(xs.max())),res-1)
        y0,y1=max(int(np.floor(ys.min())),0),min(int(np.ceil(ys.max())),res-1)
        if x0>x1 or y0>y1:
            continue
        gx,gy=np.meshgrid(np.arange(x0,x1+1),np.arange(y0,y1+1))
        den=(ys[1]-ys[2])*(xs[0]-xs[2])+(xs[2]-xs[1])*(ys[0]-ys[2])
        if abs(den)<1e-12:
            continue
        w0=((ys[1]-ys[2])*(gx-xs[2])+(xs[2]-xs[1])*(gy-ys[2]))/den
        w1=((ys[2]-ys[0])*(gx-xs[2])+(xs[0]-xs[2])*(gy-ys[2]))/den
        w2=1-w0-w1
        inside=(w0>=-1e-9)&(w1>=-1e-9)&(w2>=-1e-9)
        if not inside.any():
            continue
        zz=w0*z[a]+w1*z[b]+w2*z[c]
        yy,xx=gy[inside],gx[inside]
        zi=zz[inside]
        closer=zi<zbuf[yy,xx]
        yy,xx,zi=yy[closer],xx[closer],zi[closer]
        zbuf[yy,xx]=zi
        face_idx[yy,xx]=fi
        bary[yy,xx]=np.stack([w0[inside][closer],w1[inside][closer],w2[inside][closer]],1)
    hit=face_idx>=0
    n=np.zeros((res,res,3))
    fv=faces[face_idx[hit]] # [P,3]
    n[hit]=(bary[hit][:,:,None]*vnormals[fv]).sum(1)
    n[hit]/=np.linalg.norm(n[hit],axis=1,keepdims=True)
    l=-fwd if light_jitter is None else light_jitter
    l=l/np.linalg.norm(l)
    ndotl=np.clip(n@l,0,None)
    vdir=-fwd
    r=2*(n@l)[...,None]*n-l
    spec=SPECULAR*np.clip(r@vdir,0,None)**SHININESS*(ndotl>0)
    mult=AMBIENT+DIFFUSE*ndotl
    mult[~hit]=0
    spec[~hit]=0
    uv=np.zeros((res,res,2))
    uv[hit]=(bary[hit][:,:,None]*verts[fv][:,:,:2]).sum(1)
    uv[...,0]=uv[...,0]+0.5
    uv[...,1]=0.5-uv[...,1]
    uv[~hit]=0
    return face_idx,mult.astype(np.float32),spec.astype(np.float32),uv.astype(np.float32)

def sample_view(rng,elev=(60,90),dist_jitter=0.1):
    return dict(elev=float(rng.uniform(*elev)),azim=float(rng.uniform(0,360)),
                roll=0.0, # heading is applied in place.py (patch turns with the hull)
                dist=float(NOMINAL_DIST*rng.uniform(1-dist_jitter,1+dist_jitter)))

class ViewBank:
    """Pre-rasterised camera views."""
    def __init__(self,mesh_npz,n_views=256,res=128,seed=0,elev=(60,90),device="cpu",cache=None):
        m=np.load(mesh_npz)
        self.verts,self.faces=m["verts"].astype(np.float64),m["faces"]
        self.n_faces=len(self.faces)
        self.res=res
        key=dict(n_views=n_views,res=res,seed=seed,elev=list(elev),n_faces=self.n_faces,
                   vsum=float(np.round(self.verts.sum(),6)))
        loaded=False
        if cache is not None:
            try:
                c=np.load(cache,allow_pickle=True)
                if c["key"].item()==key and "uv" in c.files:
                    fi,mu,sp,uv,self.views=c["face_idx"],c["mult"],c["spec"],c["uv"],list(c["views"])
                    loaded=True
            except Exception: # missing / half-written cache -> rebuild
                pass
        if not loaded:
            vn=vertex_normals(self.verts,self.faces)
            rng=np.random.default_rng(seed)
            self.views=[sample_view(rng,elev) for _ in range(n_views)]
            out=[rasterize_view(self.verts,self.faces,vn,res=res,**v) for v in self.views]
            fi,mu,sp,uv=(np.stack([o[j] for o in out]) for j in range(4))
            if cache is not None:
                tmp=str(cache)+".tmp.npz" # atomic: a killed job never leaves a broken cache
                np.savez_compressed(tmp,key=np.array(key,dtype=object),face_idx=fi,mult=mu,spec=sp,uv=uv,
                                    views=np.array(self.views,dtype=object))
                os.replace(tmp,cache)
        self.face_idx=torch.from_numpy(fi).to(device)
        self.mult=torch.from_numpy(mu).to(device)
        self.spec=torch.from_numpy(sp).to(device)
        self.alpha=(self.face_idx>=0).float()
        self.grid=torch.from_numpy(uv).to(device)*2-1 # grid_sample coords for image patches
    def __len__(self):
        return len(self.face_idx)
    def render(self,texture,idx):
        """texture: [3, F] per-facet colours."""
        if texture.dim()==3: # image patch, sampled through the UVs
            tex=F.grid_sample(texture[None],self.grid[idx][None].to(texture.dtype),mode="bilinear",
                                padding_mode="border",align_corners=False)[0]
        else:
            fi=self.face_idx[idx]
            tex=texture[:,fi.clamp(min=0)] # [3,R,R]
        rgb=(tex*self.mult[idx]+self.spec[idx]).clamp(0,1)*self.alpha[idx]
        return rgb,self.alpha[idx]
