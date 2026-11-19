"""Contact sheet for checking placement: rendered sheet, both hull estimates and the label box, optionally
with the training (light, shadow) and test (rain, mud) corruptions."""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from common import list_images_with_target,load_data_cfg
from corrupt_paper import add_flare,add_shadow,add_spatter,flare_params,shadow_params
from place import composite,hull_from_box
from render import ViewBank

def crop_around(img,box,pad=0.25,size=220):
    x0,y0,x1,y1=box
    w,h=x1-x0,y1-y0
    X0,Y0=int(max(x0-pad*w,0)),int(max(y0-pad*h,0))
    X1,Y1=int(min(x1+pad*w,img.shape[1])),int(min(y1+pad*h,img.shape[0]))
    c=img[Y0:Y1,X0:X1]
    s=size/max(c.shape[:2])
    return cv2.resize(c,(int(c.shape[1]*s),int(c.shape[0]*s))),(X0,Y0,s)

def draw_hull(img,box,aspect):
    cx,cy,Wv,L,th=hull_from_box(box,aspect)
    for sg in (1,-1):
        rect=((cx,cy),(L,Wv),math.degrees(sg*th))
        cv2.drawContours(img,[cv2.boxPoints(rect).astype(np.int32)],0,(255,255,0),1)
    x0,y0,x1,y1=map(int,box)
    cv2.rectangle(img,(x0,y0),(x1,y1),(0,255,255),1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--split",default="val")
    ap.add_argument("--target-class",default="Tank")
    ap.add_argument("--texture",default="",help="cpa_train dir or texture .npy (default: random)")
    ap.add_argument("--mesh",default=str(Path(__file__).with_name("aux_mesh.npz")))
    ap.add_argument("--n",type=int,default=12,help="number of tanks to show")
    ap.add_argument("--max-files-per-photo",type=int,default=1)
    ap.add_argument("--patch-m",type=float,default=1.0)
    ap.add_argument("--hull-width-m",type=float,default=3.6)
    ap.add_argument("--aspect",type=float,default=1.95)
    ap.add_argument("--corruptions",action="store_true")
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--out",default="preview.jpg")
    a=ap.parse_args()
    cfg,names,tid=load_data_cfg(a.data,a.target_class)
    bank=ViewBank(a.mesh,32,128,seed=a.seed)
    if a.texture:
        p=Path(a.texture)
        tex=torch.from_numpy(np.load(p/"texture.npy" if p.is_dir() else p)).float()
    else:
        tex=torch.from_numpy(np.random.default_rng(a.seed).random((3,bank.n_faces))).float()
    rng=np.random.default_rng(a.seed)
    tiles=[]
    for path,boxes_n in list_images_with_target(cfg,a.split,tid,a.max_files_per_photo):
        rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        H,W=rgb.shape[:2]
        base=torch.from_numpy(rgb).permute(2,0,1).float()/255
        for bn in boxes_n:
            if len(tiles)>=a.n*(5 if a.corruptions else 1):
                break
            g=(bn[0]*W,bn[1]*H,bn[2]*W,bn[3]*H)
            r_,al=bank.render(tex,int(rng.integers(len(bank))))
            img,m=composite(base,g,r_,al,a.patch_m,a.hull_width_m,a.aspect,sign=int(rng.choice([-1,1])))
            row=[("patch",img)]
            if a.corruptions:
                row+=[("light",add_flare(img,flare_params(m.numpy(),rng))),
                        ("shadow",add_shadow(img,[shadow_params(g,rng)])),
                        ("rain",add_spatter(img,"rain",rng)),("mud",add_spatter(img,"mud",rng))]
            for name,im in row:
                arr=(im.clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8).copy()
                if name=="patch":
                    draw_hull(arr,g,a.aspect)
                t,_=crop_around(arr,g)
                canvas=np.full((240,240,3),255,np.uint8)
                canvas[20:20+t.shape[0],10:10+t.shape[1]]=t[:220,:220]
                cv2.putText(canvas,f"{name}  {path.stem[:14]}",(4,14),0,0.4,(0,0,0),1)
                tiles.append(canvas)
    cols=5 if a.corruptions else 6
    while len(tiles)%cols:
        tiles.append(np.full((240,240,3),255,np.uint8))
    sheet=np.concatenate([np.concatenate(tiles[i:i+cols],1) for i in range(0,len(tiles),cols)],0)
    cv2.imwrite(a.out,cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR))
    print(f"wrote {a.out} ({len(tiles)} tiles)")

if __name__=="__main__":
    main()
