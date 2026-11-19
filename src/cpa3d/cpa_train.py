"""3-D CPA on real UAV photos (Zhang et al. 2024). Per step: generator texture -> render each tank from a
random UAV view -> place a 1 m sheet on the hull -> global affine -> light / shadow corruptions (p=0.5)
-> loss=5*sum(per-tank score)+5*TV3d. Adam 0.01, 10 epochs, SWA from epoch 6.
    python cpa_train.py --data <data.yaml> --weights <best.pt> --corrupt light,shadow --out <dir>"""

import argparse
import json
import zlib
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel

from common import (cpa_instance_score,list_images_with_target,load_data_cfg,load_image_and_boxes,
                    raw_scores,training_imgsz)
from corrupt_paper import add_flare,add_shadow,flare_params,random_affine,shadow_params
from generator import GeneratorV2,texture
from place import composite
from render import ViewBank

def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data",required=True,help="fold data.yaml")
    p.add_argument("--weights",required=True,help="surrogate detector (YOLOv8s fold best.pt)")
    p.add_argument("--target-class",default="Tank")
    p.add_argument("--split",default="train")
    p.add_argument("--corrupt",default="light,shadow",help="'' | light | shadow | light,shadow")
    p.add_argument("--corrupt-prob",type=float,default=0.5)
    p.add_argument("--mesh",default=str(Path(__file__).with_name("aux_mesh.npz")))
    p.add_argument("--n-views",type=int,default=256)
    p.add_argument("--view-res",type=int,default=128)
    p.add_argument("--elev",type=float,nargs=2,default=(60,90))
    p.add_argument("--patch-m",type=float,default=1.0,help="printed patch side in metres")
    p.add_argument("--hull-width-m",type=float,default=3.6)
    p.add_argument("--aspect",type=float,default=1.95,help="hull length / width")
    p.add_argument("--margin-frac",type=float,default=0.05)
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--batch",type=int,default=2)
    p.add_argument("--lr",type=float,default=0.01)
    p.add_argument("--swa-start",type=int,default=6,help="0-based epoch from which SWA is updated")
    p.add_argument("--det-weight",type=float,default=5.0)
    p.add_argument("--tv-weight",type=float,default=5.0)
    p.add_argument("--score-thresh",type=float,default=0.05)
    p.add_argument("--max-files-per-photo",type=int,default=0,help="0 = all augmented copies")
    p.add_argument("--imgsz",type=int,default=0,help="0 = detector's training imgsz")
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--device",default="")
    p.add_argument("--out",required=True)
    p.add_argument("--max-steps",type=int,default=0,help="debug: stop each epoch after N steps")
    p.add_argument("--fixed-views",action="store_true",
                   help="diversity ablation: one fixed camera view + heading per object for the whole training")
    p.add_argument("--no-affine",action="store_true",help="diversity ablation: switch off the +-10 deg affine")
    return p.parse_args()

class TV3d:
    """Paper Eq."""
    def __init__(self,mesh_npz,device):
        m=np.load(mesh_npz)
        self.pairs=torch.from_numpy(m["adj_pairs"]).to(device)
        self.len=torch.from_numpy(m["adj_len"]).to(device)
        self.n=len(m["faces"])
    def __call__(self,tex):
        d=(tex[:,self.pairs[:,0]]-tex[:,self.pairs[:,1]]).abs()
        return (self.len[None]*d).sum()/self.n

def printable(tex,mesh_npz,res=620):
    """Flat top view of the sheet."""
    m=np.load(mesh_npz)
    v,f=m["verts"],m["faces"]
    t=(tex.detach().cpu().numpy().T*255).round().astype(np.uint8)
    img=np.full((res,res,3),255,np.uint8)
    xy=np.stack([(v[:,0]+0.5),(0.5-v[:,1])],1)*(res-1) # mesh +y is image-up, as rendered
    for k,tri in enumerate(f):
        pts=np.stack([xy[tri,0],xy[tri,1]],1).round().astype(np.int32)
        cv2.fillPoly(img,[pts],tuple(int(c) for c in t[k]))
    return img

def build_batch(items,tex,bank,args,rng,device,imgsz,jitter=None):
    imgs,boxes_all,masks=[],[],[]
    for path,boxes_n in items:
        img,boxes=load_image_and_boxes(path,boxes_n,imgsz)
        img=img.to(device)
        m_tot=torch.zeros(img.shape[1:],device=device)
        for bi,b in enumerate(boxes):
            if getattr(args,"fixed_views",False):
                h=zlib.crc32(f"{Path(path).name}#{bi}".encode())
                vi,sg=h%len(bank),(1 if (h>>16)&1 else -1)
            else:
                vi,sg=int(rng.integers(len(bank))),int(rng.choice([-1,1]))
            rgb,a=bank.render(jitter(tex) if jitter else tex,vi)
            img,m=composite(img,b,rgb,a,args.patch_m,args.hull_width_m,args.aspect,
                               args.margin_frac,sign=sg)
            m_tot=torch.maximum(m_tot,m)
        imgs.append(img)
        boxes_all.append(boxes)
        masks.append(m_tot)
    return torch.stack(imgs),boxes_all,torch.stack(masks)

def apply_affine(imgs,masks,boxes,rng):
    x=torch.cat([imgs,masks[:,None]],1)
    x,boxes=random_affine(x,boxes,rng)
    return x[:,:3],(x[:,3]>0.5),boxes

def corrupt(imgs,masks,boxes,kinds,prob,rng):
    out=[]
    for b in range(len(imgs)):
        im=imgs[b]
        if "light" in kinds and rng.random()<prob:
            im=add_flare(im,flare_params(masks[b].cpu().numpy(),rng))
        if "shadow" in kinds and rng.random()<prob and boxes[b]:
            bx=boxes[b][int(rng.integers(len(boxes[b])))]
            im=add_shadow(im,[shadow_params(bx,rng)])
        out.append(im)
    return torch.stack(out)

def save_example(path,imgs,boxes):
    im=(imgs[0].detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8).copy()
    for x0,y0,x1,y1 in boxes[0]:
        cv2.rectangle(im,(int(x0),int(y0)),(int(x1),int(y1)),(255,255,0),1)
    cv2.imwrite(str(path),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))

def main():
    a=parse_args()
    kinds=[k.strip() for k in a.corrupt.split(",") if k.strip()]
    assert all(k in ("light","shadow") for k in kinds),"training corruptions are light / shadow"
    device=a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    out=Path(a.out);(out/"examples").mkdir(parents=True,exist_ok=True)
    cfg,names,tid=load_data_cfg(a.data,a.target_class)
    imgsz=a.imgsz or training_imgsz(a.weights) or 960
    from common import load_detector
    model=load_detector(a.weights).model.to(device).eval() # YOLO or RT-DETR surrogate
    for q in model.parameters():
        q.requires_grad_(False)
    nc=int(getattr(model,'nc',0) or model.yaml.get('nc',len(names)))
    items=list_images_with_target(cfg,a.split,tid,a.max_files_per_photo or None)
    n_inst=sum(len(b) for _,b in items)
    print(f"{len(items)} images / {n_inst} '{names[tid]}' instances in '{a.split}' | imgsz {imgsz} | "
 f"corruptions {kinds or 'none'} | device {device}")
    bank=ViewBank(a.mesh,a.n_views,a.view_res,seed=a.seed,elev=tuple(a.elev),device=device,
                    cache=str(out/"viewbank.npz"))
    tv=TV3d(a.mesh,device)
    gen=GeneratorV2(bank.n_faces).to(device)
    opt=torch.optim.Adam(gen.parameters(),lr=a.lr,betas=(0.9,0.999))
    swa=AveragedModel(gen)
    rng=np.random.default_rng(a.seed)
    log,prev_small=[],-1
    json.dump(vars(a)|{"imgsz":imgsz,"n_images":len(items),"n_instances":n_inst,"device":device},
              open(out/"config.json","w"),indent=2)
    for ep in range(a.epochs):
        random.Random(a.seed*100+ep).shuffle(items)
        t0,s_sum,s_n,small,l_sum,steps=time.time(),0.0,0,0,0.0,0
        for i in range(0,len(items),a.batch):
            torch.manual_seed(1234+ep+i//a.batch+1) # seed_everything as in the code
            tex=texture(gen,32,seed=1234+ep+i//a.batch+1,add_noise=0.1,device=device)
            imgs,boxes,masks=build_batch(items[i:i+a.batch],tex,bank,a,rng,device,imgsz)
            if not a.no_affine:
                imgs,masks,boxes=apply_affine(imgs,masks,boxes,rng)
            imgs=corrupt(imgs,masks,boxes,kinds,a.corrupt_prob,rng)
            centres,scores=raw_scores(model,imgs,tid,nc)
            inst=[cpa_instance_score(centres[b],scores[b],bx,a.score_thresh)
                    for b in range(len(imgs)) for bx in boxes[b]]
            got=[s for s in inst if s is not None]
            small+=sum(1 for s in inst if s is None or float(s.detach())<0.5)
            det=torch.stack(got).sum() if got else torch.zeros((),device=device)
            loss=a.det_weight*det+a.tv_weight*tv(tex)
            opt.zero_grad()
            if loss.requires_grad:
                loss.backward()
                opt.step()
            if ep>=a.swa_start:
                swa.update_parameters(gen)
            s_sum+=sum(float(s.detach()) for s in got);s_n+=len(inst);l_sum+=float(loss.detach());steps+=1
            if i==0:
                save_example(out/"examples"/f"epoch{ep:02d}.jpg",imgs,boxes)
            if a.max_steps and steps>=a.max_steps:
                break
        if prev_small>=0 and small<=prev_small:
            for g in opt.param_groups:
                g["lr"]*=0.5
        prev_small=small
        rec=dict(epoch=ep,mean_score=s_sum/max(s_n,1),n_below_05=small,n_inst=s_n,
                   loss=l_sum/max(steps,1),lr=opt.param_groups[0]["lr"],sec=round(time.time()-t0,1))
        log.append(rec)
        print(json.dumps(rec),flush=True)
        json.dump(log,open(out/"train_log.json","w"),indent=2)
    final=swa.module if a.epochs>a.swa_start else gen
    with torch.no_grad():
        tex_final=texture(final,32,seed=1234,add_noise=0.0,device=device) # BN in train mode, as in the code
        tex_last=texture(gen,32,seed=1234,add_noise=0.0,device=device)
    np.save(out/"texture.npy",tex_final.cpu().numpy())
    np.save(out/"texture_last.npy",tex_last.cpu().numpy())
    torch.save({"gen":gen.state_dict(),"swa":swa.module.state_dict()},out/"generator.pt")
    cv2.imwrite(str(out/"patch_print.png"),cv2.cvtColor(printable(tex_final,a.mesh),cv2.COLOR_RGB2BGR))
    json.dump({"done":True,"config":vars(a),"weights_mtime":Path(a.weights).stat().st_mtime},
              open(out/"stage_done.json","w"),indent=2)
    print(f"saved {out/'texture.npy'} and {out/'patch_print.png'}")

if __name__=="__main__":
    main()
