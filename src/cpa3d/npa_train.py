"""Naturalistic patch (Hu et al. 2021): optimise only the BigGAN-128 latent z (patch=(G(z)+1)/2), clamp z to
[-tau,tau]; loss=mean per-tank max score+0.1*TV. Same 1 m sheet, views and detectors as CPA.
Options: --z-proj sphere (keep ||z||=sqrt(d)), --z-chunks, --sat-weight, --natural-stop S (keep the last
epoch with <=S% saturated pixels); --mode pixel (no GAN) and --mode camo (AdvCaT camouflage).
    python npa_train.py --mode npa --gan-class 977 --tau 3 --z-proj sphere --natural-stop 20 --out <dir>"""

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import list_images_with_target,load_data_cfg,raw_scores,training_imgsz
from cpa_train import apply_affine,build_batch,corrupt,save_example
from render import ViewBank

def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data",required=True)
    p.add_argument("--weights",required=True,help="surrogate detector")
    p.add_argument("--target-class",default="Tank")
    p.add_argument("--split",default="train")
    p.add_argument("--mode",default="npa",choices=["npa","pixel","camo"])
    p.add_argument("--biggan-weights",default="",help="G_ema.pth from the authors' pretrained_biggan.tar")
    p.add_argument("--gan-class",type=int,default=259,help="ImageNet class of the patch (259 = their default)")
    p.add_argument("--tau",type=float,default=10.0,help="latent clamp (max_value_latent_item)")
    p.add_argument("--patch-res",type=int,default=128,help="pixel mode: patch resolution")
    p.add_argument("--det-loss",default="inst_max",choices=["inst_max","img_max"])
    p.add_argument("--tv-weight",type=float,default=None,help="default 0.1 (npa, pixel), 0 (camo, as AdvCaT)")
    p.add_argument("--z-proj",default="none",choices=["none","sphere"],
                   help="npa: also keep z on the sphere ||z||2 = sqrt(d) where N(0,I) samples live")
    p.add_argument("--z-chunks",default="",help="npa: optimise only these BigGAN latent chunks (6 x 20), e.g. 0,1,2")
    p.add_argument("--sat-weight",type=float,default=0.0,help="penalty on saturated pixels (naturalness prior)")
    p.add_argument("--natural-stop",type=float,default=0.0,
                   help="keep the last epoch's patch whose saturated-pixel %% is <= this")
    p.add_argument("--palette",default="woodland",help="camo: woodland | desert | auto (k-means of the tanks)")
    p.add_argument("--camo-points",type=int,default=15,help="camo: control points per colour")
    p.add_argument("--camo-lambda",type=float,default=0.7,help="camo: trainable share of the Gumbel seed")
    p.add_argument("--camo-tau",type=float,default=0.3)
    p.add_argument("--camo-lr",type=float,default=0.001,help="camo: control-point learning rate")
    p.add_argument("--camo-seed-lr",type=float,default=0.01)
    p.add_argument("--ctrl-weight",type=float,default=1.0,help="camo: concentration loss weight")
    p.add_argument("--lr",type=float,default=0.005)
    p.add_argument("--plateau-patience",type=int,default=2)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--batch",type=int,default=4)
    p.add_argument("--corrupt",default="",help="optional CPA corruptions during training (light,shadow)")
    p.add_argument("--corrupt-prob",type=float,default=0.5)
    p.add_argument("--mesh",default=str(Path(__file__).with_name("aux_mesh.npz")))
    p.add_argument("--n-views",type=int,default=256)
    p.add_argument("--view-res",type=int,default=128)
    p.add_argument("--elev",type=float,nargs=2,default=(60,90))
    p.add_argument("--patch-m",type=float,default=1.0)
    p.add_argument("--hull-width-m",type=float,default=3.6)
    p.add_argument("--aspect",type=float,default=1.95)
    p.add_argument("--margin-frac",type=float,default=0.05)
    p.add_argument("--max-files-per-photo",type=int,default=0)
    p.add_argument("--imgsz",type=int,default=0)
    p.add_argument("--seed",type=int,default=15089,help="their default seed")
    p.add_argument("--device",default="")
    p.add_argument("--max-steps",type=int,default=0)
    p.add_argument("--out",required=True)
    return p.parse_args()

def tv2d(x):
    """ensemble_tool.model.TotalVariation"""
    a=torch.abs(x[:,:,1:]-x[:,:,:-1]+1e-6).sum()
    b=torch.abs(x[:,1:,:]-x[:,:-1,:]+1e-6).sum()
    return (a+b)/x.numel()

def make_jitter(rng):
    """adversarialYolo PatchTransformer: contrast U(0.8,1.2)."""
    def j(p):
        c=float(rng.uniform(0.8,1.2))
        b=float(rng.uniform(-0.1,0.1))
        n=(torch.rand_like(p)*2-1)*0.1
        return (p*c+b+n).clamp(1e-6,0.99999)
    return j

def main():
    a=parse_args()
    device=a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    out=Path(a.out);(out/"examples").mkdir(parents=True,exist_ok=True)
    kinds=[k.strip() for k in a.corrupt.split(",") if k.strip()]
    cfg,names,tid=load_data_cfg(a.data,a.target_class)
    imgsz=a.imgsz or training_imgsz(a.weights) or 960
    from common import load_detector
    model=load_detector(a.weights).model.to(device).eval() # YOLO or RT-DETR surrogate
    for q in model.parameters():
        q.requires_grad_(False)
    nc=int(getattr(model,"nc",0) or model.yaml.get("nc",len(names)))
    items=list_images_with_target(cfg,a.split,tid,a.max_files_per_photo or None)
    bank=ViewBank(a.mesh,a.n_views,a.view_res,seed=0,elev=tuple(a.elev),device=device,
                    cache=str(out/"viewbank.npz"))
    if a.tv_weight is None:
        a.tv_weight=0.0 if a.mode=="camo" else 0.1
    camo=None
    if a.mode=="npa":
        from biggan.loader import make_big_gan
        G=make_big_gan(a.biggan_weights or None,a.gan_class,device)
        z0=torch.normal(0.0,torch.ones(G.dim_z))
        if a.z_proj=="sphere":
            z0=z0/z0.norm()*G.dim_z**0.5
        z=z0.clamp(-a.tau,a.tau).to(device).requires_grad_(True)
        param=z
        groups=[{"params":[z],"lr":a.lr}]
        zmask=torch.ones_like(z)
        if a.z_chunks:
            cs=G.big_gan.z_chunk_size
            zmask=torch.zeros_like(z)
            for c in a.z_chunks.split(","):
                zmask[int(c)*cs:(int(c)+1)*cs]=1
        z_fixed=z.detach().clone()
        def patch_fn(hard=False):
            zz=z*zmask+z_fixed*(1-zmask)
            return (G(zz[None])[0]+1)*0.5
    elif a.mode=="camo":
        from camo import PALETTES,CamoTexture,auto_palette
        pal=auto_palette(items,seed=a.seed) if a.palette=="auto" else PALETTES[a.palette]
        print("palette:",[[round(c,3) for c in col] for col in pal])
        camo=CamoTexture(pal,res=a.patch_res,points_per_color=a.camo_points,lam=a.camo_lambda,
                           seed=a.seed).to(device)
        groups=[{"params":[camo.points],"lr":a.camo_lr},{"params":[camo.u_train],"lr":a.camo_seed_lr}]
        def patch_fn(hard=False):
            return camo(tau=a.camo_tau,hard=hard)
    else:
        G=None
        param=torch.rand((3,a.patch_res,a.patch_res),device=device).requires_grad_(True)
        groups=[{"params":[param],"lr":a.lr}]
        def patch_fn(hard=False):
            return param.clamp(0,1)
    with torch.no_grad():
        p0=patch_fn(hard=True).clamp(0,1).cpu().numpy()
    np.save(out/"patch_init.npy",p0)
    opt=torch.optim.Adam(groups,betas=(0.5,0.999),amsgrad=True)
    if camo is not None:
        sched=torch.optim.lr_scheduler.StepLR(opt,step_size=max(1,a.epochs//4),gamma=0.5)
    else:
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,"min",patience=a.plateau_patience)
    rng=np.random.default_rng(a.seed)
    jitter=make_jitter(rng)
    print(f"{a.mode}: {len(items)} images / {sum(len(b) for _, b in items)} tanks | imgsz {imgsz} | "
 f"class {a.gan_class if a.mode == 'npa' else '-'} | det-loss {a.det_loss} | corrupt {kinds or 'none'}"
 f"{' | z-proj ' + a.z_proj if a.mode == 'npa' else ''}")
    json.dump(vars(a)|{"imgsz":imgsz,"device":device},open(out/"config.json","w"),indent=2)
    log,kept=[],None
    for ep in range(a.epochs):
        random.Random(a.seed*100+ep).shuffle(items)
        t0,l_det,n_inst,below,steps=time.time(),0.0,0,0,0
        for i in range(0,len(items),a.batch):
            opt.zero_grad()
            patch=patch_fn()
            imgs,boxes,masks=build_batch(items[i:i+a.batch],patch,bank,a,rng,device,imgsz,jitter=jitter)
            imgs,masks,boxes=apply_affine(imgs,masks,boxes,rng)
            if kinds:
                imgs=corrupt(imgs,masks,boxes,kinds,a.corrupt_prob,rng)
            centres,scores=raw_scores(model,imgs,tid,nc)
            per=[]
            for b in range(len(imgs)):
                if a.det_loss=="img_max":
                    per.append(scores[b].max())
                    continue
                for x0,y0,x1,y1 in boxes[b]:
                    c=centres[b]
                    inside=(c[:,0]>=x0)&(c[:,0]<=x1)&(c[:,1]>=y0)&(c[:,1]<=y1)
                    if inside.any():
                        per.append(scores[b][inside].max())
            if not per:
                continue
            det=torch.stack(per)
            loss=det.mean()+a.tv_weight*tv2d(patch)
            if a.sat_weight:
                loss=loss+a.sat_weight*F.relu((patch-0.5).abs()-0.45).mean()/0.05
            if camo is not None:
                loss=loss+a.ctrl_weight*camo.ctrl_loss()
            loss.backward()
            opt.step()
            with torch.no_grad(): # their latent clamp + 1e-4 rounding
                if a.mode=="npa":
                    if a.z_proj=="sphere":
                        z.data=z.data/z.data.norm()*z.numel()**0.5
                    z.data=torch.round(z.data.clamp(-a.tau,a.tau)*1e4)*1e-4
                elif camo is not None:
                    camo.clip_()
                else:
                    param.data.clamp_(0,1)
            l_det+=float(det.detach().sum());n_inst+=len(per);below+=int((det.detach()<0.5).sum())
            steps+=1
            if i==0:
                save_example(out/"examples"/f"epoch{ep:02d}.jpg",imgs,boxes)
            if a.max_steps and steps>=a.max_steps:
                break
        ep_loss=l_det/max(n_inst,1)
        sched.step() if camo is not None else sched.step(ep_loss)
        rec=dict(epoch=ep,mean_max_score=ep_loss,n_below_05=below,n_inst=n_inst,
                   lr=opt.param_groups[0]["lr"],sec=round(time.time()-t0,1))
        if a.mode=="npa":
            rec|=dict(z_inf=float(z.detach().abs().max()),z_l1=float(z.detach().abs().mean()))
        log.append(rec)
        print(json.dumps(rec),flush=True)
        json.dump(log,open(out/"train_log.json","w"),indent=2)
        with torch.no_grad():
            pe=patch_fn(hard=True).clamp(0,1)
        sat_pct=float(((pe<0.03)|(pe>0.97)).any(0).float().mean()*100)
        rec["saturated_pct"]=round(sat_pct,1)
        json.dump(log,open(out/"train_log.json","w"),indent=2)
        if a.natural_stop and sat_pct<=a.natural_stop:
            kept=(ep,pe.cpu().numpy(),(z.detach().clone() if a.mode=="npa" else None))
        cv2.imwrite(str(out/"examples"/f"patch_epoch{ep:02d}.png"),
                    cv2.cvtColor((pe.permute(1,2,0).cpu().numpy()*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
    with torch.no_grad():
        pf=patch_fn(hard=True).clamp(0,1).cpu().numpy() # camo: hard colours, as printed
    if a.natural_stop: # naturalness-budget early stop
        if kept is None:
            print("natural_stop_epoch",None)
            pf=p0
        else:
            print("natural_stop_epoch",kept[0])
            pf=kept[1]
            if kept[2] is not None:
                z.data=kept[2]
        json.dump({"natural_stop_epoch":None if kept is None else kept[0]},open(out/"natural_stop.json","w"))
    np.save(out/"patch.npy",pf)
    if a.mode=="npa":
        np.save(out/"latent.npy",z.detach().cpu().numpy())
    if camo is not None:
        torch.save(camo.state_dict(),out/"camo.pt")
    big=cv2.resize((pf.transpose(1,2,0)*255).astype(np.uint8),(600,600),
                     interpolation=cv2.INTER_NEAREST if camo is not None else cv2.INTER_CUBIC)
    cv2.imwrite(str(out/"patch_print.png"),cv2.cvtColor(big,cv2.COLOR_RGB2BGR))
    side=np.concatenate([(p0.transpose(1,2,0)*255).astype(np.uint8),np.full((128,8,3),255,np.uint8),
                           cv2.resize((pf.transpose(1,2,0)*255).astype(np.uint8),(128,128))],1)
    cv2.imwrite(str(out/"patch_init_vs_final.png"),cv2.cvtColor(side,cv2.COLOR_RGB2BGR))
    json.dump({"done":True,"config":vars(a)},open(out/"stage_done.json","w"),indent=2)
    print(f"saved {out/'patch.npy'}  ({out/'patch_init_vs_final.png'})")

if __name__=="__main__":
    main()
