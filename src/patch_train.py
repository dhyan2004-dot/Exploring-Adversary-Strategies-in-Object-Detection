"""2-D adversarial patch (EOT, Brown 2017 / Thys 2019) against YOLOv8 on object crops.
Loss = peak pre-NMS target score over the patched crop + lambda*TV. EOT: rotation, scale, brightness,
contrast, position. --corrupt adds CPA-style patch corruptions during training.
    python patch_train.py --data <data.yaml> --weights <best.pt> --target-class Tank --epochs 80"""

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import nn

from corruptions import KINDS,apply_layers,sample_corruption

DATASET_ROOT=Path(__file__).resolve().parents[1]/"data"/"Aerial Image Dataset"
DEFAULT_DATA_YAML=DATASET_ROOT/"yolo_dataset"/"data.yaml"

def parse_args():
    p=argparse.ArgumentParser(description="Train a standard EOT adversarial patch against YOLOv8.")
    p.add_argument("--data",type=str,default=str(DEFAULT_DATA_YAML))
    p.add_argument("--weights",type=str,required=True,help="Path to your fine-tuned best.pt")
    p.add_argument("--target-class",type=str,default="Military Truck",
                    help="Class name (as in data.yaml) or integer class id to attack.")
    p.add_argument("--split",type=str,default="train",choices=["train","val","test","dev_val"],
                    help="Which image set to optimize the patch against. 'train' (default) is standard -- "
 "the patch should generalize to unseen instances of the class, same as the model does.")
    p.add_argument("--crop-resolution",type=int,default=320,
                    help="Each object crop is resized to this (square) resolution before the forward pass. "
 "Must be divisible by 32. Independent of the detector's own training imgsz -- "
 "YOLOv8 is fully convolutional and handles any valid input size.")
    p.add_argument("--crop-context",type=float,default=2.5,
                    help="Crop size as a multiple of the target object's largest dimension -- e.g. 2.5 means "
 "the crop is 2.5x the object's size, giving it room plus some surrounding context. "
 "Lower this if dense scenes are pulling neighboring instances of the same class into "
 "the crop (which would dilute the loss the same way full-image training did).")
    p.add_argument("--min-crop-px",type=int,default=64,
                    help="Floor on crop size in source-image pixels, for very small objects.")
    p.add_argument("--max-crop-px",type=int,default=1600,
                    help="Ceiling on crop size in source-image pixels, for very large objects.")
    p.add_argument("--patch-size",type=int,default=128,help="Patch resolution in pixels (square).")
    p.add_argument("--patch-scale",type=float,default=0.3,
                    help="Patch width as a fraction of the target object's bounding-box width.")
    p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--batch-size",type=int,default=8,help="Instance-crops per step.")
    p.add_argument("--max-instances",type=int,default=None,
                    help="Cap the instance pool to the first N in filename order (quick smoke tests only -- "
 "this is NOT a random sample).")
    p.add_argument("--instances-per-epoch",type=int,default=None,
                    help="Randomly sample this many instances from the full pool each epoch, to cap epoch "
 "time. A different random subset is drawn every epoch.")
    p.add_argument("--min-clean-conf",type=float,default=0.25,
                    help="Only instances the detector already detects WITHOUT a patch (peak class confidence "
 ">= this) contribute to the loss. Training on instances the detector doesn't see "
 "anyway gives a meaningless gradient.")
    p.add_argument("--lr",type=float,default=0.01)
    p.add_argument("--tv-weight",type=float,default=0.005)
    p.add_argument("--corrupt",type=str,default="",
                    help="CPA mode. Comma-separated corruptions to apply to the patch during training "
 "(from: light,shadow,rain,mud, or 'all'). Empty (default) = standard EOT patch with "
 "no corruption, i.e. the baseline.")
    p.add_argument("--corrupt-prob",type=float,default=0.8,
                    help="Probability that a given training step's patch is corrupted at all (the rest see the "
 "clean patch, so it still works uncorrupted). 1-2 corruption types are stacked when it is.")
    p.add_argument("--rotate-deg",type=float,default=20.0,help="Max +/- rotation in degrees (EOT).")
    p.add_argument("--scale-jitter",type=float,default=0.2,help="+/- fractional scale jitter (EOT).")
    p.add_argument("--brightness-jitter",type=float,default=0.15,help="+/- additive brightness jitter (EOT).")
    p.add_argument("--contrast-jitter",type=float,default=0.15,help="+/- multiplicative contrast jitter (EOT).")
    p.add_argument("--device",type=str,default=None)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output-dir",type=str,default="patch_runs/standard_eot")
    return p.parse_args()

def load_target_class_instances(images_dir:Path,labels_dir:Path,target_id:int,max_instances=None):
    """Returns a list of (image_path, (xc,yc,w,h)) -- one entry per instance of target_id, since each instance..."""
    samples=[]
    for label_file in sorted(labels_dir.glob("*.txt")):
        if "_dup" in label_file.stem:
            continue
        candidates=list(images_dir.glob(f"{label_file.stem}.*"))
        if not candidates:
            continue
        img_path=candidates[0]
        for line in label_file.read_text().splitlines():
            if not line.strip():
                continue
            parts=line.split()
            if int(parts[0])!=target_id:
                continue
            xc,yc,w,h=map(float,parts[1:5])
            samples.append((img_path,(xc,yc,w,h)))
            if max_instances and len(samples)>=max_instances:
                return samples
    return samples

def compute_crop_window(xc,yc,w,h,img_w,img_h,context,min_px,max_px):
    """Square crop window (in source-image pixel coords) centered on the object, sized to `context` times its..."""
    obj_px=max(w*img_w,h*img_h)
    crop_px=int(round(min(max(obj_px*context,min_px),max_px,min(img_w,img_h))))
    crop_px=max(crop_px,1)
    cx,cy=xc*img_w,yc*img_h
    x0=int(round(cx-crop_px/2))
    y0=int(round(cy-crop_px/2))
    x0=max(0,min(x0,img_w-crop_px))
    y0=max(0,min(y0,img_h-crop_px))
    return x0,y0,crop_px

def load_crop_tensor_and_box(img_path:Path,box,crop_res:int,context:float,min_px:int,max_px:int):
    """Loads the source image, crops a context window around `box`, resizes to crop_res, and returns (crop_tensor..."""
    img=Image.open(img_path).convert("RGB")
    img_w,img_h=img.size
    xc,yc,w,h=box
    x0,y0,crop_px=compute_crop_window(xc,yc,w,h,img_w,img_h,context,min_px,max_px)
    crop=img.crop((x0,y0,x0+crop_px,y0+crop_px)).resize((crop_res,crop_res),Image.BILINEAR)
    box_cx_px,box_cy_px=xc*img_w-x0,yc*img_h-y0
    box_in_crop=(box_cx_px/crop_px,box_cy_px/crop_px,
                   (w*img_w)/crop_px,(h*img_h)/crop_px)
    arr=torch.from_numpy(np.array(crop)).float().permute(2,0,1)/255.0
    return arr,box_in_crop

def rotate_patch(patch:torch.Tensor,angle_deg:float):
    """Rotate a [3,S,S] patch about its center via grid_sample; returns the rotated patch plus a same-size mask..."""
    device=patch.device
    theta=math.radians(angle_deg)
    cos_t,sin_t=math.cos(theta),math.sin(theta)
    rot_mat=torch.tensor([[cos_t,-sin_t,0.0],[sin_t,cos_t,0.0]],device=device,dtype=patch.dtype)
    grid=F.affine_grid(rot_mat.unsqueeze(0),[1,3,patch.shape[1],patch.shape[2]],align_corners=False)
    rotated=F.grid_sample(patch.unsqueeze(0),grid,mode="bilinear",padding_mode="zeros",align_corners=False)
    ones=torch.ones(1,1,patch.shape[1],patch.shape[2],device=device,dtype=patch.dtype)
    mask=F.grid_sample(ones,grid,mode="bilinear",padding_mode="zeros",align_corners=False)
    return rotated.squeeze(0),mask.squeeze(0) # [3,S,S], [1,S,S]

def make_corruptor(kinds,prob,rng):
    """Returns tile -> corrupted tile ([3,S,S] torch, differentiable), or None for no corruption."""
    if not kinds:
        return None
    def corrupt(tile):
        layers=sample_corruption(tile.shape[-1],rng,kinds,prob)
        return apply_layers(tile,layers) if layers else tile
    return corrupt

def apply_patch_to_image(image:torch.Tensor,patch:torch.Tensor,box,args):
    """image: [3,imgsz,imgsz] 0..1."""
    imgsz=image.shape[-1]
    xc,yc,w,h=box
    box_w_px=w*imgsz
    angle=random.uniform(-args.rotate_deg,args.rotate_deg)
    scale=args.patch_scale*(1.0+random.uniform(-args.scale_jitter,args.scale_jitter))
    brightness=random.uniform(-args.brightness_jitter,args.brightness_jitter)
    contrast=1.0+random.uniform(-args.contrast_jitter,args.contrast_jitter)
    jitter_frac=0.1 # placement jitter as a fraction of the box size
    jitter_x=random.uniform(-jitter_frac,jitter_frac)*w*imgsz
    jitter_y=random.uniform(-jitter_frac,jitter_frac)*h*imgsz
    place_size=max(4,int(round(box_w_px*scale)))
    place_size=min(place_size,imgsz) # never larger than the image itself
    rotated,mask=rotate_patch(patch,angle)
    rotated=torch.clamp(rotated*contrast+brightness,0.0,1.0)
    corruptor=getattr(args,"corruptor",None)
    if corruptor is not None:
        rotated=corruptor(rotated) # CPA: light / shadow / rain / mud on the patch tile itself
    resized=F.interpolate(rotated.unsqueeze(0),size=(place_size,place_size),
                             mode="bilinear",align_corners=False).squeeze(0)
    resized_mask=F.interpolate(mask.unsqueeze(0),size=(place_size,place_size),
                                  mode="bilinear",align_corners=False).squeeze(0)
    cx=xc*imgsz+jitter_x
    cy=yc*imgsz+jitter_y
    x0=int(round(cx-place_size/2))
    y0=int(round(cy-place_size/2))
    img_x0,img_x1=max(0,x0),min(imgsz,x0+place_size)
    img_y0,img_y1=max(0,y0),min(imgsz,y0+place_size)
    if img_x1<=img_x0 or img_y1<=img_y0:
        return image # box fell entirely outside the image after jitter; skip
    patch_x0,patch_y0=img_x0-x0,img_y0-y0
    patch_x1,patch_y1=patch_x0+(img_x1-img_x0),patch_y0+(img_y1-img_y0)
    padded_patch=torch.zeros_like(image)
    padded_mask=torch.zeros(1,imgsz,imgsz,device=image.device,dtype=image.dtype)
    padded_patch[:,img_y0:img_y1,img_x0:img_x1]=resized[:,patch_y0:patch_y1,patch_x0:patch_x1]
    padded_mask[:,img_y0:img_y1,img_x0:img_x1]=resized_mask[:,patch_y0:patch_y1,patch_x0:patch_x1]
    return image*(1-padded_mask)+padded_patch*padded_mask

def extract_class_confidence(raw_pred:torch.Tensor,class_id:int,num_classes:int):
    """Find the (4+num_classes)-sized channel axis in the raw YOLO prediction tensor regardless of its exact..."""
    target_c=4+num_classes
    channel_axis=None
    for i,d in enumerate(raw_pred.shape):
        if i==0:
            continue
        if d==target_c:
            channel_axis=i
            break
    if channel_axis is None:
        raise RuntimeError(
            f"Could not find a channel axis of size {target_c} (=4 box coords + {num_classes} classes) "
 f"in the raw prediction tensor, shape {tuple(raw_pred.shape)}. Ultralytics' internal output "
 f"format may differ in your installed version -- print raw_pred.shape (and, if it's a tuple, "
 f"each element's shape) right after the forward pass to inspect it, then adjust this function."
        )
    raw_pred=raw_pred.movedim(channel_axis,1) # -> [B, C, ...]
    conf=raw_pred[:,4+class_id,...]
    return conf.reshape(conf.shape[0],-1) # [B, num_anchors]

def total_variation(patch:torch.Tensor):
    dh=(patch[:,1:,:]-patch[:,:-1,:]).abs().mean()
    dw=(patch[:,:,1:]-patch[:,:,:-1]).abs().mean()
    return dh+dw

def main():
    args=parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device=args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    with open(args.data) as f:
        data_cfg=yaml.safe_load(f)
    names=data_cfg["names"]
    name_to_id={v:k for k,v in names.items()}
    num_classes=len(names)
    if args.target_class.isdigit():
        target_id=int(args.target_class)
    else:
        if args.target_class not in name_to_id:
            raise SystemExit(f"'{args.target_class}' not found in data.yaml names: {list(names.values())}")
        target_id=name_to_id[args.target_class]
    print(f"Target class: {names[target_id]} (id={target_id})")
    kinds=KINDS if args.corrupt.strip().lower()=="all" else [k.strip() for k in args.corrupt.split(",") if k.strip()]
    bad=[k for k in kinds if k not in KINDS]
    if bad:
        raise SystemExit(f"Unknown --corrupt value(s) {bad}; choose from {KINDS} or 'all'.")
    args.corruptor=make_corruptor(kinds,args.corrupt_prob,np.random.default_rng(args.seed))
    if kinds:
        print(f"CPA mode: corrupting the patch during training with {kinds} (p={args.corrupt_prob})")
    else:
        print("Standard EOT mode: no corruption during training")
    if args.crop_resolution%32!=0:
        raise SystemExit(f"--crop-resolution must be divisible by 32 (got {args.crop_resolution})")
    weights=Path(args.weights)
    from ultralytics import YOLO
    yolo=YOLO(str(weights))
    model=yolo.model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    dataset_root=Path(data_cfg["path"])
    split_dir={"train":"train","val":"val","test":"test","dev_val":"dev_val"}[args.split]
    images_dir=dataset_root/"images"/split_dir
    labels_dir=dataset_root/"labels"/split_dir
    samples=load_target_class_instances(images_dir,labels_dir,target_id,args.max_instances)
    if not samples:
        raise SystemExit(f"No instances of class '{names[target_id]}' found in {images_dir}")
    print(f"Found {len(samples)} instances of '{names[target_id]}' in '{args.split}'")
    patch=nn.Parameter(torch.rand(3,args.patch_size,args.patch_size,device=device)*0.5+0.25)
    optimizer=torch.optim.Adam([patch],lr=args.lr)
    out_dir=Path(args.output_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    print("\nStarting patch optimization...")
    first_step=True
    for epoch in range(1,args.epochs+1):
        random.shuffle(samples)
        epoch_samples=samples[:args.instances_per_epoch] if args.instances_per_epoch else samples
        t0=time.time()
        sum_loss,sum_patched,sum_clean,sum_clean_kept=0.0,0.0,0.0,0.0
        n_steps,n_seen,n_kept=0,0,0
        for b0 in range(0,len(epoch_samples),args.batch_size):
            batch=epoch_samples[b0:b0+args.batch_size]
            cleans,patched=[],[]
            for img_path,box in batch:
                crop,box_in_crop=load_crop_tensor_and_box(
                    img_path,box,args.crop_resolution,args.crop_context,args.min_crop_px,args.max_crop_px
                )
                crop=crop.to(device)
                cleans.append(crop)
                patched.append(apply_patch_to_image(crop,patch,box_in_crop,args))
            clean_batch=torch.stack(cleans,dim=0) # [B,3,crop_res,crop_res]
            patched_batch=torch.stack(patched,dim=0)
            with torch.no_grad():
                clean_pred=model(clean_batch)
                if isinstance(clean_pred,(tuple,list)):
                    clean_pred=clean_pred[0]
                clean_peak=extract_class_confidence(clean_pred,target_id,num_classes).max(dim=1).values # [B]
            keep=clean_peak>=args.min_clean_conf
            n_seen+=len(batch)
            n_kept+=int(keep.sum().item())
            sum_clean+=clean_peak.sum().item()
            if not keep.any():
                continue # nothing detectable in this batch -> no useful gradient
            sum_clean_kept+=clean_peak[keep].mean().item()
            raw_pred=model(patched_batch)
            if first_step:
                if isinstance(raw_pred,(tuple,list)):
                    print(f"[debug] model(x) returned a {type(raw_pred).__name__} of "
 f"{len(raw_pred)} tensors; shapes: "
 f"{[tuple(t.shape) if torch.is_tensor(t) else type(t) for t in raw_pred]}")
                else:
                    print(f"[debug] model(x) returned a single tensor; shape: {tuple(raw_pred.shape)}")
                print(f"[debug] looking for a channel axis of size 4+{num_classes}={4+num_classes} "
 f"(4 box coords + {num_classes} classes)")
                first_step=False
            if isinstance(raw_pred,(tuple,list)):
                raw_pred=raw_pred[0]
            peak=extract_class_confidence(raw_pred,target_id,num_classes).max(dim=1).values # [B]
            obj_loss=peak[keep].mean()
            tv_loss=total_variation(patch)
            loss=obj_loss+args.tv_weight*tv_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                patch.clamp_(0.0,1.0)
            sum_loss+=loss.item()
            sum_patched+=obj_loss.item()
            n_steps+=1
        elapsed=time.time()-t0
        if n_steps==0:
            print(f"Epoch {epoch:3d}/{args.epochs}  no usable instances (detector found none at "
 f"conf>={args.min_clean_conf} on {n_seen} crops) -- lower --min-clean-conf or check the "
 f"crop settings.  [{elapsed:.0f}s]")
            continue
        ck,pk=sum_clean_kept/n_steps,sum_patched/n_steps
        print(f"Epoch {epoch:3d}/{args.epochs}  clean(all)={sum_clean/max(n_seen,1):.3f}  "
 f"clean(kept)={ck:.3f}  patched(kept)={pk:.3f}  drop={ck-pk:+.3f}  "
 f"loss={sum_loss/n_steps:.3f}  kept={n_kept}/{n_seen}  [{elapsed:.0f}s]")
        if epoch%5==0 or epoch==args.epochs:
            save_patch_png(patch,out_dir/f"patch_epoch{epoch}.png")
    final_path=out_dir/"patch_final.png"
    save_patch_png(patch,final_path)
    print(f"\nDone. Final patch saved to: {final_path}")
    print("Next step: patch_evaluate.py to measure attack success rate / confidence reduction.")

def save_patch_png(patch:torch.Tensor,path:Path):
    arr=(patch.detach().clamp(0,1).cpu().permute(1,2,0).numpy()*255).astype("uint8")
    Image.fromarray(arr).save(path)

if __name__=="__main__":
    main()
