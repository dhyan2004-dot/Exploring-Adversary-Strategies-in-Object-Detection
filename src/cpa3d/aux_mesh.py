"""Auxiliary roof mesh carrying the 3-D patch (CPA, Zhang et al. 2024): a 32x32 vertex grid
(1,922 facets, one colour each) shaped as a gentle dome z=h(1-4x^2)(1-4y^2), plus facet adjacency for TV3d.
    python aux_mesh.py --out aux_mesh.npz"""

import argparse
from collections import defaultdict

import numpy as np

def grid_faces(n):
    faces=[]
    for i in range(n-1):
        for j in range(n-1):
            a,b,c,d=i*n+j,(i+1)*n+j,(i+1)*n+j+1,i*n+j+1
            faces+=[(a,b,c),(a,c,d)]
    return np.asarray(faces,dtype=np.int64)

def adjacency(verts,faces):
    """Pairs of facets sharing an edge + that edge's 3-D length (for the smoothness loss)."""
    edge_faces=defaultdict(list)
    for fi,(a,b,c) in enumerate(faces):
        for u,v in ((a,b),(b,c),(c,a)):
            edge_faces[(min(u,v),max(u,v))].append(fi)
    pairs,lengths=[],[]
    for (u,v),fs in edge_faces.items():
        if len(fs)==2:
            pairs.append(fs)
            lengths.append(np.linalg.norm(verts[u]-verts[v]))
    return np.asarray(pairs,dtype=np.int64),np.asarray(lengths,dtype=np.float32)

def dome_heights(xy,height):
    x,y=xy[:,0],xy[:,1]
    return height*(1-(2*x)**2)*(1-(2*y)**2)

def load_model_triangles(path):
    import trimesh
    scene=trimesh.load(path,force="scene")
    mesh=trimesh.util.concatenate([g for g in scene.dump() if hasattr(g,"triangles")])
    tri=np.asarray(mesh.triangles,dtype=np.float64) # [T,3,3]
    return tri

def drape_on_model(tri,n,roof_fraction,up_axis):
    """Ray-cast a grid downwards onto the model's roof around its centre."""
    pts=tri.reshape(-1,3)
    if up_axis=="y": # glTF convention (Y up) -> Z up
        tri=tri[...,[0,2,1]]*np.array([1,-1,1])
        pts=tri.reshape(-1,3)
    lo,hi=pts.min(0),pts.max(0)
    centre=(lo+hi)/2
    tri=tri-centre
    ext=hi-lo
    if ext[1]>ext[0]: # make x the long axis
        tri=tri[...,[1,0,2]]
        ext=ext[[1,0,2]]
    width=ext[1]
    side=roof_fraction*width # patch side in model units
    g=np.linspace(-0.5,0.5,n)*side
    xx,yy=np.meshgrid(g,g,indexing="ij")
    origins=np.stack([xx.ravel(),yy.ravel(),np.full(xx.size,ext[2]*2)],1)
    d=np.array([0.0,0.0,-1.0])
    z=np.full(len(origins),-np.inf)
    v0,v1,v2=tri[:,0],tri[:,1],tri[:,2]
    e1,e2=v1-v0,v2-v0
    p=np.cross(d,e2)
    det=(e1*p).sum(1)
    ok=np.abs(det)>1e-12
    v0,e1,e2,p,det=v0[ok],e1[ok],e2[ok],p[ok],det[ok]
    inv=1.0/det
    for k,o in enumerate(origins): # Moller-Trumbore, vectorised over triangles
        t_=o-v0
        u=(t_*p).sum(1)*inv
        q=np.cross(t_,e1)
        v=(q@d)*inv
        t=(q*e2).sum(1)*inv
        hit=(u>=0)&(v>=0)&(u+v<=1)&(t>0)
        if hit.any():
            z[k]=(o[2]-t[hit]).max()
    if np.isinf(z).any():
        raise SystemExit("rays missed the model")
    z=(z-z.min())/side # heights in patch units, lowest point = 0
    return z

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=32,help="vertices per side (32 -> 1,922 facets)")
    ap.add_argument("--dome-height",type=float,default=0.06,
                    help="procedural roof: dome height in units of the patch side")
    ap.add_argument("--model",default=None,help="optional tank model (OBJ/GLB/GLTF) to drape onto")
    ap.add_argument("--up-axis",default="y",choices=["y","z"],help="model up axis (glTF = y)")
    ap.add_argument("--roof-fraction",type=float,default=0.3,
                    help="with --model: patch side as a fraction of the vehicle width")
    ap.add_argument("--out",default="aux_mesh.npz")
    a=ap.parse_args()
    n=a.n
    g=np.linspace(-0.5,0.5,n)
    xx,yy=np.meshgrid(g,g,indexing="ij")
    xy=np.stack([xx.ravel(),yy.ravel()],1)
    if a.model:
        z=drape_on_model(load_model_triangles(a.model),n,a.roof_fraction,a.up_axis)
        source=f"draped on {a.model} (roof fraction {a.roof_fraction})"
    else:
        z=dome_heights(xy,a.dome_height)
        source=f"procedural dome, height {a.dome_height}"
    verts=np.concatenate([xy,z[:,None]],1).astype(np.float32)
    faces=grid_faces(n)
    pairs,lengths=adjacency(verts,faces)
    np.savez(a.out,verts=verts,faces=faces,adj_pairs=pairs,adj_len=lengths,source=source)
    print(f"{a.out}: {len(verts)} vertices, {len(faces)} facets, {len(pairs)} shared edges; {source}")

if __name__=="__main__":
    main()
