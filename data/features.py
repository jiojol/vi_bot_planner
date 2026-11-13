"""Deterministic current-sensor-only preprocessing, identical offline and online."""
import numpy as np
from PIL import Image

BEV_SHAPE=(40,40)

def lidar_bev(xyz,valid):
    pts=np.asarray(xyz)[np.asarray(valid,dtype=bool)];out=np.zeros((5,40,40),np.float32)
    if not len(pts):return out
    def indices(p):
        idx=np.floor((p[:,:2]-[-2,-4])/.2).astype(int);v=(idx[:,0]>=0)&(idx[:,0]<40)&(idx[:,1]>=0)&(idx[:,1]<40);return idx,v
    idx,v=indices(pts);q=pts[v];idx=idx[v];ix,iy=idx.T
    hit=q[:,2]>.12
    np.add.at(out[0],(ix[hit],iy[hit]),1);out[0]=np.minimum(out[0]/8,1)
    np.maximum.at(out[1],(ix,iy),q[:,2]/2)
    np.add.at(out[2],(ix,iy),1);out[2]=np.minimum(out[2]/8,1)
    # Observed ray space vs unknown. Returns determine free prefix, never full map.
    ray=(pts[:,None,:2]*np.linspace(0,.97,32)[None,:,None]).reshape(-1,2)
    ri,rv=indices(ray);out[3,ri[rv,0],ri[rv,1]]=1
    out[3][out[0]>0]=0
    out[4]=np.maximum(out[3],(out[2]>0).astype('float32'))
    return out

def rgb_small(rgb):return np.asarray(Image.fromarray(rgb).resize((80,60),Image.Resampling.BILINEAR)).copy()

def current_inputs(rgb,lidar_xyz,lidar_valid_mask,goal_xy,device='cpu',full_resolution=False):
    import torch
    return {'rgb':torch.from_numpy((rgb if full_resolution else rgb_small(rgb)).transpose(2,0,1).copy()).unsqueeze(0).to(device).float()/255,'bev':torch.from_numpy(lidar_bev(lidar_xyz,lidar_valid_mask)).unsqueeze(0).to(device),'goal':torch.tensor(np.asarray(goal_xy),device=device,dtype=torch.float32).unsqueeze(0)}
