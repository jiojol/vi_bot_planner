"""Small fitted RGB ground-material head; not a general scene segmenter."""
import numpy as np
from scipy.ndimage import gaussian_filter


def features(rgb):
    x=gaussian_filter(np.asarray(rgb,np.float32),sigma=(1.,1.,0.));norm=x/(x.sum(2,keepdims=True)+1.)
    excess=norm[:,:,1]-np.maximum(norm[:,:,0],norm[:,:,2]);saturation=(x.max(2)-x.min(2))/(x.max(2)+1.)
    return np.concatenate([norm,excess[:,:,None],saturation[:,:,None],(np.log1p(x.mean(2))/6.)[:,:,None]],2).astype(np.float32)


class GroundHead:
    def __init__(self,path):self.weights=dict(np.load(path))
    def __call__(self,rgb):
        f=features(rgb);shape=f.shape[:2];x=f.reshape(-1,6)
        layers=len([k for k in self.weights if k.startswith('w')])
        for i in range(layers):
            x=x@self.weights[f'w{i}'].T+self.weights[f'b{i}']
            if i<layers-1:x=np.maximum(x,0)
        return (1/(1+np.exp(-np.clip(x[:,0],-40,40)))).reshape(shape)
