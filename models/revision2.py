"""Revision 2: frozen pretrained vision + bidirectional FiLM/BEV/Q-Former + conditioned DiT.
DINOv2 is explicitly named as a substitute until official DINOv3 access is available.
"""
import math,json,hashlib
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from models.network import LidarBEVEncoder,TrajectoryVAE

def spatial_pe(h,w,d):
    y,x=torch.meshgrid(torch.linspace(-1,1,h),torch.linspace(-1,1,w),indexing='ij');xy=torch.stack([x,y],-1).reshape(-1,2)
    frequencies=torch.logspace(0,math.log10(32),d//4)*math.pi
    return torch.cat([fn(xy[:,j,None]*frequencies) for j in range(2) for fn in [torch.sin,torch.cos]],-1)

class FrozenVision(nn.Module):
    def __init__(self,c):
        super().__init__()
        from transformers import AutoModel,AutoModelForDepthEstimation,AutoImageProcessor
        base=Path(__file__).resolve().parents[1]
        resolve=lambda value: str(Path(value) if Path(value).is_absolute() else base/value)
        semantic_path=resolve(c['semantic_weights']);geometry_path=resolve(c['geometry_weights'])
        lock=base/'configs/vision_weights.lock.json'
        if lock.exists():
            for record in json.loads(lock.read_text()):
                if resolve(record['directory']) in [semantic_path,geometry_path]:
                    for name,digest in record['sha256'].items():
                        assert hashlib.sha256(Path(resolve(record['directory']),name).read_bytes()).hexdigest()==digest,'Pretrained weight checksum mismatch'
        self.semantic=AutoModel.from_pretrained(semantic_path).eval().requires_grad_(False)
        self.depth=AutoModelForDepthEstimation.from_pretrained(geometry_path).eval().requires_grad_(False)
        self.sp=AutoImageProcessor.from_pretrained(semantic_path,use_fast=False);self.gp=AutoImageProcessor.from_pretrained(geometry_path,use_fast=False)
    @torch.no_grad()
    def forward(self,images):
        dev=next(self.parameters()).device;sx=self.sp(images=images,return_tensors='pt').to(dev);gx=self.gp(images=images,return_tensors='pt').to(dev)
        s=self.semantic(**sx).last_hidden_state[:,1+getattr(self.semantic.config,'num_register_tokens',0):]
        patch=self.semantic.config.patch_size;h,w=sx['pixel_values'].shape[-2:];s=s.transpose(1,2).reshape(len(images),-1,h//patch,w//patch)
        # Use actual DA2 geometry backbone feature map, NOT only its scalar relative-depth output.
        g=self.depth.backbone(**gx).feature_maps[-1]
        if g.ndim==3:
            gh,gw=gx['pixel_values'].shape[-2:];gp=self.depth.config.backbone_config.patch_size;g=g[:,1:].transpose(1,2).reshape(len(images),-1,gh//gp,gw//gp)
        return tuple(F.adaptive_avg_pool2d(z,(6,8)).flatten(2).transpose(1,2).half().float() for z in [s,g])

class QBlock(nn.Module):
    def __init__(self,d):
        super().__init__();self.block=nn.TransformerDecoderLayer(d,4,d*4,dropout=0,batch_first=True,norm_first=True)
    def forward(self,q,m):return self.block(q,m)

class PerceptionV2(nn.Module):
    def __init__(self,c):
        super().__init__();d=c['dim'];self.sproj=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,d));self.gproj=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,d))
        self.sf=nn.Linear(d,2*d);self.gf=nn.Linear(d,2*d);self.image_norm=nn.LayerNorm(d)
        self.queries=nn.Parameter(torch.randn(1,1600,d)*.02);self.register_buffer('bp',spatial_pe(40,40,d)[None]);self.register_buffer('ip',spatial_pe(6,8,d)[None]);self.modality=nn.Parameter(torch.randn(2,1,d)*.02)
        self.cross=nn.MultiheadAttention(d,4,batch_first=True);self.lidar=LidarBEVEncoder(d);self.lnorm=nn.LayerNorm(d);self.gate=nn.Linear(d*2,d)
        self.q=nn.Parameter(torch.randn(1,32,d)*.02);self.qblocks=nn.ModuleList([QBlock(d) for _ in range(2)]);self.out=nn.LayerNorm(d)
    def forward(self,b):
        s=self.sproj(b['sem']);g=self.gproj(b['geo']);sa,sb=self.sf(g.mean(1)).chunk(2,-1);ga,gb=self.gf(s.mean(1)).chunk(2,-1)
        si=s*(1+.1*sa[:,None])+sb[:,None];gi=g*(1+.1*ga[:,None])+gb[:,None]
        mem=torch.cat([si+self.ip+self.modality[0],gi+self.ip+self.modality[1]],1)
        im=self.image_norm(self.cross((self.queries+self.bp).expand(len(s),-1,-1),mem,mem,need_weights=False)[0]);li=self.lnorm(self.lidar(b['bev']))+self.bp
        gate=torch.sigmoid(self.gate(torch.cat([im,li],-1)));env=gate*im+(1-gate)*li;q=self.q.expand(len(s),-1,-1)
        for block in self.qblocks:q=block(q,env)
        return self.out(q)

class DiTBlock(nn.Module):
    def __init__(self,d):
        super().__init__();self.n1=nn.LayerNorm(d,elementwise_affine=False);self.n2=nn.LayerNorm(d);self.n3=nn.LayerNorm(d,elementwise_affine=False)
        self.attn=nn.MultiheadAttention(d,4,batch_first=True);self.cross=nn.MultiheadAttention(d,4,batch_first=True);self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
        self.mod=nn.Sequential(nn.SiLU(),nn.Linear(d,6*d));nn.init.zeros_(self.mod[-1].weight);nn.init.zeros_(self.mod[-1].bias)
    def forward(self,x,env,c):
        s1,a1,g1,s2,a2,g2=self.mod(c).chunk(6,-1);q=self.n1(x)*(1+a1[:,None])+s1[:,None];x=x+g1[:,None]*self.attn(q,q,q,need_weights=False)[0]
        x=x+self.cross(self.n2(x),env,env,need_weights=False)[0]/4
        return x+g2[:,None]*self.ff(self.n3(x)*(1+a2[:,None])+s2[:,None])

class PolicyV2(nn.Module):
    def __init__(self,c,mode='direct_flow'):
        super().__init__();self.config=c;self.mode=mode;self.near_zero_count=0;d=c['dim'];self.perception=PerceptionV2(c)
        self.goal=nn.Sequential(nn.Linear(3,d),nn.SiLU(),nn.Linear(d,d));self.time=nn.Sequential(nn.Linear(32,d),nn.SiLU(),nn.Linear(d,d));self.envcond=nn.Linear(d,d)
        self.latent=mode=='vae_latent_flow';n,ch=(4,8) if self.latent else (10,4);self.n=n;self.ch=ch
        self.input=nn.Linear(ch,d);self.pos=nn.Parameter(torch.randn(1,n,d)*.02);self.blocks=nn.ModuleList([DiTBlock(d) for _ in range(c['layers'])]);self.norm=nn.LayerNorm(d);self.output=nn.Linear(d,ch)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.register_buffer('center',torch.tensor([0.,0.,0.,1.]));self.register_buffer('scale',torch.tensor([1.,1.,.5,.5]));self.vae=TrajectoryVAE(32) if self.latent else None
        self.register_buffer('latent_mean',torch.zeros(32));self.register_buffer('latent_std',torch.ones(32))
    def condition(self,b):
        env=self.perception(b);goal=b['goal']/3;gc=self.goal(torch.cat([goal,goal.norm(dim=-1,keepdim=True)],-1));return env,gc
    def velocity(self,x,s,env,gc):
        f=torch.exp(torch.linspace(0,math.log(1000),16,device=x.device));t=s[:,None]*f;c=gc+self.time(torch.cat([t.sin(),t.cos()],-1))+self.envcond(env.mean(1));h=self.input(x)+self.pos+gc[:,None]
        for block in self.blocks:h=block(h,env,c)
        return self.output(self.norm(h))
    def loss(self,b):
        env,gc=self.condition(b);y=b['target']
        if self.latent:
            with torch.no_grad(),torch.autocast(device_type=y.device.type,enabled=False):y=((self.vae.encode(y.float())[0]-self.latent_mean)/self.latent_std).reshape(-1,4,8)
        else:y=(y-self.center)/self.scale
        if self.mode=='regression':return F.mse_loss(self.velocity(torch.zeros_like(y),torch.ones(len(y),device=y.device),env,gc),y)
        eps=torch.randn_like(y);s=torch.rand(len(y),device=y.device);x=(1-s[:,None,None])*eps+s[:,None,None]*y
        return F.mse_loss(self.velocity(x,s,env,gc),y-eps)
    @torch.no_grad()
    def predict(self,b,generator=None):
        env,gc=self.condition(b);n=len(gc);dev=gc.device
        if self.mode=='regression':x=self.velocity(torch.zeros(n,10,4,device=dev),torch.ones(n,device=dev),env,gc)
        else:
            x=torch.cat([torch.randn(1,self.n,self.ch,device=dev,generator=g) for g in generator]) if isinstance(generator,list) else torch.randn(n,self.n,self.ch,device=dev,generator=generator)
            for k in range(self.config['flow_steps']):x=x+self.velocity(x,torch.full((n,),k/self.config['flow_steps'],device=dev),env,gc)/self.config['flow_steps']
        x=self.vae.decode(x.flatten(1)*self.latent_std+self.latent_mean) if self.latent else x*self.scale+self.center
        v=x[:,:,2:];norm=v.norm(dim=-1,keepdim=True);bad=norm<1e-6;self.near_zero_count+=int(bad.sum());self.last_bad_counts=bad.flatten(1).sum(1).cpu().tolist();v=torch.where(bad,torch.tensor([0.,1.],device=dev),v/norm.clamp_min(1e-6));return torch.cat([x[:,:,:2],v],-1)

class StudentV2:
    def __init__(self,model,device):self.model=model.eval();self.device=device;self.vision=FrozenVision(model.config).to(device).eval()
    def predict(self,rgb,lidar_xyz,lidar_valid_mask,goal_xy,generator=None):
        from data.features import current_inputs
        b=current_inputs(rgb,lidar_xyz,lidar_valid_mask,goal_xy,self.device);s,g=self.vision([rgb]);b.update(sem=s,geo=g);return self.model.predict(b,generator)[0].cpu().numpy()
