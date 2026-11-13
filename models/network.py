import math
import torch
from torch import nn
from torch.nn import functional as F

def grid_pe(h,w,d):
    y,x=torch.meshgrid(torch.linspace(-1,1,h),torch.linspace(-1,1,w),indexing='ij');xy=torch.stack([x,y],-1).reshape(-1,2)
    f=2**torch.arange(d//4)*math.pi;return torch.cat([(xy[:,i,None]*f).sin() for i in range(2)]+[(xy[:,i,None]*f).cos() for i in range(2)],-1)

class VisionCNN(nn.Module):
    def __init__(self,d):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GELU(),nn.Conv2d(64,d,3,2,1),nn.GELU(),nn.AdaptiveAvgPool2d((6,8)))
    def forward(self,rgb):return self.net(rgb).flatten(2).transpose(1,2)

class OfficialVision(nn.Module):
    """Strict official loading: any missing/gated checkpoint is an error, never fallback."""
    def __init__(self,d):
        super().__init__()
        from transformers import AutoImageProcessor,AutoModel,AutoModelForDepthEstimation
        self.ids=['facebook/dinov3-vits16-pretrain-lvd1689m','depth-anything/Depth-Anything-V2-Small-hf']
        self.processors=[AutoImageProcessor.from_pretrained(x) for x in self.ids]
        self.dino=AutoModel.from_pretrained(self.ids[0]);self.da=AutoModelForDepthEstimation.from_pretrained(self.ids[1]);self.dino.requires_grad_(False);self.da.requires_grad_(False)
        self.semantic_projection=nn.Linear(self.dino.config.hidden_size,d);self.geometry_projection=nn.Conv2d(1,d,1)
    def train(self,mode=True):
        super().train(mode);self.dino.eval();self.da.eval();return self
    def forward(self,rgb):
        # This interface receives original RGB at the target configuration's preprocessing boundary.
        images=[x.detach().cpu().permute(1,2,0).numpy() for x in rgb]
        with torch.no_grad():
            inp=self.processors[0](images=images,do_rescale=False,return_tensors='pt').to(rgb.device)
            tokens=self.dino(**inp).last_hidden_state;registers=getattr(self.dino.config,'num_register_tokens',4);s=tokens[:,1+registers:]
            patch=self.dino.config.patch_size;ih,iw=inp['pixel_values'].shape[-2:];grid_h,grid_w=ih//patch,iw//patch
            inp2=self.processors[1](images=images,do_rescale=False,return_tensors='pt').to(rgb.device)
            dep=self.da(**inp2).predicted_depth.unsqueeze(1)
            # Relative DA2 signal, NOT metric projection or ground-truth depth.
            dep=(dep-dep.mean((2,3),keepdim=True))/(dep.std((2,3),keepdim=True)+1e-6)
        s=F.adaptive_avg_pool2d(s.transpose(1,2).reshape(len(rgb),-1,grid_h,grid_w),(6,8)).flatten(2).transpose(1,2)
        return self.semantic_projection(s),self.geometry_projection(F.adaptive_avg_pool2d(dep,(6,8))).flatten(2).transpose(1,2)

class LidarBEVEncoder(nn.Module):
    """Explicit RadScatter substitute: hit/height/density/free/observed + CNN."""
    def __init__(self,d):super().__init__();self.net=nn.Sequential(nn.Conv2d(5,d//2,3,padding=1),nn.GELU(),nn.Conv2d(d//2,d,3,padding=1))
    def forward(self,x):return self.net(x).flatten(2).transpose(1,2)

class Perception(nn.Module):
    def __init__(self,c):
        super().__init__();d=c['dim'];self.official=c['backbone']=='official'
        if self.official:self.vision=OfficialVision(d)
        else:self.semantic=VisionCNN(d);self.geometry=VisionCNN(d)
        self.sfilm=nn.Linear(d,2*d);self.gfilm=nn.Linear(d,2*d)
        self.queries=nn.Parameter(torch.randn(1,1600,d)*.02);self.register_buffer('bev_pe',grid_pe(40,40,d)[None]);self.register_buffer('img_pe',grid_pe(6,8,d)[None])
        self.image_attention=nn.MultiheadAttention(d,4,batch_first=True)
        self.lidar=LidarBEVEncoder(d);self.gate=nn.Linear(2*d,d)
        self.env_query=nn.Parameter(torch.randn(1,c['env_tokens'],d)*.02);self.qformer=nn.MultiheadAttention(d,4,batch_first=True);self.norm=nn.LayerNorm(d)
    def forward(self,rgb,bev):
        s,g=self.vision(rgb) if self.official else (self.semantic(rgb),self.geometry(rgb))
        sa,sb=self.sfilm(g.mean(1)).chunk(2,-1);ga,gb=self.gfilm(s.mean(1)).chunk(2,-1)
        ss=s*(1+.1*sa[:,None])+sb[:,None];gg=g*(1+.1*ga[:,None])+gb[:,None]
        img=torch.cat([ss+self.img_pe,gg+self.img_pe],1)
        q=(self.queries+self.bev_pe).expand(len(rgb),-1,-1);im=self.image_attention(q,img,img,need_weights=False)[0]
        li=self.lidar(bev)+self.bev_pe;gate=torch.sigmoid(self.gate(torch.cat([im,li],-1)));fused=gate*im+(1-gate)*li
        env=self.qformer(self.env_query.expand(len(rgb),-1,-1),fused,fused,need_weights=False)[0]
        return self.norm(env)

class TrajectoryVAE(nn.Module):
    def __init__(self,z=32):
        super().__init__();self.encoder=nn.Sequential(nn.Linear(40,128),nn.GELU(),nn.Linear(128,128),nn.GELU());self.mu=nn.Linear(128,z);self.logvar=nn.Linear(128,z);self.decoder=nn.Sequential(nn.Linear(z,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,40))
    def encode(self,y):
        h=self.encoder(y.flatten(1));return self.mu(h),self.logvar(h).clamp(-10,8)
    def decode(self,z):return self.decoder(z).reshape(-1,10,4)
    def forward(self,y):
        mu,lv=self.encode(y);z=mu+torch.randn_like(mu)*torch.exp(.5*lv);return self.decode(z),mu,lv

class Policy(nn.Module):
    def __init__(self,c,mode):
        super().__init__();self.config=c;self.mode=mode;d=c['dim'];self.perception=Perception(c);self.goal=nn.Sequential(nn.Linear(2,d),nn.GELU(),nn.Linear(d,d));self.time=nn.Sequential(nn.Linear(16,d),nn.SiLU(),nn.Linear(d,d))
        self.latent=mode=='vae_latent_flow'
        if self.latent and c['latent_dim']!=32:raise ValueError('This latent tokenization requires latent_dim=32 (4x8)')
        self.n=4 if self.latent else 10;self.channels=8 if self.latent else 4
        self.input=nn.Linear(self.channels,d);self.pos=nn.Parameter(torch.randn(1,self.n,d)*.02)
        layer=nn.TransformerDecoderLayer(d,4,dim_feedforward=d*4,dropout=0,batch_first=True,norm_first=True)
        self.transformer=nn.TransformerDecoder(layer,c['layers']);self.output=nn.Linear(d,self.channels)
        self.vae=TrajectoryVAE(c['latent_dim']) if self.latent else None
        self.register_buffer('latent_mean',torch.zeros(32));self.register_buffer('latent_std',torch.ones(32));self.near_zero_count=0
    def condition(self,b):return torch.cat([self.perception(b['rgb'],b['bev']),self.goal(b['goal']/3)[:,None]],1)
    def velocity(self,x,flow_s,env):
        f=2**torch.arange(8,device=x.device)*math.pi;t=flow_s[:,None]*f;te=self.time(torch.cat([t.sin(),t.cos()],-1))
        h=self.input(x)+self.pos+te[:,None];return self.output(self.transformer(h,env))
    def loss(self,b):
        env=self.condition(b);y=b['target']
        if self.latent:
            with torch.no_grad():y=((self.vae.encode(y)[0]-self.latent_mean)/self.latent_std).reshape(-1,4,8)
        if self.mode=='regression':return F.mse_loss(self.velocity(torch.zeros_like(y),torch.ones(len(y),device=y.device),env),y)
        eps=torch.randn_like(y);flow_s=torch.rand(len(y),device=y.device);xs=(1-flow_s[:,None,None])*eps+flow_s[:,None,None]*y
        return F.mse_loss(self.velocity(xs,flow_s,env),y-eps)
    @torch.no_grad()
    def predict(self,b,generator=None):
        env=self.condition(b);n=len(b['goal']);dev=env.device
        if self.mode=='regression':x=self.velocity(torch.zeros(n,10,4,device=dev),torch.ones(n,device=dev),env)
        else:
            x=torch.randn(n,self.n,self.channels,device=dev,generator=generator);steps=self.config['flow_steps']
            for i in range(steps):x=x+self.velocity(x,torch.full((n,),i/steps,device=dev),env)/steps
        if self.latent:x=self.vae.decode(x.flatten(1)*self.latent_std+self.latent_mean)
        heading=x[:,:,2:];norm=heading.norm(dim=-1,keepdim=True);bad=norm<1e-6;self.near_zero_count+=int(bad.sum());heading=heading/norm.clamp_min(1e-6);heading=torch.where(bad,torch.tensor([0.,1.],device=dev),heading)
        return torch.cat([x[:,:,:2],heading],-1)

class Student:
    """Public inference API accepts only current RGB, LiDAR, validity, and goal."""
    def __init__(self,model,device):self.model=model.eval();self.device=device
    def predict(self,rgb,lidar_xyz,lidar_valid_mask,goal_xy,generator=None):
        from data.features import current_inputs
        return self.model.predict(current_inputs(rgb,lidar_xyz,lidar_valid_mask,goal_xy,self.device,full_resolution=self.model.perception.official),generator)[0].cpu().numpy()
