"""Sensor-only tensor primitives for the next terrain navigation experiment.

These are observation statistics, NOT a certified ground/foothold estimator.
Unknown cells stay unknown. No simulator, SDK, map truth or robot commands.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F


def transform_points(points, transform):
    """Column-vector convention: T_target_source. Supports per-point deskew T.

    Time synchronization/interpolation must happen upstream: a single pose for
    a whole moving MID-360 scan does not perform per-return motion deskew.
    """
    if transform.shape[-2:]!=(4,4):raise ValueError('Expected T_target_source[...,4,4]')
    if points.shape[-1]!=3:raise ValueError('Expected xyz[...,3]')
    return torch.einsum('...ij,...j->...i',transform[...,:3,:3],points)+transform[...,:3,3]


def gravity_aligned_points(points_sensor,T_odom_sensor,T_odom_base_current):
    """Scan-time sensor pose into current yaw-only, gravity-aligned base frame.

    Full scan-time rotation removes sensor roll/pitch. Current origin includes
    base z; it is not a measured ground level and must never be called one.
    """
    world=transform_points(points_sensor,T_odom_sensor)
    current=T_odom_base_current
    yaw=torch.atan2(current[1,0],current[0,0]);c,s=torch.cos(yaw),torch.sin(yaw)
    zero=torch.zeros_like(c);one=torch.ones_like(c)
    rotation=torch.stack([torch.stack([c,-s,zero]),torch.stack([s,c,zero]),torch.stack([zero,zero,one])])
    return (world-current[:3,3])@rotation


def depth_to_optical_points(depth_m,intrinsics,valid=None):
    """Rectified pinhole z-depth in meters, optical x-right/y-down/z-forward.

    Caller supplies the D435i device depth scale and matching intrinsics before
    this function. Raw uint16 depth, DA2 output and zero returns are not meters.
    """
    if not depth_m.is_floating_point():raise ValueError('Depth must already be metric floating-point z')
    h,w=depth_m.shape
    yy,xx=torch.meshgrid(torch.arange(h,device=depth_m.device),torch.arange(w,device=depth_m.device),indexing='ij')
    mask=torch.isfinite(depth_m)&(depth_m>0)
    if valid is not None:mask&=valid
    z=depth_m[mask];u=xx[mask];v=yy[mask]
    return torch.stack([(u-intrinsics[0,2])*z/intrinsics[0,0],(v-intrinsics[1,2])*z/intrinsics[1,1],z],-1)


def consistent_motion(positions_odom,timestamps,min_segments=4,min_displacement=.25,
                      direction_cosine=.9,max_gap=.25,min_speed=.1):
    """Gate an already-associated track in ONE fixed odometry coordinate frame.

    This does not perform clustering/association or prove a target is dynamic.
    Direction reversals require fresh evidence rather than confirming through
    a gap. Numerical thresholds are experimental and need real-data tuning.
    """
    if len(positions_odom)<min_segments+1:return False
    # Direction uses the last four segments; net translation uses a longer
    # measured history. Restricting BOTH to 0.4s would never confirm a 0.4m/s
    # campus cart at a 0.25m displacement threshold.
    positions=positions_odom[-20:];times=timestamps[-20:]
    delta=positions[1:]-positions[:-1];dt=times[1:]-times[:-1]
    if not bool(torch.isfinite(positions).all()&torch.isfinite(times).all()):return False
    if not bool(((dt>0)&(dt<=max_gap)).all()):return False
    distance=torch.linalg.vector_norm(delta,dim=-1);speed=distance/dt
    unit=delta/distance.clamp_min(1e-9)[:,None]
    recent=unit[-min_segments:]
    aligned=(recent[1:]*recent[:-1]).sum(-1)>=direction_cosine
    return bool(aligned.all()&(speed[-min_segments:]>=min_speed).all()&
                (torch.linalg.vector_norm(positions[-1]-positions[0])>min_displacement))


@dataclass
class ElevationObservation:
    count: torch.Tensor
    mean_z: torch.Tensor
    min_z: torch.Tensor
    max_z: torch.Tensor
    observed: torch.Tensor
    resolution: float
    minimum_xy: tuple

    @property
    def height_range(self):return self.max_z-self.min_z

    def discontinuities(self,flat_threshold=.04,max_step=.15,max_drop=.15):
        """Measured neighbor differences; no return is NOT a negative obstacle.

        Upper/lower labels refer to observed surface cells, not gait feasibility.
        Wall tops, vegetation and mixed returns need further surface estimation.
        """
        up=torch.zeros_like(self.mean_z);drop=torch.zeros_like(up)
        for axis in [0,1]:
            a=[slice(None)]*2;b=[slice(None)]*2;a[axis]=slice(0,-1);b[axis]=slice(1,None)
            a,b=tuple(a),tuple(b);valid=self.observed[a]&self.observed[b]
            delta=torch.where(valid,self.min_z[b]-self.min_z[a],0.)
            up[a]=torch.maximum(up[a],delta.clamp_min(0));drop[a]=torch.maximum(drop[a],(-delta).clamp_min(0))
            up[b]=torch.maximum(up[b],(-delta).clamp_min(0));drop[b]=torch.maximum(drop[b],delta.clamp_min(0))
        return dict(step_up_m=up,drop_m=drop,stair_candidate=self.observed&(up>=flat_threshold)&(up<=max_step),
                    high_obstacle=self.observed&(up>max_step),negative_edge=self.observed&(drop>max_drop),unknown=~self.observed)


def elevation_observation(points,n=100,resolution=.1,minimum_xy=(-5.,-5.),min_hits=3):
    """GPU scatter statistics over actual finite returns; tensors are [y,x]."""
    if resolution<=0 or n<=0 or min_hits<1:raise ValueError('Invalid grid parameters')
    finite=torch.isfinite(points).all(-1);q=points[finite]
    xy=torch.floor((q[:,:2]-q.new_tensor(minimum_xy))/resolution).long()
    valid=(xy>=0).all(-1)&(xy<n).all(-1);xy=xy[valid];z=q[valid,2]
    index=xy[:,1]*n+xy[:,0];count=z.new_zeros(n*n);total=count.clone()
    count.scatter_add_(0,index,torch.ones_like(z));total.scatter_add_(0,index,z)
    low=z.new_full((n*n,),float('inf'));high=z.new_full((n*n,),-float('inf'))
    low.scatter_reduce_(0,index,z,reduce='amin',include_self=True);high.scatter_reduce_(0,index,z,reduce='amax',include_self=True)
    any_hit=count>0;observed=count>=min_hits
    low=torch.where(any_hit,low,0.);high=torch.where(any_hit,high,0.)
    return ElevationObservation(count.reshape(n,n),(total/count.clamp_min(1)).reshape(n,n),low.reshape(n,n),high.reshape(n,n),observed.reshape(n,n),resolution,minimum_xy)


class TrajectoryFieldCost:
    """Differentiable map sampling and FIXED odometry mission progress.

    Field is externally built from measured data, including explicit unknown
    costs and body inflation. A low soft cost never certifies hard clearance.
    """
    def __init__(self,field,resolution,minimum_xy,mission_axis,robot_odom_xy,
                 robot_yaw,mission_start_xy,best_progress,regression_allowance=.2):
        self.field=field;self.resolution=resolution;self.minimum_xy=minimum_xy
        self.axis=mission_axis/torch.linalg.vector_norm(mission_axis).clamp_min(1e-6)
        self.robot_xy=robot_odom_xy;self.yaw=robot_yaw;self.start=mission_start_xy
        self.best_progress=best_progress;self.allowance=regression_allowance

    def __call__(self,trajectory):
        xy=trajectory[...,:2];n=self.field.shape[-1]
        grid=(xy-xy.new_tensor(self.minimum_xy))/(n*self.resolution)*2-1
        values=F.grid_sample(self.field[None,None].expand(len(xy),-1,-1,-1),grid[:,None],mode='bilinear',padding_mode='border',align_corners=False)[:,0,0]
        outside=(grid.abs()-1).clamp_min(0).square().sum(-1)
        c,s=torch.cos(self.yaw),torch.sin(self.yaw)
        rotation=torch.stack([torch.stack([c,-s]),torch.stack([s,c])])
        world=xy@rotation.T+self.robot_xy
        progress=((world-self.start)*self.axis).sum(-1)
        # Current body-frame x>=0 cannot forbid turning around and returning.
        # Keep a mission-coordinate historical maximum across decisions.
        regress=(self.best_progress-self.allowance-progress).clamp_min(0)
        return values.mean(-1)+100*outside.mean(-1)+20*regress.square().mean(-1)
