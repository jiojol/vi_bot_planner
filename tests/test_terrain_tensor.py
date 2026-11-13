import math
import pytest
import torch
from navigation.terrain_tensor import transform_points,gravity_aligned_points,depth_to_optical_points,elevation_observation,TrajectoryFieldCost,consistent_motion
from navigation.guided_flow import sample_guided


def pose(roll=0.,pitch=0.,yaw=0.,xyz=(0.,0.,0.)):
    cr,sr=math.cos(roll),math.sin(roll);cp,sp=math.cos(pitch),math.sin(pitch);cy,sy=math.cos(yaw),math.sin(yaw)
    rx=torch.tensor([[1,0,0],[0,cr,-sr],[0,sr,cr]],dtype=torch.float64)
    ry=torch.tensor([[cp,0,sp],[0,1,0],[-sp,0,cp]],dtype=torch.float64)
    rz=torch.tensor([[cy,-sy,0],[sy,cy,0],[0,0,1]],dtype=torch.float64)
    t=torch.eye(4,dtype=torch.float64);t[:3,:3]=rz@ry@rx;t[:3,3]=torch.tensor(xyz);return t


def test_full_attitude_and_translation_recover_static_landmarks():
    landmarks=torch.tensor([[5.,0.,0.],[4.,1.,.1],[3.,-1.,1.]],dtype=torch.float64)
    t=pose(.1,-.08,.5,(1.,2.,.6));sensor=(landmarks-t[:3,3])@t[:3,:3]
    assert torch.max(torch.abs(transform_points(sensor,t)-landmarks))<1e-12
    restored=gravity_aligned_points(sensor,t,torch.eye(4,dtype=torch.float64))
    assert torch.max(torch.abs(restored-landmarks))<1e-12
    uncompensated=sensor+ t[:3,3]
    assert torch.linalg.vector_norm(uncompensated-landmarks,dim=1).max()>.25


def test_per_return_transform_handles_motion_in_one_scan():
    transforms=torch.stack([pose(xyz=(0,0,0)),pose(xyz=(1,0,0))])
    points=torch.tensor([[5.,0.,0.],[4.,0.,0.]],dtype=torch.float64)
    assert torch.allclose(transform_points(points,transforms),torch.tensor([[5.,0.,0.],[5.,0.,0.]],dtype=torch.float64))


def test_metric_depth_and_optical_axes_invalid_returns():
    depth=torch.tensor([[1.,0.],[float('nan'),2.]])
    k=torch.tensor([[2.,0.,0.],[0.,2.,0.],[0.,0.,1.]])
    assert torch.equal(depth_to_optical_points(depth,k),torch.tensor([[0.,0.,1.],[1.,1.,2.]]))
    with pytest.raises(ValueError):depth_to_optical_points(torch.ones(2,2,dtype=torch.uint16),k)


def test_unknown_is_not_free_and_missing_returns_are_not_drop_edges():
    q=torch.tensor([[.05,.05,0.],[.05,.05,.01],[.05,.05,-.01],[float('nan'),0,0]])
    grid=elevation_observation(q,n=4,minimum_xy=(0.,0.))
    assert grid.observed.sum()==1
    risk=grid.discontinuities();assert not risk['negative_edge'].any()
    assert risk['unknown'].sum()==15
    assert grid.mean_z[0,0]==0 and torch.isclose(grid.height_range[0,0],torch.tensor(.02))


def test_observed_height_transition_and_wall_range_are_explicit():
    q=torch.tensor([[.05,.05,0.]]*3+[[.15,.05,.12]]*3+[[.25,.05,.5]]*3)
    grid=elevation_observation(q,n=4,minimum_xy=(0.,0.));risk=grid.discontinuities()
    assert risk['stair_candidate'][0,0]
    assert risk['high_obstacle'][0,1]
    assert risk['negative_edge'][0,2]
    assert not risk['negative_edge'][3,3]


def field_cost(yaw):
    z=torch.zeros(100,100)
    return TrajectoryFieldCost(z,.1,(-5.,-5.),torch.tensor([1.,0.]),torch.zeros(2),torch.tensor(yaw),torch.zeros(2),best_progress=0.)


def test_turn_then_forward_is_penalized_in_fixed_mission_frame():
    trajectory=torch.zeros(1,10,4);trajectory[:,:,0]=1.;trajectory[:,:,3]=1.
    assert field_cost(0.)(trajectory)<.01
    assert field_cost(math.pi)(trajectory)>10.  # positive local x, mission retreat


def test_static_jitter_missing_frames_and_real_translation_are_distinct():
    times=torch.arange(5)*.1
    stationary=torch.tensor([[0.,0.],[.02,0.],[-.02,.01],[.01,-.01],[0.,.02]])
    moving=torch.stack([torch.arange(5)*.1,torch.zeros(5)],-1)
    assert not consistent_motion(stationary,times)
    assert consistent_motion(moving,times)
    assert not consistent_motion(moving,times*10)
    slow=torch.stack([torch.arange(11)*.04,torch.zeros(11)],-1)
    assert consistent_motion(slow,torch.arange(11)*.1)


class ZeroFlow:
    latent=False;n=10;ch=4
    scale=torch.ones(4);center=torch.zeros(4)
    def velocity(self,x,time,env,goal):return torch.zeros_like(x)


def test_guidance_reduces_cost_without_parameter_updates():
    model=ZeroFlow();env=torch.zeros(1,32,4);goal=torch.zeros(1,4)
    cost=lambda t:t[:,:,:2].square().sum((1,2))
    a,_=sample_guided(model,env,goal,steps=5,cost=None,generator=torch.Generator().manual_seed(13))
    b,_=sample_guided(model,env,goal,steps=5,cost=cost,strength=.5,generator=torch.Generator().manual_seed(13))
    assert cost(b).mean()<cost(a).mean()
    assert torch.isfinite(b).all()
    assert torch.allclose(torch.linalg.vector_norm(b[:,:,2:],dim=-1),torch.ones(16,10))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_grid_matches_cpu_with_repeated_and_invalid_returns():
    torch.manual_seed(3);q=torch.randn(20000,3);q[0]=float('nan')
    cpu=elevation_observation(q);gpu=elevation_observation(q.cuda())
    assert torch.equal(cpu.count,gpu.count.cpu())
    assert torch.equal(cpu.observed,gpu.observed.cpu())
    assert torch.allclose(cpu.mean_z,gpu.mean_z.cpu(),atol=1e-5)
    assert torch.equal(cpu.min_z,gpu.min_z.cpu())
