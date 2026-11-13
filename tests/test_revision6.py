"""Regression checks for measured surface identity and crossing reservations."""
from collections import deque
import numpy as np
import torch
from types import SimpleNamespace
from navigation.odom_tracker import extended_vertical_mask,OdomTracker
from navigation.crossing_guard import CrossingGuard
from navigation.revision5 import Revision5Navigator
from navigation.revision5 import CompositeCost
from navigation.revision5_config import resolve_revision5_config
from navigation.surface_motion import SurfaceMotionMemory,fresh_motion_confirmation,compact_surface_observable


def test_rank_one_wall_ring_is_static_but_compact_object_is_not_removed():
    wall=np.c_[np.linspace(-3,3,100),np.full(100,5.),np.full(100,.65)]
    compact=np.c_[np.linspace(-.2,.2,20),np.full(20,2.),np.full(20,.65)]
    mask=extended_vertical_mask(np.r_[wall,compact])
    assert mask[:100].mean()>.95
    assert not mask[100:].any()


def test_confirmed_track_survives_short_missing_measurements_but_expires():
    tracker=OdomTracker(resolve_revision5_config())
    tracker.last_stamp=2.
    tracker.tracks=[dict(id=1,stamp=2.,is_dynamic=True)]
    observation=dict(scan_time=2.,time=2.7,odometry=np.zeros(3))
    tracker.update(observation);assert len(tracker.tracks)==1
    observation['time']=3.1;tracker.update(observation);assert not tracker.tracks


def walker(side=0.,velocity=.6,planes=()):
    times=np.arange(7)*.1
    points=[np.array([2.,side+velocity*(t-times[-1])]) for t in times]
    return dict(id=1,center=np.array([2.,side]),velocity=np.array([0.,velocity]),
                stamp=.6,is_dynamic=True,history=deque(zip(times,points)),turn_planes=list(planes))


def test_crossing_waits_before_line_instead_of_parking_on_it():
    guard=CrossingGuard();track=walker()
    guard.update([track],np.zeros(3),np.array([2.5,0.]),.6,.5)
    assert guard.waiting
    guard.update([track],np.array([1.05,0.,0.]),np.array([2.5,0.]),.6,.0)
    v,w=guard.cap_command(.5,.2)
    assert v==0 and w==0


def test_departure_is_not_enough_if_return_can_precede_full_crossing():
    guard=CrossingGuard();track=walker(side=.5)
    guard.update([track],np.zeros(3),np.array([2.5,0.]),.6,.0)
    assert guard.waiting and not guard.committed
    # Observed future turnaround plane is input evidence, not a true endpoint.
    guard=CrossingGuard()
    for index,side in enumerate(np.linspace(-.4,.4,8)):
        track=walker(side=side,planes=[(np.array([0.,1.]),4.5)]);track['stamp']=.6+index*.1
        guard.update([track],np.array([1.05,0.,0.]),np.array([2.5,0.]),track['stamp'],.0)
    assert guard.committed and not guard.waiting


def test_narrow_paving_allows_a_short_turn_that_brakes_angular_velocity():
    nav=Revision5Navigator.__new__(Revision5Navigator)
    nav.cfg={'robot':dict(length=.6,width=.35,amax=1.,alphamax=2.)}
    nav.research={'control':{'body_margin_m':.06}};nav.yield_axis=None
    n=101;res=.02;minimum=np.array([-1.01,-1.01]);yy,xx=np.mgrid[:n,:n]
    road=np.abs((yy+.5)*res+minimum[1])<.32
    terrain=np.zeros((n,n));obs={'measured_twist':np.zeros(2)}
    assert nav._rollout_safe(obs,(0.,1.),minimum,res,road,terrain,[])


def test_padding_recovery_never_relaxes_the_physical_body_envelope():
    nav=Revision5Navigator.__new__(Revision5Navigator)
    nav.cfg={'robot':dict(length=.6,width=.35,amax=1.,alphamax=2.)}
    nav.research={'control':{'body_margin_m':.06}};nav.yield_axis=None
    n=201;res=.01;minimum=np.array([-1.005,-1.005]);yy,xx=np.mgrid[:n,:n]
    x=(xx+.5)*res+minimum[0];y=(yy+.5)*res+minimum[1]
    side=x*np.sin(.3)+y*np.cos(.3)+.19
    road=np.abs(side)<.5;terrain=np.zeros((n,n));obs={'measured_twist':np.zeros(2)}
    command=nav._clearance_recovery(obs,minimum,res,road,terrain,[])
    assert command is not None and 0.<=command[0]<=.10
    assert nav._clearance_recovery(obs,minimum,res,np.abs(side)<.4,terrain,[]) is None


def test_dense_native_footprint_rejects_a_notch_between_old_corner_samples():
    nav=Revision5Navigator.__new__(Revision5Navigator)
    nav.cfg={'robot':dict(length=.6,width=.35,amax=1.,alphamax=2.)}
    nav.research={'control':{'body_margin_m':.06}};nav.yield_axis=None
    origin=np.array([-1.,-1.]);res=.01;n=201;xx,yy=np.meshgrid(np.arange(n)*res-1,np.arange(n)*res-1)
    safe=np.ones((n,n),bool);notch=(xx>.10)&(xx<.18)&(yy>.12)&(yy<.20);safe[notch]=False
    nav.road=SimpleNamespace(origin=origin,res=res,n=n,score=(safe.T.astype(int)*8),index=lambda p:np.rint((p-origin)/res).astype(int))
    obs={'odometry':np.zeros(3),'measured_twist':np.zeros(2)}
    coarse=np.ones((80,80),bool);terrain=np.zeros((80,80))
    assert not nav._rollout_safe(obs,(0.,0.),np.array([-4.,-4.]),.1,coarse,terrain,[])


def test_footprint_guidance_cost_changes_independent_heading_channels():
    axis=torch.linspace(-1.95,1.95,40);yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    fields=torch.stack([torch.zeros_like(xx),torch.zeros_like(xx),.3-yy.abs()])
    weights={key:0. for key in ['terrain','grass','dyn_ttc','retreat','pullover','goal','smooth']};weights['grass']=1.
    cost=CompositeCost(fields,[-2.,-2.],.1,[1.,0.],[],weights);cost.footprint_guidance=True
    paths=torch.zeros(2,10,4);paths[:,:,0]=.5;paths[:,:,1]=.05
    paths[0,:,3]=1.;paths[1,:,2]=1.
    values=cost(paths);assert values[1]>values[0]+.04
    paths=paths.detach();paths[:,:,2:]=.7071;paths.requires_grad_(True)
    cost(paths).sum().backward()
    assert paths.grad[:,:,2:].abs().sum()>0


def test_yield_uses_envelope_contact_time_before_centers_are_close():
    nav=Revision5Navigator.__new__(Revision5Navigator)
    nav.cfg={'robot':{'length':.6}};nav.research=resolve_revision5_config()
    nav.yield_anchor=nav.yield_axis=nav.yield_track=nav.pullover_world=None
    nav.event_best=nav.event_retreat=0.;nav.stats={'max_event_retreat_m':0.}
    target=dict(id=1,center=np.array([5.,0.]),velocity=np.array([-.5,0.]),stamp=0.)
    nav.tracker=SimpleNamespace(tracks=[target])
    observation={'odometry':np.zeros(3),'time':0.,'measured_twist':np.array([.5,0.])}
    nav._pullover(observation,np.array([2.5,0.]),np.ones((80,80),bool),np.ones((80,80)),[target],np.array([-4.,-4.]),.1)
    assert nav.yield_anchor is not None


def test_parallel_spatial_history_cannot_create_a_crossing_from_velocity_jitter():
    guard=CrossingGuard();track=walker(side=.2)
    track['history']=deque((t,np.array([1.+t,.2])) for t in np.arange(10)*.1)
    guard.update([track],np.zeros(3),np.array([2.5,0.]),.9,0.)
    assert guard.reservation is None


def test_free_ray_witness_distinguishes_static_patch_drift_from_arriving_surface():
    memory=SurfaceMotionMemory();yy,zz=np.meshgrid(np.linspace(-1,1,101),np.linspace(.1,1.2,61))
    wall=np.c_[np.full(yy.size,4.),yy.ravel(),zz.ravel()];origin=np.array([0.,0.,.65])
    for t in [0.,.3,.6]:memory.add(t,wall,origin)
    # The sampled subset moves along the same wall; its centroid changes, but
    # old rays never certify this occupied wall surface as free.
    patch=np.c_[np.full(20,4.),np.linspace(.1,.7,20),np.full(20,.65)]
    assert not memory.support(patch,1.1)['confirmed']
    arriving=patch.copy();arriving[:,0]=3.6
    evidence=memory.support(arriving,1.1)
    assert evidence['confirmed'] and evidence['count']>=2


def test_missing_returns_cannot_certify_motion_into_known_free_space():
    memory=SurfaceMotionMemory();memory.add(0.,np.empty((0,3)),np.zeros(3))
    assert not memory.support(np.array([[1.,0.,0.],[1.,.1,0.]]),1.)['confirmed']


def test_new_motion_requires_fresh_scans_and_does_not_reuse_stale_witnesses():
    state={}
    assert not fresh_motion_confirmation(state,1.,True)
    assert not fresh_motion_confirmation(state,1.,True)  # retransmitted packet
    assert not fresh_motion_confirmation(state,1.1,False)
    assert fresh_motion_confirmation(state,1.2,True)
    assert not fresh_motion_confirmation(state,1.7,True)


def test_single_ring_edge_cannot_establish_a_new_compact_moving_body():
    edge=np.c_[np.linspace(-.2,.2,10),np.zeros(10),np.full(10,.65)]
    assert not compact_surface_observable(edge)
    yy,zz=np.meshgrid(np.linspace(-.2,.2,6),np.linspace(.2,.75,4))
    body=np.c_[np.ones(yy.size),yy.ravel(),zz.ravel()]
    assert compact_surface_observable(body)
