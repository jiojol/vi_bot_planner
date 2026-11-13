"""Revision 5 ablations: shared frozen perception, guided Flow, no local A*.

Soft guidance never certifies clearance: executed commands are separately
rolled out against measured fields with the same acceleration limits. A failed
candidate set brakes; there is no hidden Revision 4 planner fallback.
"""
import time
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt,map_coordinates
from navigation.revision5_config import resolve_revision5_config
from navigation.sensor_planner import RoadMemory, SensorPlanner, rotation, wrap, predicted_center
from navigation.elevation import ElevationMapper
from navigation.odom_tracker import OdomTracker
from navigation.guided_flow import sample_guided
from navigation.kinematic_guidance import refine_kinematic, track_trajectory
from navigation.calibrated_road import CalibratedRoadMemory
from data.features import current_inputs
from navigation.crossing_guard import CrossingGuard
from functools import lru_cache


@lru_cache(maxsize=16)
def body_samples(length,width,margin):
    x=np.linspace(-length/2-margin,length/2+margin,int(np.ceil((length+2*margin)/.035))+1)
    y=np.linspace(-width/2-margin,width/2+margin,int(np.ceil((width+2*margin)/.035))+1)
    xx,yy=np.meshgrid(x,y);return np.c_[xx.ravel(),yy.ravel()]


class CompositeCost:
    def __init__(self, fields, minimum_xy, resolution, goal, dynamic, weights, pull=None):
        self.fields = fields                  # terrain, grass, dynamic occupancy
        self.minimum = fields.new_tensor(minimum_xy)
        self.resolution = resolution
        self.goal = fields.new_tensor(goal)
        self.dynamic = dynamic
        self.weights = weights
        self.pull = fields.new_tensor(pull) if pull is not None else None
        self.footprint_guidance=False
        self.far_goal=False

    def components(self, trajectory):
        xy = trajectory[..., :2]
        k, n = xy.shape[:2]
        size = xy.new_tensor([self.fields.shape[-1], self.fields.shape[-2]])
        # Pixel-center convention matches align_corners=False exactly.
        grid = (xy-self.minimum)/(size*self.resolution)*2-1
        sampled = F.grid_sample(self.fields[None].expand(k, -1, -1, -1), grid[:, None],
                                 align_corners=False, padding_mode='border')[:, :, 0]
        footprint_cost=xy.new_zeros(k)
        if self.footprint_guidance and len(self.fields)>=3:
            angle=torch.atan2(trajectory[...,2],trajectory[...,3]);c,s=angle.cos(),angle.sin()
            corners=xy.new_tensor([[-.3,-.175],[-.3,.175],[.3,-.175],[.3,.175],[0.,-.175],[0.,.175],[.3,0.],[-.3,0.]])
            bx=c[:,:,None]*corners[:,0]-s[:,:,None]*corners[:,1]
            by=s[:,:,None]*corners[:,0]+c[:,:,None]*corners[:,1]
            body=torch.stack([bx,by],-1)+xy[:,:,None]
            bg=(body-self.minimum)/(size*self.resolution)*2-1
            sdf=F.grid_sample(self.fields[None,2:3].expand(k,-1,-1,-1),bg.reshape(k,1,-1,2),align_corners=False,padding_mode='border')[:,0,0]
            footprint_cost=F.softplus((.06-sdf)*25).amax(-1)/25
        outside = (grid.abs()-1).clamp_min(0).square().sum(-1).mean(-1)*100
        retreat = (-(xy[..., 0]+.15)).clamp_min(0).square().mean(-1)
        # Discourage backward segments even if endpoints stay ahead of origin.
        retreat = retreat+(xy[:, :-1, 0]-xy[:, 1:, 0]-.03).clamp_min(0).square().mean(-1)
        dyn_cost = xy.new_zeros(k)
        times = torch.arange(1, n+1, device=xy.device, dtype=xy.dtype)*.2
        for position, velocity, shape in self.dynamic:
            future = xy.new_tensor(position)[None]+times[:, None]*xy.new_tensor(velocity)[None]
            if np.ndim(shape) == 0:
                clearance = torch.linalg.vector_norm(xy-future[None], dim=-1)-shape
            else:
                longitudinal, lateral, heading = shape
                axis = xy.new_tensor([np.cos(heading), np.sin(heading)])
                normal = xy.new_tensor([-np.sin(heading), np.cos(heading)])
                relative = xy-future[None]
                robot_heading = torch.atan2(trajectory[..., 2], trajectory[..., 3])-heading
                extent_x = longitudinal+.30*robot_heading.cos().abs()+.175*robot_heading.sin().abs()
                extent_y = lateral+.30*robot_heading.sin().abs()+.175*robot_heading.cos().abs()
                clearance = torch.maximum((relative*axis).sum(-1).abs()-extent_x,
                                          (relative*normal).sum(-1).abs()-extent_y)
            dyn_cost = dyn_cost+F.softplus((.10-clearance)*10).mean(-1)/10
        target = self.pull if self.pull is not None else self.goal
        # Match the short trajectory horizon, not a distant mission endpoint.
        if not self.far_goal:
            target = target*torch.clamp(1./torch.linalg.vector_norm(target).clamp_min(1e-6), max=1.)
        progress = ((xy[:, -1]-target)**2).sum(-1)
        smooth = (xy[:, 2:]-2*xy[:, 1:-1]+xy[:, :-2]).square().mean((1, 2))
        pull = ((xy[:, -1]-self.pull)**2).sum(-1) if self.pull is not None else xy.new_zeros(k)
        return dict(terrain=sampled[:, 0].mean(-1)+outside, grass=sampled[:, 1].mean(-1)+footprint_cost,
                    dyn_ttc=dyn_cost, retreat=retreat, pullover=pull, goal=progress, smooth=smooth)

    def __call__(self, trajectory):
        parts = self.components(trajectory)
        return sum(self.weights[key]*value for key, value in parts.items())


class Revision5Navigator:
    def __init__(self, config, policy, ground_head, research=None, mode=None, device='cuda', seed=1):
        self.cfg = config
        self.research = resolve_revision5_config(research, mode)
        self.mode = self.research['mode']
        self.policy = policy
        self.device = device
        self.seed = seed
        self.ground_head = ground_head
        self.terrain_adapter = None
        adapter_path = self.research.get('terrain_adapter_checkpoint')
        if adapter_path:
            if not self.research['use_elevation']:
                raise ValueError('Terrain adapter is only valid in the full 2.5D mode')
            from navigation.terrain_adapter import TerrainConditionAdapter
            state = torch.load(adapter_path, map_location='cpu', weights_only=True)
            if state.get('data_split') != 'train' or state.get('architecture') != 'terrain_condition_adapter_v1':
                raise ValueError('Terrain adapter requires explicit new training-data provenance')
            self.terrain_adapter = TerrainConditionAdapter(self.policy.model.config['dim']).to(device).eval().requires_grad_(False)
            self.terrain_adapter.load_state_dict(state['model'])
        self.reset()

    def reset(self):
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.v4 = SensorPlanner({**self.cfg, 'ground_head':self.ground_head}) if self.mode == 's1_astar_v4' else None
        self.road = CalibratedRoadMemory(self.cfg['camera'], head=self.ground_head) if self.research['use_guidance'] else None
        if self.v4 is not None and self.research['simulation_sensors'].get('primary_rgb_source') == 'd435i_downward':
            self.v4.road = CalibratedRoadMemory(self.cfg['camera'], head=self.ground_head)
        self.tracker = OdomTracker(self.research)
        self.mapper = ElevationMapper(self.research, self.ground_head)
        self.yield_anchor = None
        self.yield_axis = None
        self.yield_track = None
        self.pullover_world = None
        self.event_best = 0.
        self.event_retreat = 0.
        self.previous_pose = None
        self.terrain_boundary = None
        self.last_neural = None
        self.last_grid = None
        self.last_candidates = None
        self.crossing_guard=CrossingGuard()
        self.stats = dict(neural_calls=0, guided_steps=0, yield_steps=0, rejected_candidates=0,
                          safe_stop_steps=0, dynamic_frames=0, max_event_retreat_m=0.,
                          astar_calls=0, stair_follow_steps=0)

    def _sync(self):
        if str(self.device).startswith('cuda'):
            torch.cuda.synchronize()

    def _perception(self, observation, goal):
        b = current_inputs(observation['rgb'], observation['lidar_xyz'],
                           observation['lidar_valid_mask'], goal, self.device)
        with torch.no_grad():
            sem, geo = self.policy.vision([observation['rgb']])
            b.update(sem=sem, geo=geo)
            env, _ = self.policy.model.condition(b)
            if self.terrain_adapter is not None and self.last_grid is not None:
                terrain = torch.as_tensor(self.last_grid.channels[None], device=self.device)
                observed = torch.as_tensor(self.last_grid.observed[None], device=self.device)
                env = self.terrain_adapter.condition(env, terrain, observed)
        return env

    def _fields(self, observation):
        pose = np.asarray(observation['odometry'])
        self.road.latest_observation = observation if 'downward_rgb_calibrated' in str(observation.get('sensor_protocol', '')) else None
        self.road.update(observation['rgb'], observation['camera_pose'], pose,
                         observation['lidar_xyz'], observation['lidar_valid_mask'])
        self.tracker.update(observation)
        # Grid is always built in the full mode; never injected into 2D mode.
        grid = self.mapper.update(observation, self.tracker.tracks) if self.research['use_elevation'] else None
        self.last_grid = grid
        settings = self.research['grid']
        n, res = settings['cells'], settings['resolution_m']
        minimum = np.asarray(settings['minimum_xy_m'])
        yy, xx = np.mgrid[:n, :n]
        local = np.stack([xx+.5, yy+.5], -1)*res+minimum
        world = local@rotation(pose[2]).T+pose[:2]
        ij = self.road.index(world)
        inside = (ij >= 0).all(-1) & (ij < self.road.n).all(-1)
        road = np.zeros((n, n), bool)
        road[inside] = self.road.score[ij[inside, 0], ij[inside, 1]] > 0
        clearance = distance_transform_edt(road)*res
        radius = np.hypot(self.cfg['robot']['length'], self.cfg['robot']['width'])/2
        grass = np.maximum(0., radius+.035-clearance)/(radius+.035)
        occupied = np.zeros((n, n), bool)
        points = self.tracker.points
        moving = [t for t in self.tracker.tracks if t['is_dynamic']]
        if len(points):
            keep = np.ones(len(points), bool)
            for track in moving:
                axis = track['velocity']/max(.01, np.linalg.norm(track['velocity']))
                normal = np.array([-axis[1], axis[0]])
                relative = points-track['center']
                keep &= ~((np.abs(relative@axis) < 1.1) & (np.abs(relative@normal) < .40))
            local_points = (points[keep]-pose[:2])@rotation(pose[2])
            ids = np.floor((local_points-minimum)/res).astype(int)
            good = (ids >= 0).all(-1) & (ids < n).all(-1)
            occupied[ids[good, 1], ids[good, 0]] = True
        obstacle_distance = distance_transform_edt(~occupied)*res if occupied.any() else np.full((n, n), 20.)
        terrain = np.maximum(0., radius+.045-obstacle_distance)/(radius+.045)
        if grid is not None:
            terrain = np.maximum(terrain, grid.terrain_cost(self.research['control']['allow_stair_proxy']))
            grass = np.maximum(grass, grid.channels[4])
        road_sdf=(distance_transform_edt(road)-distance_transform_edt(~road))*res
        fields = torch.as_tensor(np.stack([terrain, grass, road_sdf]), dtype=torch.float32, device=self.device)
        dynamic = []
        for track in moving:
            position = (predicted_center(track, observation['time']-track['stamp'])-pose[:2])@rotation(pose[2])
            velocity = track['velocity']@rotation(pose[2])
            heading = float(np.arctan2(velocity[1], velocity[0]))
            # Use an oriented compact-object envelope. A circumscribed cart
            # disk wrongly blocks safe side-by-side passing on a 2.4m road.
            axis = np.array([np.cos(heading), np.sin(heading)])
            normal = np.array([-axis[1], axis[0]])
            points_local = (track['points']-pose[:2])@rotation(pose[2])
            long_extent = max(1.10, float(np.ptp(points_local@axis))/2+.03)
            side_extent = max(.30, float(np.ptp(points_local@normal))/2+.03)
            dynamic.append((position, velocity, np.array([long_extent, side_extent, heading])))
        self.stats['dynamic_frames'] += bool(dynamic)
        return fields, minimum, res, road, clearance, terrain, moving, dynamic

    def _pullover(self, observation, goal, road, clearance, moving, minimum, res):
        pose = np.asarray(observation['odometry'])
        now = observation['time']
        axis = goal/max(1e-6, np.linalg.norm(goal))
        threats = []
        for track in moving:
            delta = (predicted_center(track, now-track['stamp'])-pose[:2])@rotation(pose[2])
            velocity = track['velocity']@rotation(pose[2])
            approach = -velocity@axis+max(0., observation['measured_twist'][0])
            lateral = abs(np.cross(axis, delta))
            # Time to contact of the SAME conservative envelopes used by the
            # hard checker, rather than time until centers coincide. The old
            # center TTC began the lateral maneuver about 1.5m too late.
            contact_distance=delta@axis-1.10-self.cfg['robot']['length']/2-.10
            ttc = max(0.,contact_distance)/max(.01, approach)
            if velocity@axis < -.2 and 0 < ttc < self.research['control']['ttc_s'] and lateral < .75:
                threats.append((ttc, track))
        if self.yield_anchor is None and threats:
            _, track = min(threats, key=lambda pair:pair[0])
            self.yield_anchor = pose[:2].copy()
            self.yield_axis = rotation(pose[2])@axis
            self.yield_track = track['id']
            self.event_best = self.event_retreat = 0.
        if self.yield_anchor is None:
            return None
        progress = float((pose[:2]-self.yield_anchor)@self.yield_axis)
        self.event_best = max(self.event_best, progress)
        self.event_retreat = max(self.event_retreat, self.event_best-progress)
        self.stats['max_event_retreat_m'] = max(self.stats['max_event_retreat_m'], self.event_retreat)
        track = next((t for t in self.tracker.tracks if t['id'] == self.yield_track), None)
        if track is None:
            candidates = [t for t in moving if t['velocity']@self.yield_axis < -.2 and
                abs(np.cross(self.yield_axis, t['center']-pose[:2])) < 1.5]
            if candidates:
                track = min(candidates, key=lambda t:np.linalg.norm(t['center']-pose[:2]))
                self.yield_track = track['id']
        if track is not None and (track['center']-pose[:2])@self.yield_axis < -1.6:
            self.yield_anchor = self.yield_axis = self.yield_track = None
            self.pullover_world = None
            return None
        if self.pullover_world is not None:
            return (self.pullover_world-pose[:2])@rotation(pose[2])
        cells = np.argwhere(road & (clearance >= self.research['control']['grass_margin_m']+.175))
        local = np.c_[cells[:, 1]+.5, cells[:, 0]+.5]*res+minimum
        # The refuge search is strictly forward, right of current corridor.
        selected = (local[:, 0] >= .50) & (local[:, 0] <= 2.) & (local[:, 1] < -.35) & (local[:, 1] > -2.)
        local = local[selected]
        if not len(local):
            return np.zeros(2)
        score = np.abs(local[:, 0]-.9)+np.abs(local[:, 1]+.75)
        chosen = local[np.argmin(score)]
        self.pullover_world = pose[:2]+rotation(pose[2])@chosen
        return chosen

    def _body_known(self,observation,q,margin,minimum,res,road):
        robot=self.cfg['robot'];points=body_samples(robot['length'],robot['width'],margin)
        points=points@rotation(q[2]).T+q[:2]
        memory=getattr(self,'road',None)
        if memory is not None and memory.origin is not None:
            pose=observation['odometry'];world=points@rotation(pose[2]).T+pose[:2]
            if margin<=.035:
                # Use subcell interpolation only between four actually observed
                # cells. This reduces nearest-cell aliasing without painting
                # unknown cells as paved or consulting simulator boundaries.
                fractional=(world-memory.origin)/memory.res
                lo=np.floor(fractional).astype(int);valid=(lo>=0).all(1)&(lo+1<memory.n).all(1)
                known=np.zeros(len(lo),bool)
                if np.any(valid):
                    q=lo[valid];observed=np.all(np.stack([memory.score[q[:,0]+dx,q[:,1]+dy]!=0 for dx,dy in [(0,0),(0,1),(1,0),(1,1)]]),axis=0)
                    value=map_coordinates(memory.score,fractional[valid].T,order=1,mode='constant',cval=-24.,output=np.float32,prefilter=False)
                    known[valid]=observed&(value>2.)
                return known
            ids=memory.index(world);valid=(ids>=0).all(1)&(ids<memory.n).all(1)
            known=np.zeros(len(ids),bool);known[valid]=memory.score[ids[valid,0],ids[valid,1]]>0
            return known
        ids=np.floor((points-minimum)/res).astype(int);valid=(ids>=0).all(1)&(ids<road.shape).all(1)
        known=np.zeros(len(ids),bool);known[valid]=road[ids[valid,1],ids[valid,0]];return known

    def _rollout_safe(self, observation, command, minimum, res, road, terrain, dynamic, future_controls=None,margin_recovery=False):
        q = np.zeros(3)
        v, w = observation['measured_twist']
        robot = self.cfg['robot']
        base_corners=np.array([[-1,-1],[-1,1],[1,-1],[1,1],[0,-1],[0,1],[1,0],[-1,0]],float)
        pad=self.research['control'].get('body_margin_m',.06)
        start_body=base_corners*np.array([robot['length']/2+pad,robot['width']/2+pad])
        start_ids=np.floor((start_body-minimum)/res).astype(int)
        if np.any(start_ids<0) or np.any(start_ids>=road.shape):return False
        start_bad=int(np.sum(~self._body_known(observation,np.zeros(3),pad,minimum,res,road)))
        # Include braking after the current decision. A trajectory's endpoints
        # alone are insufficient: a turn can cut the lawn between samples.
        for step in range(200 if future_controls is not None else 50):
            dt = .02
            if future_controls is not None:
                pv, pw = command if step < 5 else (future_controls[min(9, step//10)] if step < 100 else (0., 0.))
            else:
                # A stop brakes angular velocity too. Holding yaw rate for a
                # full second spuriously rejects safe short turns on narrow
                # paving although the next decision can stop that rotation.
                pv, pw = command if step < 5 else (0., 0.)
            v += np.clip(pv-v, -robot['amax']*dt, robot['amax']*dt)
            w += np.clip(pw-w, -robot['alphamax']*dt, robot['alphamax']*dt)
            mid = q[2]+w*dt/2
            q[:2] += v*dt*np.array([np.cos(mid), np.sin(mid)])
            q[2] += w*dt
            ij = np.floor((q[:2]-minimum)/res).astype(int)
            if np.any(ij < 0) or np.any(ij >= road.shape):
                return False
            if not road[ij[1], ij[0]] or terrain[ij[1], ij[0]] >= .99:
                return False
            # Use a dense oriented rectangular footprint for grass, preserving
            # legal narrow turns that an overinflated disk would reject.
            corners = np.array([[-.30, -.175], [-.30, .175], [.30, -.175], [.30, .175],
                                [0., -.175], [0., .175], [.30, 0.], [-.30, 0.]])
            margin = self.research['control'].get('body_margin_m', .06)
            corners *= np.array([(robot['length']/2+margin)/.30, (robot['width']/2+margin)/.175])
            body = corners@rotation(q[2]).T+q[:2]
            bi = np.floor((body-minimum)/res).astype(int)
            if np.any(bi < 0) or np.any(bi >= road.shape):return False
            bad=int(np.sum(~self._body_known(observation,q,margin,minimum,res,road)))
            if bad:
                if not margin_recovery or command[0]>.10 or not start_bad or bad>start_bad:return False
                # Padding is an uncertainty margin, not the physical body. A
                # bounded crawl/turn may reduce an existing padding violation,
                # while retaining an independently checked 2cm subcell
                # envelope around the entire physical footprint throughout.
                if not np.all(self._body_known(observation,q,.02,minimum,res,road)):return False
            for position, velocity, shape in dynamic:
                relative = q[:2]-position-velocity*(step+1)*dt
                if np.ndim(shape) == 0:
                    separation = np.linalg.norm(relative)-shape
                else:
                    length, width, heading = shape
                    local = relative@rotation(heading)
                    angle = q[2]-heading
                    half_x = length+.30*abs(np.cos(angle))+.175*abs(np.sin(angle))
                    half_y = width+.30*abs(np.sin(angle))+.175*abs(np.cos(angle))
                    separation = max(abs(local[0])-half_x, abs(local[1])-half_y)
                if separation < .07:
                    return False
            if self.yield_axis is not None:
                delta = rotation(observation['odometry'][2])@q[:2]
                progress = (observation['odometry'][:2]+delta-self.yield_anchor)@self.yield_axis
                if self.event_best-progress >= self.research['control']['max_retreat_m']-.02:
                    return False
        return True

    def _clearance_recovery(self,observation,minimum,res,road,terrain,dynamic):
        robot=self.cfg['robot'];pad=self.research['control'].get('body_margin_m',.06)
        corners=np.array([[-1,-1],[-1,1],[1,-1],[1,1],[0,-1],[0,1],[1,0],[-1,0]],float)*np.array([robot['length']/2+pad,robot['width']/2+pad])
        initial=np.floor((corners-minimum)/res).astype(int)
        if np.any(initial<0) or np.any(initial>=road.shape) or np.all(self._body_known(observation,np.zeros(3),pad,minimum,res,road)):return None
        signed=(distance_transform_edt(road)-distance_transform_edt(~road))*res
        def score(angle):
            points=corners@rotation(angle).T
            ids=np.floor((points-minimum)/res).astype(int)
            if np.any(ids<0) or np.any(ids>=road.shape):return -1e6
            return float(np.min(signed[ids[:,1],ids[:,0]])+.1*np.mean(signed[ids[:,1],ids[:,0]]))
        candidates=[]
        for v in [.10,.05,0.]:
            for w in [-1.,-.5,0.,.5,1.]:
                if v==0 and w==0:continue
                # Local clearance descent only; no grid path or hidden route.
                angle=w*.15;position=np.array([v*.3,0.])
                q=np.r_[position,angle]
                deficit=np.sum(~self._body_known(observation,q,pad,minimum,res,road))
                candidates.append((deficit,-v,-score(angle),v,w))
        for _,_,_,v,w in sorted(candidates):
            if self._rollout_safe(observation,(v,w),minimum,res,road,terrain,dynamic,margin_recovery=True):return (v,w)
        return None

    def _terrain_remaining(self, observation, measured_edge):
        pose = np.asarray(observation['odometry'])
        forward = rotation(pose[2])[:, 0]
        if self.terrain_boundary is not None:
            point, _ = self.terrain_boundary
            if any(track['is_dynamic'] and np.linalg.norm(track['center']-point) < 1.1 for track in self.tracker.tracks):
                self.terrain_boundary = None
        if measured_edge is not None and measured_edge < 1.35 and self.mapper.forward_hazard_kind != 'high':
            point = pose[:2]+measured_edge*forward
            dynamic_occlusion = any(track['is_dynamic'] and np.linalg.norm(track['center']-point) < 1.1 for track in self.tracker.tracks)
            if self.terrain_boundary is None and not dynamic_occlusion:
                self.terrain_boundary = (point, forward.copy())
            elif self.terrain_boundary is not None and not dynamic_occlusion:
                previous, normal = self.terrain_boundary
                if forward@normal > .85 and (point-previous)@normal < 0.:
                    self.terrain_boundary = (point, normal)
        if self.terrain_boundary is None:
            return None
        point, normal = self.terrain_boundary
        side = rotation(pose[2])[:, 1]
        body_extent = self.cfg['robot']['length']/2*abs(forward@normal)+self.cfg['robot']['width']/2*abs(side@normal)
        return float((point-pose[:2])@normal-body_extent-self.research['control']['stop_edge_gap_m'])

    @torch.inference_mode(False)
    def command(self, observation, local_goal):
        self._sync()
        start = time.perf_counter()
        goal = np.asarray(local_goal, dtype=np.float32)
        if not self.research['use_guidance']:
            self.stats['neural_calls'] += 1
            proposal = self.policy.predict(observation['rgb'], observation['lidar_xyz'],
                observation['lidar_valid_mask'], goal, generator=self.generator)
            if self.v4 is not None:
                if isinstance(self.v4.road, CalibratedRoadMemory):
                    self.v4.road.latest_observation = observation if 'downward_rgb_calibrated' in str(observation.get('sensor_protocol', '')) else None
                command, info = self.v4.command(observation, goal, proposal)
                self.stats['astar_calls'] += 1
            else:
                command, info = track_trajectory(proposal), dict(mode='vanilla_s1')
            self._sync()
            return dict(vx=command[0], vy=0., yaw_rate=command[1], neural_trajectory=proposal,
                        diagnostic={**info, 'latency_ms':(time.perf_counter()-start)*1000})
        fields, minimum, res, road, clearance, terrain, moving, dynamic = self._fields(observation)
        self.crossing_guard.update(self.tracker.tracks,observation['odometry'],goal,
                                   observation['time'],observation['measured_twist'][0])
        measured_edge = self.mapper.forward_hazard_distance(observation) if self.last_grid is not None else None
        terrain_remaining = self._terrain_remaining(observation, measured_edge) if not self.research['control']['allow_stair_proxy'] else None
        if terrain_remaining is not None and terrain_remaining < .025 and abs(observation['measured_twist'][0]) < .03 and self.last_neural is not None:
            self.stats['safe_stop_steps'] += 1
            self._sync()
            return dict(vx=0., vy=0., yaw_rate=0., neural_trajectory=self.last_neural,
                diagnostic=dict(mode='SAFE_STOP', moving_tracks=len(dynamic), candidates=0,
                    astar_used=False, neural_skipped_latched_static_stop=True,
                    measured_edge_distance_m=measured_edge, terrain_remaining_m=terrain_remaining,
                    latency_ms=(time.perf_counter()-start)*1000))
        pull = self._pullover(observation, goal, road, clearance, moving, minimum, res)
        cost = CompositeCost(fields, minimum, res, goal, dynamic, self.research['cost_weights'], pull)
        turning=abs(np.arctan2(goal[1],goal[0]))>.30 and not dynamic and pull is None
        cost.footprint_guidance=turning
        environment = self._perception(observation, goal)
        self.stats['neural_calls'] += 1
        sampling = self.research['sampling']
        k = sampling['candidates']
        targets = torch.tensor(np.tile(goal, (k, 1)), device=self.device)
        targets[:, 1] += torch.linspace(-.8, .8, k, device=self.device)
        if pull is not None:
            targets[:] = targets.new_tensor(pull)
        with torch.no_grad():
            normalized = targets/3
            conditions = self.policy.model.goal(torch.cat([normalized, normalized.norm(dim=-1, keepdim=True)], -1))
        trajectories, scores = sample_guided(self.policy.model, environment, conditions, candidates=k,
            steps=sampling['euler_steps'], cost=cost, strength=sampling['guidance_strength'],
            max_gradient_norm=sampling['maximum_guidance_gradient_norm'], generator=self.generator)
        self.stats['guided_steps'] += sampling['euler_steps']
        controls = None
        if pull is not None and np.linalg.norm(pull) > .15:
            trajectories, scores, controls = refine_kinematic(trajectories, cost, observation['measured_twist'],
                iterations=sampling.get('kinematic_iterations', 16), step_size=sampling.get('kinematic_step_size', .30))
            controls = controls.cpu().numpy()
        elif turning and terrain_remaining is None and not self.crossing_guard.waiting:
            cost.far_goal=True
            trajectories,scores,controls=refine_kinematic(trajectories,cost,observation['measured_twist'],iterations=8,step_size=.30)
            controls=controls.cpu().numpy()
        candidates = trajectories.cpu().numpy()
        self.last_candidates = candidates
        ordered = scores.argsort().cpu().numpy()
        chosen = int(ordered[0])
        command = (0., 0.)
        state = 'YIELD_WAIT' if pull is not None else 'NORMAL_TROT'
        safe_count = 0
        tested = 0
        for index in ordered:
            tested += 1
            trajectory = candidates[index]
            if not np.isfinite(trajectory).all() or (trajectory[:, 0] < -.15).any():
                continue
            v, w = track_trajectory(trajectory)
            if controls is not None:
                v, w = controls[index, 0]
            if pull is not None and np.linalg.norm(pull) < .10:
                v = 0.
                heading = np.arctan2(self.yield_axis[1], self.yield_axis[0])
                w = float(np.clip(2.5*wrap(heading-observation['odometry'][2]), -1., 1.))
            if self.last_grid is not None:
                # Inspect the swept leading corridor, including .45m body-edge
                # clearance and braking distance, independently of soft costs.
                braking = observation['measured_twist'][0]**2/(2*self.cfg['robot']['amax'])
                distance = self.cfg['robot']['length']/2+self.research['control']['stop_edge_gap_m']+braking
                xs = np.arange(.55, distance+.05, .05)
                path = np.c_[xs, np.clip(w, -.6, .6)*xs**2*.5]
                mode = self.mapper.decision(self.last_grid, path)
                if mode == 'STAIR_CLIMB' and self.research['control']['allow_stair_proxy']:
                    v = min(v, self.research['control']['stair_speed_mps'])
                    state = mode
                elif mode != 'NORMAL_TROT':
                    v, w = 0., 0.
                    state = 'SAFE_STOP' if mode != 'IMPASSABLE_HIGH' else mode
                if terrain_remaining is not None and not self.research['control']['allow_stair_proxy']:
                    remaining = terrain_remaining
                    # Maintain measured clearance through acceleration-limited
                    # braking; include half a decision step of stopping slack.
                    allowed = np.sqrt(2*self.cfg['robot']['amax']*max(0., remaining-.025))
                    nominal_v, nominal_w = track_trajectory(trajectory)
                    v = min(nominal_v, float(allowed))
                    w = float(np.clip(2.5*np.arctan2(goal[1], goal[0]), -1., 1.)) if v > .03 else 0.
                    state = 'SAFE_STOP' if v < .01 else 'NORMAL_TROT'
            if np.linalg.norm(goal) < .17:
                v, w, state = 0., 0., 'terminal_stop'
            elif self.crossing_guard.waiting:
                v,w=self.crossing_guard.cap_command(v,w,self.cfg['robot']['amax'])
                state='CROSSING_WAIT' if v<.01 else 'CROSSING_APPROACH'
            if self._rollout_safe(observation, (v, w), minimum, res, road, terrain, dynamic,
                                  controls[index] if controls is not None else None):
                command = (v, w)
                chosen = int(index)
                safe_count += 1
                break
        self.stats['rejected_candidates'] += tested-safe_count
        if not safe_count and not dynamic and pull is None and terrain_remaining is None and not self.crossing_guard.waiting:
            # Safety projection of the SAME neural candidates through unicycle
            # controls. Body/heading costs use the observed footprint field;
            # this is no grid search, privileged path or A* fallback.
            recovery_cost=CompositeCost(fields,minimum,res,goal,dynamic,self.research['cost_weights'])
            recovery_cost.footprint_guidance=True
            recovery_cost.far_goal=True
            paths,values,commands=refine_kinematic(trajectories,recovery_cost,observation['measured_twist'],iterations=8,step_size=.30)
            projected=paths.cpu().numpy();commands=commands.cpu().numpy()
            for index in values.argsort().cpu().numpy():
                v,w=commands[index,0]
                if self._rollout_safe(observation,(v,w),minimum,res,road,terrain,dynamic):
                    command=(float(v),float(w));chosen=int(index);candidates[chosen]=projected[chosen]
                    safe_count=1;state='KINEMATIC_RECOVERY';break
        if not safe_count:
            state = 'SAFE_STOP'
            recovery=self._clearance_recovery(observation,minimum,res,road,terrain,dynamic)
            if recovery is not None and not self.crossing_guard.waiting and terrain_remaining is None:
                command=recovery;state='CLEARANCE_RECOVERY'
        if state in ('SAFE_STOP', 'IMPASSABLE_HIGH'):
            self.stats['safe_stop_steps'] += 1
        if pull is not None:
            self.stats['yield_steps'] += 1
        self._sync()
        self.last_neural = candidates[chosen].copy()
        return dict(vx=float(command[0]), vy=0., yaw_rate=float(command[1]),
            neural_trajectory=candidates[chosen], diagnostic=dict(mode=state, moving_tracks=len(dynamic),
            candidates=k, selected=chosen, safe_candidates_found=safe_count,
            pullover=pull.tolist() if pull is not None else None,
            support_z=self.mapper.support_z, max_event_retreat_m=self.stats['max_event_retreat_m'],
            measured_edge_distance_m=measured_edge,
            crossing_wait=self.crossing_guard.waiting,crossing_distance_m=self.crossing_guard.distance,
            crossing_return_budget_s=self.crossing_guard.last_budget,
            astar_used=False, latency_ms=(time.perf_counter()-start)*1000))
