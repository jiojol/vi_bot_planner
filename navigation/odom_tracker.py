"""Gravity-compensated, fixed-odom CV Kalman tracks with a dual motion gate."""
from collections import deque
import numpy as np
from scipy.spatial import cKDTree
from navigation.observation_geometry import sensor_points_odom, odom_to_local
from navigation.sensor_planner import ReturnTracker
from navigation.surface_motion import SurfaceMotionMemory,fresh_motion_confirmation,compact_surface_observable


def extended_surface_mask(points, sensor_z=None, ray_origin=None):
    """Robust long-plane segmentation, independent of point ordering.

    Dynamic association is defined for compact objects (<1.8m). Extended
    measured planes (>2m) belong to the static occupancy layer. Their visible
    patch centroids are not object positions. This is a sensor/model assumption,
    not a test-scene semantic label or a static-world oracle.
    """
    static = np.zeros(len(points), bool)
    remaining = np.arange(len(points))
    rng = np.random.default_rng(5821)
    for _ in range(12):
        if len(remaining) < 12:
            break
        q = points[remaining]
        triples = q[rng.integers(0, len(q), (96, 3))]
        normals = np.cross(triples[:, 1]-triples[:, 0], triples[:, 2]-triples[:, 0])
        norms = np.linalg.norm(normals, axis=1)
        good = norms > .01
        if not good.any():
            break
        normals = normals[good]/norms[good, None]
        centers = triples[good, 0]
        residual = np.abs(q@normals.T-np.sum(centers*normals, axis=1))
        inliers = residual < .045
        best = np.argmax(inliers.sum(0))
        mask = inliers[:, best]
        if mask.sum() < 10:
            break
        patch = q[mask]
        _, singular, _ = np.linalg.svd(patch-patch.mean(0), full_matrices=False)
        extensive = np.linalg.norm(np.ptp(patch, axis=0)) > 2.0
        # Two-dimensional support is required; a sparsely sampled line cannot
        # establish a ground/wall plane orientation.
        # A horizontal LiDAR ring hits unrelated objects at the laser height.
        # Their co-planarity is a sampling artifact, not a physical tabletop.
        ring_alias = sensor_z is not None and abs(normals[best, 2]) > .9 and abs(np.median(patch[:, 2])-sensor_z) < .04
        # Nonzero elevation rings lie on a cone: a small arc can fit a sloped
        # plane through the laser origin. This is also a sampling degeneracy.
        if ray_origin is not None:
            ring_alias |= abs(normals[best]@(np.asarray(ray_origin)-centers[best])) < .10
        if extensive and singular[1]/np.sqrt(len(patch)) > .04 and not ring_alias:
            static[remaining[mask]] = True
        remaining = remaining[~mask]
    return static


def extended_vertical_mask(points):
    """Long connected XY wall support is not a compact moving object.

    A single elevation ring can make a physical vertical surface rank deficient
    for 3-D PCA. Require a connected >2m XY segment; keep these points in static
    occupancy. As elsewhere, this assumes targets are compact (<1.8m).
    """
    mask=np.zeros(len(points),bool)
    if len(points)<12:return mask
    remaining=np.arange(len(points));rng=np.random.default_rng(6339)
    for _ in range(10):
        if len(remaining)<12:break
        q=points[remaining,:2];pairs=q[rng.integers(0,len(q),(64,2))]
        tangent=pairs[:,1]-pairs[:,0];norm=np.linalg.norm(tangent,axis=1);valid=norm>.6
        if not np.any(valid):break
        tangent=tangent[valid]/norm[valid,None];origins=pairs[valid,0]
        normals=np.c_[-tangent[:,1],tangent[:,0]]
        residual=np.abs(q@normals.T-np.sum(origins*normals,axis=1))
        inliers=residual<.035;best=int(np.argmax(inliers.sum(0)));selected=np.flatnonzero(inliers[:,best])
        if len(selected)<12:break
        projection=q[selected]@tangent[best];order=np.argsort(projection);selected=selected[order];projection=projection[order]
        for group in np.split(np.arange(len(selected)),np.flatnonzero(np.diff(projection)>.55)+1):
            if len(group)>=12 and projection[group[-1]]-projection[group[0]]>2.:
                mask[remaining[selected[group]]]=True
        remaining=remaining[~inliers[:,best]]
    return mask


class OdomTracker:
    def __init__(self, config):
        self.config = config['tracking']
        self.range_limit=config.get('simulation_sensors',{}).get('mid360',{}).get('range_m',8.)
        self.clusterer = ReturnTracker()
        self.filters = {}
        self.tracks = []
        self.points = np.empty((0, 2))
        self.last_stamp = -np.inf
        self.raw_history=deque(maxlen=50)
        self.aliases={}
        self.surface_memory=SurfaceMotionMemory()

    def _track_age(self,track):
        return 1.0 if track.get('is_dynamic') else self.config['max_gap_s']

    def update(self, observation):
        stamp = float(observation.get('sensor_time', observation['scan_time']))
        now = float(observation['time'])
        pose = np.asarray(observation['odometry'])
        if stamp <= self.last_stamp or now-stamp > self.config['max_gap_s']:
            self.tracks = [t for t in self.tracks if now-t['stamp'] <= self._track_age(t)]
            # Repeated delayed packets are not contradictory motion evidence.
            # Retain the last dual-gate result only within bounded track age.
            return
        self.last_stamp = stamp
        world = sensor_points_odom(observation)
        support = float(observation.get('support_odom_z', 0.))
        # Elevated treads/tabletops are horizontal support-like surfaces, not
        # moving vertical objects. Segment them after gravity compensation.
        # A changing visible patch on a long plane can translate its centroid
        # by metres, far beyond any centroid-only displacement gate.
        relevant = world[(world[:, 2]-support > .10) & (world[:, 2]-support < 1.3)]
        all_relevant = relevant.copy()
        older=[q for t,q in self.raw_history if 2.0<=stamp-t<=3.0]
        static_tree=cKDTree(np.concatenate(older)) if older else None
        measurement_tree=cKDTree(all_relevant[:,:2]) if len(all_relevant) else None
        self.raw_history.append((stamp,world.copy()))
        relevant = relevant[~extended_vertical_mask(relevant)]
        sensor_z = float(np.asarray(observation['T_odom_lidar'])[2, 3]) if 'T_odom_lidar' in observation else support+.65
        ray_origin = np.asarray(observation['T_odom_lidar'])[:3, 3] if 'T_odom_lidar' in observation else np.r_[observation.get('camera_pose', pose)[:2], sensor_z]
        self.surface_memory.add(stamp,world,ray_origin)
        relevant = relevant[~extended_surface_mask(relevant, sensor_z, ray_origin)]
        if len(relevant) >= 8:
            distances, neighbors = cKDTree(relevant).query(relevant, k=8)
            patches = relevant[neighbors]
            centered = patches-patches.mean(1, keepdims=True)
            covariance = np.einsum('nki,nkj->nij', centered, centered)/8
            values, vectors = np.linalg.eigh(covariance)
            horizontal = (np.abs(vectors[:, 2, 0]) > .90) & (values[:, 0] < .0006)
            horizontal &= (values[:, 1] > .0003) & (distances[:, -1] < .65)
            world = relevant[~horizontal]
        else:
            world = relevant
        local = odom_to_local(world, pose, support)
        self.clusterer.update(local, np.ones(len(local), bool), pose, stamp, now)
        # Segmented static planes remain obstacles; filtering must never erase
        # them from collision fields merely to avoid dynamic false positives.
        if len(all_relevant):
            self.points = all_relevant[:, :2]
        updated = []
        used_filters=set()
        settings = self.config
        for measurement in self.clusterer.tracks:
            if measurement['stamp'] != stamp:
                continue
            raw_key = measurement['id']
            key = self.aliases.get(raw_key,raw_key)
            center = measurement['center']
            state = self.filters.get(key)
            continuation=False
            if state is None:
                matches=[]
                for candidate,previous in self.filters.items():
                    dt=stamp-previous['stamp']
                    if candidate in used_filters or not (previous.get('was_dynamic') or previous.get('was_candidate')) or not 0<dt<=.6:continue
                    residual=np.linalg.norm(center-(previous['x'][:2]+previous['x'][2:]*dt))
                    if residual<.30:matches.append((residual,candidate,previous))
                if matches:
                    _,key,state=min(matches,key=lambda q:q[0]);self.aliases[raw_key]=key;continuation=True
            if key in used_filters:continue
            used_filters.add(key)
            if state is not None and (state.get('was_dynamic') or state.get('was_candidate')) and stamp-state['stamp']<=.35:
                prediction=state['x'][:2]+state['x'][2:]*(stamp-state['stamp'])
                continuation|=np.linalg.norm(center-prediction)<max(.20,.20+.25*np.linalg.norm(measurement['extent']))
            if state is None or stamp-state['stamp'] > settings['max_gap_s']:
                state = dict(x=np.r_[center, 0., 0.], covariance=np.diag([.02, .02, 1., 1.]),
                             stamp=stamp, first=center.copy(), history=deque(maxlen=50),
                             velocities=deque(maxlen=settings['min_frames']), confirmed=False)
                self.filters[key] = state
            else:
                dt = stamp-state['stamp']
                if dt > .21:
                    state['velocities'].clear()
                f = np.eye(4)
                f[:2, 2:] = np.eye(2)*dt
                g = np.vstack([np.eye(2)*dt**2/2, np.eye(2)*dt])
                covariance = f@state['covariance']@f.T + g@g.T*.6**2
                prediction = f@state['x']
                residual = center-prediction[:2]
                # Position covariance also represents changing visible surfaces.
                noise = np.eye(2)*.035**2
                gain = np.linalg.solve(covariance[:2, :2]+noise, covariance[:2]).T
                state['x'] = prediction+gain@residual
                h = np.c_[np.eye(2), np.zeros((2, 2))]
                a = np.eye(4)-gain@h
                state['covariance'] = a@covariance@a.T+gain@noise@gain.T
                state['stamp'] = stamp
            state['history'].append((stamp, state['x'][:2].copy()))
            velocity = state['x'][2:].copy()
            # Stable observed motion axis rejects orthogonal visible-surface
            # centroid jitter without relaxing either dynamic-confirmation gate.
            history = np.array([p for _, p in state['history']])
            if len(history) >= 6:
                _, singular, axes = np.linalg.svd(history-history.mean(0), full_matrices=False)
                if singular[0] > .25 and singular[0] > 3*singular[1]:
                    velocity = axes[0]*(velocity@axes[0])
            state['velocities'].append(velocity)
            speed = float(np.linalg.norm(velocity))
            vectors = np.asarray(state['velocities'])
            speeds = np.linalg.norm(vectors, axis=1)
            cosine = (vectors[1:]*vectors[:-1]).sum(-1)/np.maximum(1e-9, speeds[1:]*speeds[:-1])
            displacement = float(np.linalg.norm(state['x'][:2]-state['first']))
            direction_gate = len(vectors) >= settings['min_frames'] and np.all(cosine > settings['direction_cosine'])
            envelope_gate = displacement > settings['min_net_m'] and settings['min_speed_mps'] <= speed <= settings['max_speed_mps']
            # An apparent centroid can drift while a static stair/wall surface
            # splits. Require independent observed surface translation too.
            # A newly dynamic compact object needs independent measured ray
            # evidence. Its own overlapping recent side returns are not proof
            # that the whole object is a stationary wall.
            surface_gate=measurement.get('moving',False) or continuation
            surface_gate &= np.max(measurement['extent'])<1.8
            evidence=dict(confirmed=False,count=0,fraction=0.,views=0)
            spatial_support=False
            if measurement_tree is not None:
                _,ids=measurement_tree.query(measurement['points'])
                evidence=self.surface_memory.support(all_relevant[ids],stamp)
                if not state.get('confirmed',False):
                    groups=measurement_tree.query_ball_point(measurement['points'],.04)
                    full_ids=np.unique(np.concatenate(groups)).astype(int)
                    spatial_support=compact_surface_observable(all_relevant[full_ids])
            base_motion=bool(direction_gate and envelope_gate and surface_gate)
            fresh=fresh_motion_confirmation(state,stamp,base_motion and evidence['confirmed'] and spatial_support,settings['max_gap_s'])
            candidate=base_motion and (state.get('confirmed',False) or fresh)
            state['was_candidate']=base_motion
            dynamic=candidate
            if dynamic:state['last_dynamic_stamp']=stamp
            elif state.get('was_dynamic') and stamp-state.get('last_dynamic_stamp',-100.)<.6:dynamic=True
            static_fraction=0.;range_truncated=False
            state['confirmed'] |= dynamic
            state['was_dynamic']=dynamic
            updated.append({**measurement, 'id':key,'measurement_id':raw_key,'center':state['x'][:2].copy(), 'velocity':velocity,
                'history':state['history'], 'is_dynamic':dynamic, 'moving':dynamic,
                'counterflow_confirmed':dynamic, 'ever_confirmed':state['confirmed'],
                'direction_gate':bool(direction_gate), 'displacement_gate':bool(envelope_gate),
                'net_displacement_m':displacement,'static_surface_fraction':static_fraction,
                'range_truncated':range_truncated,'free_space_support':evidence})
        seen_ids = {track['id'] for track in updated}
        updated.extend(track for track in self.tracks if track['id'] not in seen_ids and
                       now-track['stamp'] <= self._track_age(track))
        self.tracks = updated
        active = {t['id'] for t in updated}
        self.filters = {k: v for k, v in self.filters.items() if k in active or stamp-v['stamp'] < 1.}
        self.aliases={raw:key for raw,key in self.aliases.items() if key in self.filters}
