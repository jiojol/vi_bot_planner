"""Five measured terrain channels and explicit observation support.

The grid is a navigation observation, not a contact/foothold feasibility proof.
All quantities are relative to the current estimated support plane. Missing
returns alone never become a cliff label; unknown space remains a hard stop.
"""
from collections import deque
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, distance_transform_edt, median_filter
from navigation.observation_geometry import sensor_points_odom, depth_points_odom, odom_to_local
from navigation.ground_head import GroundHead


@dataclass
class TerrainGrid:
    channels: np.ndarray                 # [5,y,x]: mean, step, variance, drop, grass
    observed: np.ndarray
    count: np.ndarray
    minimum_xy: np.ndarray
    resolution: float
    support_z: float
    high: np.ndarray
    stair: np.ndarray
    timestamp: float

    def index(self, xy):
        return np.floor((np.asarray(xy)-self.minimum_xy)/self.resolution).astype(int)

    def lookup(self, array, xy, unknown=1.):
        ij = self.index(xy)
        inside = (ij >= 0).all(-1) & (ij < self.observed.shape[0]).all(-1)
        result = np.full(ij.shape[:-1], unknown, dtype=float)
        result[inside] = array[ij[inside, 1], ij[inside, 0]]
        return result

    def terrain_cost(self, allow_stairs=False):
        risk = self.high | (self.channels[3] > .5)
        if not allow_stairs:
            risk |= self.channels[1] >= .04
        return np.maximum(risk.astype(np.float32), (~self.observed).astype(np.float32)*.65)


class ElevationMapper:
    def __init__(self, config, ground_head=None):
        self.config = config
        self.settings = config['grid']
        self.history = deque(maxlen=15)
        self.depth_history = deque(maxlen=15)
        self.last_stamp = -np.inf
        self.support_z = 0.
        self.head = GroundHead(ground_head) if ground_head else None
        self.last_grid = None
        self.forward_hazard_kind = None
        self.dynamic_footprints = []

    def reset(self):
        self.history.clear()
        self.depth_history.clear()
        self.last_stamp = -np.inf
        self.support_z = 0.
        self.last_grid = None

    def update(self, observation, dynamic_tracks=None):
        stamp = float(observation.get('sensor_time', observation['scan_time']))
        now = float(observation['time'])
        pose = np.asarray(observation['odometry'])
        depth, pixels = depth_points_odom(observation)
        lidar = sensor_points_odom(observation)
        points = np.vstack([lidar, depth])
        material = np.full(len(points), np.nan)
        if len(pixels) and self.head is not None:
            probability = self.head(observation['depth_rgb'])
            material[len(lidar):] = 1.-probability[pixels[:, 0], pixels[:, 1]]
        if stamp > self.last_stamp and now-stamp <= .35:
            finite = np.isfinite(points).all(-1)
            self.history.append((stamp, points[finite], material[finite]))
            self.depth_history.append((stamp, depth.copy()))
            self.last_stamp = stamp
        clouds = [q for t, q, _ in self.history if now-t <= 1.5]
        semantics = [s for t, _, s in self.history if now-t <= 1.5]
        points = np.concatenate(clouds) if clouds else np.empty((0, 3))
        material = np.concatenate(semantics) if semantics else np.empty(0)
        self.dynamic_footprints = []
        if dynamic_tracks and len(points):
            keep = np.ones(len(points), bool)
            for track in dynamic_tracks:
                if not track.get('is_dynamic'):
                    continue
                history = [p for t, p in track['history'] if now-t <= 1.5]
                centers = np.asarray(history or [track['center']])
                from scipy.spatial import cKDTree
                distance = cKDTree(centers).query(points[:, :2])[0]
                radius = max(.70, np.linalg.norm(track['extent'])/2+.18)
                self.dynamic_footprints.append((centers, radius))
                keep &= ~((distance < radius) & (points[:, 2] > self.support_z+.06))
            points, material = points[keep], material[keep]
        # Support estimator uses observed near-body surfaces, never world height
        # labels. The initial support plane is the documented odom origin.
        if self.config['control']['allow_stair_proxy'] and len(points):
            local_xy = odom_to_local(points, pose)[:, :2]
            under = (local_xy[:, 0] >= .10) & (local_xy[:, 0] <= .65) & (np.abs(local_xy[:, 1]) < .17)
            under &= np.abs(points[:, 2]-self.support_z) <= .16
            if under.sum() >= 6:
                target = float(np.percentile(points[under, 2], 75))
                self.support_z += np.clip(target-self.support_z, -.04, .04)
        local = odom_to_local(points, pose, self.support_z)
        n, res = self.settings['cells'], self.settings['resolution_m']
        minimum = np.asarray(self.settings['minimum_xy_m'])
        ij = np.floor((local[:, :2]-minimum)/res).astype(int)
        valid = (ij >= 0).all(-1) & (ij < n).all(-1) & (local[:, 2] < 2.5) & (local[:, 2] > -5.)
        ij, z, material = ij[valid], local[valid, 2], material[valid]
        index = ij[:, 1]*n+ij[:, 0]
        count = np.bincount(index, minlength=n*n).reshape(n, n)
        total = np.bincount(index, weights=z, minlength=n*n).reshape(n, n)
        total2 = np.bincount(index, weights=z*z, minlength=n*n).reshape(n, n)
        mean = total/np.maximum(count, 1)
        variance = np.maximum(0., total2/np.maximum(count, 1)-mean**2)
        observed = count >= self.settings['min_hits']
        # Smooth only measured neighbors. Unknown zero padding must never create
        # a fictitious step at a sensor's visibility boundary.
        low = minimum_filter(np.where(observed, mean, np.inf), size=3, mode='constant', cval=np.inf)
        high = maximum_filter(np.where(observed, mean, -np.inf), size=3, mode='constant', cval=-np.inf)
        step = np.where(observed, np.maximum(0., high-low), 0.)
        drop = observed & (mean < -.12)
        # Mark the upper lip as well when a lower surface is actually measured.
        drop |= observed & (low < mean-.12) & (mean <= .04)
        high_mask = observed & (step > .16)
        # A single bounded step is not evidence of a whole climbable flight.
        stair = observed & (step >= .04) & (step <= .15) & ~drop & (variance < .003)
        grass_sum = np.bincount(index[np.isfinite(material)], weights=material[np.isfinite(material)], minlength=n*n).reshape(n, n)
        grass_count = np.bincount(index[np.isfinite(material)], minlength=n*n).reshape(n, n)
        grass = grass_sum/np.maximum(grass_count, 1)
        channels = np.stack([mean, step, variance, drop.astype(float), grass]).astype(np.float32)
        self.last_grid = TerrainGrid(channels, observed, count, minimum, res,
                                     self.support_z, high_mask, stair, stamp)
        return self.last_grid

    def forward_hazard_distance(self, observation, half_width=.20):
        """Sub-cell edge estimate from dense measured stereo support samples.

        Coarse 10cm cell dilation suffices for costs but cannot place a stop in
        a 10cm-wide acceptance band. Use 2.5cm depth-only bands here; noisy
        sparse laser returns cannot impersonate a low curb. A lower return or
        a short, observed support frontier establishes a conservative stop.
        """
        self.forward_hazard_kind = None
        now = float(observation['time'])
        clouds = [q for t, q in self.depth_history if now-t <= .5]
        if not clouds:
            return None
        depth_cloud = np.concatenate(clouds)
        for centers, radius in self.dynamic_footprints:
            from scipy.spatial import cKDTree
            distance = cKDTree(centers).query(depth_cloud[:, :2])[0]
            depth_cloud = depth_cloud[~((distance < radius) & (depth_cloud[:, 2] > self.support_z+.06))]
        local = odom_to_local(depth_cloud, np.asarray(observation['odometry']), self.support_z)
        # A visible foreground object censoring ground is not a cliff frontier.
        fresh_depth, _ = depth_points_odom(observation, stride=2)
        foreground = odom_to_local(fresh_depth, np.asarray(observation['odometry']), self.support_z)
        foreground = foreground[(foreground[:, 2] > .16) & (np.abs(foreground[:, 1]) < half_width+.10)]
        keep = (local[:, 0] > .35) & (local[:, 0] < 2.5) & (np.abs(local[:, 1]) < half_width)
        q = local[keep]
        if len(q) < 30:
            return None
        boundaries = []
        # Evaluate separate lateral strips. A median across the whole width
        # hides the nearest corner of an oblique edge and stops a body too late.
        lateral = np.floor((q[:, 1]+half_width)/.05).astype(int)
        for strip in np.unique(lateral):
            patch = q[lateral == strip]
            bands = np.floor(patch[:, 0]/.025).astype(int)
            heights, centers = [], []
            for band in np.unique(bands):
                values = patch[bands == band, 2]
                if len(values) >= 3:
                    centers.append((band+.5)*.025)
                    heights.append(float(np.median(values)))
            x, h = np.asarray(centers), np.asarray(heights)
            if not len(x):
                continue
            changed = (h > .035) | (h < -.10)
            if changed.any():
                first = int(np.flatnonzero(changed)[0])
                if h[first] < -.10:
                    flat = x[(np.abs(h) < .025) & (x < x[first])]
                    edge = float(flat.max()+.0125) if len(flat) else float(x[first]-.0125)
                    boundaries.append((edge, 'drop'))
                else:
                    boundaries.append((float(x[first]-.0125), 'low_step' if h[first] <= .16 else 'high'))
            else:
                flat = x[np.abs(h) < .025]
                blocked = len(flat) and np.any((foreground[:, 0] > flat.max()-.25) & (foreground[:, 0] < flat.max()+.6))
                if len(flat) >= 10 and flat.max() < 1.35 and not blocked and np.asarray(observation.get('depth_valid', [])).sum() > 1000:
                    boundaries.append((float(flat.max()+.0125), 'unknown_frontier'))
        if len(boundaries) < 2:
            return None
        distance, self.forward_hazard_kind = min(boundaries, key=lambda value:value[0])
        return distance

    def decision(self, grid, path_xy, body_length=.6):
        """Look only along a proposed corridor, allowing real flat bypasses."""
        xy = np.asarray(path_xy)
        observed = grid.lookup(grid.observed, xy, 0.) > .5
        drop = grid.lookup(grid.channels[3], xy, 0.) > .5
        step = grid.lookup(grid.channels[1], xy, 0.)
        high = grid.lookup(grid.high, xy, 0.) > .5
        if np.any(drop):
            return 'SAFE_STOP'
        if np.any(high):
            return 'IMPASSABLE_HIGH'
        if np.any(step >= .04):
            bounded = np.all(step <= .15)
            # Verify multiple distinct successive risers, not three samples of
            # the same edge. A single curb cannot establish a continuous flight.
            forward = np.arange(.55, 2.01, .05)
            ahead = np.c_[forward, np.zeros_like(forward)]
            valid = grid.lookup(grid.observed, ahead, 0.) > .5
            heights = grid.lookup(grid.channels[0], ahead, 0.)[valid]
            levels = []
            for height in heights:
                if not levels or abs(height-levels[-1]) > .03:
                    levels.append(height)
            rises = np.diff(levels)
            continuous = np.count_nonzero((rises >= .04) & (rises <= .15)) >= 2 and np.all(rises >= -.025)
            return 'STAIR_CLIMB' if bounded and continuous else 'SAFE_STOP'
        return 'NORMAL_TROT' if observed.all() else 'SAFE_STOP'


def save_finetune_sample(path, grid, observation, goal, trajectory, label_source='unlabelled_observation'):
    """Unlabelled measured terrain samples; evaluator labels never enter runtime.

    A future training job must explicitly supply new training-only targets.
    Existing held-out trajectories must not be silently reused for fine-tuning.
    """
    np.savez_compressed(path, terrain=grid.channels, observed=grid.observed,
        odometry=observation['odometry'], timestamp=observation['time'], goal=goal,
        trajectory=trajectory, label_source=label_source, support_z=grid.support_z)
