"""Common calibrated D435i road projection for the hardware-sensor ablations."""
import numpy as np
from scipy.ndimage import map_coordinates
from navigation.sensor_planner import RoadMemory


class CalibratedRoadMemory(RoadMemory):
    latest_observation = None

    def update(self, rgb, camera_pose, current_pose, lidar_xyz=None, lidar_mask=None):
        observation = self.latest_observation
        if observation is None or 'T_odom_depth_optical' not in observation:
            return super().update(rgb, camera_pose, current_pose, lidar_xyz, lidar_mask)
        if self.origin is None:
            # Same declared initially-supported, paved start as the frozen
            # suite. No initialization reads a simulator road polygon.
            self.origin = np.asarray(current_pose[:2])-self.n*self.res/2
            center = self.index(current_pose[:2])
            xx, yy = np.meshgrid(np.arange(center[0]-30, center[0]+31),
                                 np.arange(center[1]-30, center[1]+31), indexing='ij')
            ij = np.c_[xx.ravel(), yy.ravel()]
            from navigation.sensor_planner import rotation
            local = (self.xy(ij)-current_pose[:2])@rotation(current_pose[2])
            valid = (local[:, 0] >= -.6) & (local[:, 0] <= 1.45) & (np.abs(local[:, 1]) <= .42)
            self.score[ij[valid, 0], ij[valid, 1]] = 8
        image = np.asarray(observation['depth_rgb'])
        if float(image.mean()) < 4.:
            return
        transform = np.asarray(observation['T_odom_depth_optical'])
        intrinsic = np.asarray(observation['depth_intrinsics'])
        depth = np.asarray(observation['depth_m'])
        valid_depth = np.asarray(observation['depth_valid'])
        center = self.index(transform[:2, 3])
        radius = int(2.8/self.res)
        xx, yy = np.meshgrid(np.arange(max(0, center[0]-radius), min(self.n, center[0]+radius+1)),
                             np.arange(max(0, center[1]-radius), min(self.n, center[1]+radius+1)), indexing='ij')
        ij = np.c_[xx.ravel(), yy.ravel()]
        world = np.c_[self.xy(ij), np.full(len(ij), float(observation.get('support_odom_z', 0.)))]
        optical = (world-transform[:3, 3])@transform[:3, :3]
        z = optical[:, 2]
        u = intrinsic[0, 0]*optical[:, 0]/np.maximum(z, .001)+intrinsic[0, 2]
        v = intrinsic[1, 1]*optical[:, 1]/np.maximum(z, .001)+intrinsic[1, 2]
        valid = (z >= .3) & (z <= 2.5) & (u >= 1) & (u < depth.shape[1]-2) & (v >= 1) & (v < depth.shape[0]-2)
        sampled = map_coordinates(depth, [v, u], order=0, mode='nearest')
        seen = map_coordinates(valid_depth.astype(np.uint8), [v, u], order=0, mode='nearest') > 0
        # Only a measured ground-consistent ray may update a planar road cell.
        # Foreground clothing/vehicles and missing returns cannot paint paving.
        valid &= seen & (np.abs(sampled-z) < .06)
        probability = self.head(image)
        values = map_coordinates(probability, [v[valid], u[valid]], order=1, mode='nearest')
        indices = ij[valid]
        increment = np.rint((values-.5)*8).astype(np.int16)
        self.score[indices[:, 0], indices[:, 1]] = np.clip(self.score[indices[:, 0], indices[:, 1]]+increment, -24, 24)
