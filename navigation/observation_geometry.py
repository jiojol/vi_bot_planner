"""Measured transforms only. No scene, simulator or future actor inputs."""
import numpy as np
from navigation.sensor_planner import rotation


def attitude_matrix(roll, pitch, yaw=0.):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.]]) @ np.array(
        [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array(
        [[1, 0, 0], [0, cr, -sr], [0, sr, cr]])


def transform_xyz(points, transform):
    q = np.asarray(points, dtype=float)
    t = np.asarray(transform, dtype=float)
    if q.shape[-1] != 3 or t.shape[-2:] != (4, 4):
        raise ValueError('Expected xyz and T_target_source[...,4,4]')
    return np.einsum('...ij,...j->...i', t[..., :3, :3], q) + t[..., :3, 3]


def sensor_points_odom(observation):
    """Full scan-time SE(3), including roll/pitch, BEFORE ground segmentation.

    Legacy scans have already been deskewed to the current gravity frame by
    their documented adapter. They must not be rotated by IMU a second time.
    """
    if 'lidar_sensor_xyz' in observation:
        q = np.asarray(observation['lidar_sensor_xyz'])
        valid = np.asarray(observation['lidar_sensor_valid'], bool) & np.isfinite(q).all(-1)
        transform = np.asarray(observation['T_odom_lidar'])
        return transform_xyz(q[valid], transform if transform.ndim == 2 else transform[valid])
    q = np.asarray(observation['lidar_xyz'])[np.asarray(observation['lidar_valid_mask'], bool)]
    pose = np.asarray(observation['odometry'])
    return np.c_[q[:, :2] @ rotation(pose[2]).T + pose[:2], q[:, 2]]


def depth_points_odom(observation, stride=1):
    if 'depth_m' not in observation:
        return np.empty((0, 3)), np.empty((0, 2), int)
    depth = np.asarray(observation['depth_m'])
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.floating):
        raise ValueError('D435i depth must be a floating-point metric z image')
    k = np.asarray(observation['depth_intrinsics'])
    vv, uu = np.mgrid[:depth.shape[0]:stride, :depth.shape[1]:stride]
    z = depth[::stride, ::stride]
    valid = np.isfinite(z) & (z >= .3) & (z <= 2.5)
    valid &= np.asarray(observation.get('depth_valid', np.ones_like(depth, bool)))[::stride, ::stride]
    u, v, z = uu[valid], vv[valid], z[valid]
    optical = np.c_[(u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z]
    return transform_xyz(optical, observation['T_odom_depth_optical']), np.c_[v, u]


def odom_to_local(points, pose, support_z=0.):
    q = np.asarray(points)
    return np.c_[(q[:, :2]-pose[:2]) @ rotation(pose[2]), q[:, 2]-support_z]
