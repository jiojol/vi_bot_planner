"""Small batched control-space refinement of decoded neural trajectories.

An XY-only gradient on a direct Flow output does not change independent sin/cos
heading channels. Refining through unicycle dynamics couples position costs to
the yaw command that is actually executed. This is a candidate optimizer, not
grid search or A*, and every initial candidate is a frozen-network sample.
"""
import torch
import numpy as np


def track_trajectory(trajectory):
    """Same frozen prefix tracker arithmetic, without simulator dependencies."""
    q = np.asarray(trajectory)
    point = q[1, :2]
    speed = np.clip(np.linalg.norm(point)/.4, 0., .5)
    angle = np.arctan2(point[1], point[0])
    heading = np.arctan2(q[1, 2], q[1, 3])
    half = abs(heading)/2
    factor = half/np.sin(half) if half > 1e-6 else 1.
    return (float(np.clip(speed*factor, 0, .5)) if abs(angle) <= .7 else 0.,
            float(np.clip(heading/.4, -1, 1)))


def control_rollout(controls, measured_twist=(0., 0.), dt=.2, vmax=.5, wmax=1., amax=1., alphamax=2.):
    batch = len(controls)
    position = controls.new_zeros(batch, 2)
    heading = controls.new_zeros(batch)
    velocity = controls.new_full((batch,), float(measured_twist[0]))
    omega = controls.new_full((batch,), float(measured_twist[1]))
    states = []
    for index in range(controls.shape[1]):
        desired_v = controls[:, index, 0].clamp(0., vmax)
        desired_w = controls[:, index, 1].clamp(-wmax, wmax)
        previous_v, previous_w = velocity, omega
        velocity = velocity+(desired_v-velocity).clamp(-amax*dt, amax*dt)
        omega = omega+(desired_w-omega).clamp(-alphamax*dt, alphamax*dt)
        # Midpoint velocities approximate the acceleration-limited substeps;
        # the hard checker separately integrates at finer resolution.
        average_v = (velocity+previous_v)/2
        average_w = (omega+previous_w)/2
        mid = heading+average_w*dt/2
        position = position+average_v[:, None]*dt*torch.stack([mid.cos(), mid.sin()], -1)
        heading = heading+average_w*dt
        states.append(torch.cat([position, torch.stack([heading.sin(), heading.cos()], -1)], -1))
    return torch.stack(states, 1)


def refine_kinematic(trajectories, cost, measured_twist=(0., 0.), iterations=6, step_size=.20):
    xy = trajectories[..., :2]
    segments = torch.diff(torch.cat([xy.new_zeros(len(xy), 1, 2), xy], 1), dim=1)
    speed = (torch.linalg.vector_norm(segments, dim=-1)/.2).clamp(0., .5)
    heading = torch.atan2(trajectories[..., 2], trajectories[..., 3])
    turn = torch.diff(torch.cat([heading.new_zeros(len(heading), 1), heading], 1), dim=1)
    turn = torch.atan2(turn.sin(), turn.cos())/.2
    reference = torch.stack([speed, turn.clamp(-1., 1.)], -1).detach()
    controls = reference.clone()
    for _ in range(iterations):
        with torch.enable_grad():
            controls = controls.detach().requires_grad_(True)
            path = control_rollout(controls, measured_twist)
            prior = .1*(controls-reference).square().mean((1, 2))
            smooth = .05*(controls[:, 1:]-controls[:, :-1]).square().mean((1, 2))
            gradient = torch.autograd.grad((cost(path)+prior+smooth).sum(), controls)[0]
        controls = (controls-step_size*torch.nan_to_num(gradient).clamp(-1., 1.)).detach()
        controls[..., 0].clamp_(0., .5)
        controls[..., 1].clamp_(-1., 1.)
    with torch.no_grad():
        path = control_rollout(controls, measured_twist)
        scores = cost(path)
    return path, scores, controls
