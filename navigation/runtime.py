"""Standalone navigation API: high-level motion, no gait or simulator import."""
import json
from pathlib import Path
import torch
from models.runtime import load_policy
from navigation.sensor_planner import SensorPlanner


class Navigator:
    def __init__(self,config,checkpoint,ground_head,device='cuda',seed=1,mode=None,research=None):
        self.config={**config,'ground_head':str(ground_head)};self.policy=load_policy(checkpoint,device)
        self.device=device;self.seed=seed;self.mode=mode or config.get('revision5_mode','s1_astar_v4');self.research=research;self.reset()

    def reset(self):
        if self.mode!='s1_astar_v4':
            from navigation.revision5 import Revision5Navigator
            self.revision5=Revision5Navigator(self.config,self.policy,self.config['ground_head'],self.research,self.mode,self.device,self.seed)
            self.planner=self.revision5;return
        self.revision5=None
        self.planner=SensorPlanner(self.config);self.generator=torch.Generator(device=self.device).manual_seed(self.seed)

    def command(self,observation,local_goal):
        if self.revision5 is not None:return self.revision5.command(observation,local_goal)
        proposal=self.policy.predict(rgb=observation['rgb'],lidar_xyz=observation['lidar_xyz'],lidar_valid_mask=observation['lidar_valid_mask'],goal_xy=local_goal,generator=self.generator)
        (v,w),diagnostic=self.planner.command(observation,local_goal,proposal)
        return {'vx':v,'vy':0.,'yaw_rate':w,'neural_trajectory':proposal,'diagnostic':diagnostic}
