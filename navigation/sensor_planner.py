"""Sensor/odometry-only local planning. No simulator or map imports.

Development baseline: normalized RGB ground-material classifier + measured
LiDAR; static paving memory and tracked moving returns. The neural trajectory
is a soft proposal, never a substitute for measured clearance. Ideal odometry
in current simulation is explicitly an assumption, not a real-SLAM claim.
"""
import heapq
from collections import deque
import numpy as np
from scipy.ndimage import distance_transform_edt,label,gaussian_filter,map_coordinates,minimum_filter1d
from scipy.spatial import cKDTree
from navigation.ground_head import GroundHead


def wrap(a):return (a+np.pi)%(2*np.pi)-np.pi


def rotation(a):return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])


def predicted_center(track,elapsed):
    """Constant measured velocity, reflecting only at observed past reversals.

    No path endpoints or future actor state are supplied by the simulator.
    Unobserved turns remain unknown; this is prediction, not an oracle.
    """
    point=track['center'].copy();velocity=track['velocity'].copy();remaining=max(0.,elapsed)
    for _ in range(4):
        first=remaining+1.;normal=None
        for n,limit in track.get('turn_planes',[]):
            rate=velocity@n
            if rate<=1e-5:continue
            hit=max(0.,(limit-point@n)/rate)
            if hit<first:first=hit;normal=n
        if normal is None or first>remaining:return point+velocity*remaining
        point+=velocity*first;remaining-=first;velocity-=2*(velocity@normal)*normal
    return point+velocity*remaining


def observed_turn_ahead(track,position):
    """Only trust an upcoming reversal after observing that turning boundary."""
    velocity=track['velocity'];speed=np.linalg.norm(velocity)
    if speed<.2:return False
    direction=velocity/speed
    return any(n@direction>.6 and limit-position@n>=-.15 for n,limit in track.get('turn_planes',[]))


def observed_motion_axis(track):
    points=np.asarray([p for _,p in track.get('history',[])])
    if len(points)<6:return None
    _,singular,vt=np.linalg.svd(points-np.median(points,axis=0),full_matrices=False)
    return vt[0] if singular[0]>.25 and singular[0]>3.*singular[1] else None


def crossing_intersection(track,anchor,axis):
    """Fit the measured spatial corridor; instantaneous velocity is too noisy.

    Velocity still determines whether a return is moving. The crossing plane
    comes from its observed positions, avoiding large extrapolation errors when
    a small lateral velocity error is multiplied by a distant person's offset.
    """
    points=np.asarray([p for _,p in track.get('history',[])])
    perp=np.array([-axis[1],axis[0]])
    tangent=observed_motion_axis(track)
    if tangent is not None and abs(tangent@perp)>.6:
        relative=points-anchor
        intercept=np.median(relative@axis-(relative@perp)*(tangent@axis)/(tangent@perp))
        return anchor+axis*intercept
    return anchor+axis*((track['center']-anchor)@axis)


def remember_crossing_motion(reservation,track):
    """Carry observed reversals across temporary local track identifiers.

    A reversal inferred after occlusion uses the furthest *observed* position,
    never a scene endpoint. Subtract10cm so that an unseen return is predicted
    early rather than assuming the actor continued to a farther hidden point.
    """
    points=reservation.setdefault('motion_points',[])
    planes=reservation.setdefault('turn_planes',[(n.copy(),float(value)-.1) for n,value in track.get('turn_planes',[])])
    if track['stamp']>reservation.get('motion_stamp',-1e9):
        position=track['center'].copy();points.append(position)
        if len(points)>300:points.pop(0)
        reservation['motion_stamp']=track['stamp']
        speed=np.linalg.norm(track['velocity'])
        if speed>.35:
            direction=track['velocity']/speed;previous=reservation.get('last_motion_direction')
            if previous is not None and previous@direction<-.65:
                normal=next((n for n,value in planes if n@previous>.8),previous.copy())
                value=max(p@normal for p in points)-.1
                matched=any(n@normal>.8 for n,old_value in planes)
                planes=[(n,max(old_value,value)) if n@normal>.8 else (n,old_value) for n,old_value in planes]
                if not matched:planes.append((normal,value))
            reservation['last_motion_direction']=direction.copy()
        planes=[(n,max(value,position@n-.1)) for n,value in planes][-4:]
        reservation['turn_planes']=planes
    track['turn_planes']=list(reservation['turn_planes'])


class RoadMemory:
    def __init__(self,camera,resolution=.06,head=None):
        self.camera=camera;self.res=resolution;self.n=1600;self.origin=None
        self.head=GroundHead(head) if head else None
        self.score=np.zeros((self.n,self.n),np.int16)
        x=np.arange(1.02,5.51,resolution);y=np.arange(-5.5,5.51,resolution)
        xx,yy=np.meshgrid(x,y,indexing='ij');self.local=np.c_[xx.ravel(),yy.ravel()]
        f=camera['width']/(2*np.tan(np.deg2rad(camera['hfov_deg'])/2));dx=self.local[:,0]-camera['xyz'][0]
        u=camera['width']/2-f*self.local[:,1]/dx;v=camera['height']/2+f*(camera['xyz'][2]-.002)/dx
        valid=(u>=1)&(u<camera['width']-2)&(v>camera['height']/2+2)&(v<camera['height']-2)
        self.local=self.local[valid];self.u=u[valid].astype(np.float32);self.v=v[valid].astype(np.float32)

    def index(self,xy):return np.rint((np.asarray(xy)-self.origin)/self.res).astype(int)
    def xy(self,ij):return np.asarray(ij)*self.res+self.origin

    def update(self,rgb,camera_pose,current_pose,lidar_xyz=None,lidar_mask=None):
        if self.origin is None:
            self.origin=np.asarray(current_pose[:2])-self.n*self.res/2
            # Camera blind-zone initialization: extrapolate the initial paved
            # strip under the valid starting body; never read a road polygon.
            center=self.index(current_pose[:2]);xx,yy=np.meshgrid(np.arange(center[0]-30,center[0]+31),np.arange(center[1]-30,center[1]+31),indexing='ij')
            ij=np.c_[xx.ravel(),yy.ravel()];local=(self.xy(ij)-current_pose[:2])@rotation(current_pose[2])
            inside=(local[:,0]>=-.6)&(local[:,0]<=1.45)&(np.abs(local[:,1])<=.42);ij=ij[inside];self.score[ij[:,0],ij[:,1]]=8
        # A blank/black camera is missing information, never evidence that the
        # entire projected ground is traversable. Preserve the measured map.
        if float(np.mean(rgb))<4. or float(np.percentile(rgb,95)-np.percentile(rgb,5))<4.:return
        image=gaussian_filter(rgb.astype(np.float32),sigma=(1.,1.,0.))
        # A deliberately explicit synthetic-ground baseline, not claimed as a
        # general pretrained semantic model. Replaced by a fitted head later.
        norm=image/(image.sum(2,keepdims=True)+1.)
        green=norm[:,:,1]-np.maximum(norm[:,:,0],norm[:,:,2])
        road=(1/(1+np.exp(np.clip((green-.035)*140,-40,40)))).astype(np.float32) if self.head is None else self.head(rgb)
        # Inverse warp each world-grid cell into the image. Forward splatting
        # rotated grids leaves alias holes that falsely erase narrow roads.
        center=self.index(camera_pose[:2]);r=int(6./self.res)
        xx,yy=np.meshgrid(np.arange(max(0,center[0]-r),min(self.n,center[0]+r+1)),np.arange(max(0,center[1]-r),min(self.n,center[1]+r+1)),indexing='ij')
        ij=np.c_[xx.ravel(),yy.ravel()];local=(self.xy(ij)-camera_pose[:2])@rotation(camera_pose[2]);dx=local[:,0]-self.camera['xyz'][0]
        f=self.camera['width']/(2*np.tan(np.deg2rad(self.camera['hfov_deg'])/2));safe_dx=np.maximum(.01,dx)
        u=self.camera['width']/2-f*local[:,1]/safe_dx;v=self.camera['height']/2+f*(self.camera['xyz'][2]-.002)/safe_dx
        valid=(local[:,0]>=1.02)&(local[:,0]<=5.5)&(u>=1)&(u<self.camera['width']-2)&(v>self.camera['height']/2+2)&(v<self.camera['height']-2)
        if lidar_xyz is not None:
            q=np.asarray(lidar_xyz)[np.asarray(lidar_mask,dtype=bool)]
            q=q[np.isfinite(q).all(1)&(q[:,2]>.12)&(q[:,2]<2.)]
            if len(q):
                # Re-express compensated LiDAR in the timestamped camera ego
                # frame. Cells behind foreground returns are unobserved ground,
                # not road/grass labels inferred from the occluding clothing.
                world=q[:,:2]@rotation(current_pose[2]).T+current_pose[:2]
                camera_q=(world-camera_pose[:2])@rotation(camera_pose[2]);camera_q[:,0]-=self.camera['xyz'][0]
                angles=np.arctan2(camera_q[:,1],camera_q[:,0]);bins=np.clip(((angles+np.pi)/(2*np.pi)*1024).astype(int),0,1023)
                nearest=np.full(1024,np.inf);np.minimum.at(nearest,bins,np.linalg.norm(camera_q,axis=1));nearest=minimum_filter1d(nearest,9,mode='wrap')
                cell_angles=np.arctan2(local[:,1],dx);cell_bins=np.clip(((cell_angles+np.pi)/(2*np.pi)*1024).astype(int),0,1023)
                valid&=np.hypot(dx,local[:,1])<nearest[cell_bins]-.10
        probs=map_coordinates(road,[v[valid],u[valid]],order=1,mode='nearest');ij=ij[valid];values=np.rint((probs-.5)*8).astype(np.int16)
        self.score[ij[:,0],ij[:,1]]=np.clip(self.score[ij[:,0],ij[:,1]]+values,-24,24)

    def crop(self,pose,radius=11.):
        center=self.index(pose[:2]);r=int(radius/self.res)
        lo=np.maximum(0,center-r);hi=np.minimum(self.n,center+r+1)
        known=self.score[lo[0]:hi[0],lo[1]:hi[1]]>0
        clearance=distance_transform_edt(known)*self.res
        return lo,known,clearance


def crossing_view_clear(observation,line,axis):
    """Require measured ground rays through the crossing, not missing returns.

    This uses the declared planar-ground sensor contract. A blank scan provides
    no evidence of free space. Dense ground hits beyond the line demonstrate
    visibility through the robot-width corridor at obstacle-relevant height.
    Final motion still needs the normal road and dynamic trajectory checks.
    """
    if observation['time']-observation['scan_time']>.25:return False
    pose=np.asarray(observation['odometry']);q=np.asarray(observation['lidar_xyz'])[observation['lidar_valid_mask']]
    if not len(q):return False
    world=q[:,:2]@rotation(pose[2]).T+pose[:2];perp=np.array([-axis[1],axis[0]])
    along=float((line-pose[:2])@axis)
    if not .4<along<1.2:return False
    relative=world-line
    blocked=(np.abs(relative@axis)<.65)&(np.abs(relative@perp)<.65)&(q[:,2]>.12)&(q[:,2]<1.4)
    if np.any(blocked):return False
    distance=(world-pose[:2])@axis
    lateral=((world-pose[:2])@perp)*along/np.maximum(distance,.001)
    visible=(distance>along+1.)&(np.abs(q[:,2])<.12)&(np.abs(lateral)<.65)
    support=np.sort(lateral[visible])
    return bool(len(support)>=12 and support[0]<-.5 and support[-1]>.5 and np.max(np.diff(support))<.18)


class ReturnTracker:
    def __init__(self):self.tracks=[];self.serial=0;self.points=np.empty((0,2));self.last_time=-1.;self.cloud_history=deque(maxlen=25)

    def update(self,xyz,valid,pose,stamp,now):
        points=np.asarray(xyz)[valid];points=points[(points[:,2]>.12)&(points[:,2]<1.25)]
        world=points[:,:2]@rotation(pose[2]).T+pose[:2]
        if len(world):
            cells=np.unique(np.rint(world/.04).astype(int),axis=0);world=cells*.04
        self.points=world if len(world) else self.points
        if stamp<=self.last_time+1e-6:return
        self.last_time=stamp
        older=[q for t,q in self.cloud_history if .7<=stamp-t<=1.2 and len(q)]
        old_surface=np.concatenate(older) if older else None
        old_tree=cKDTree(old_surface) if old_surface is not None else None
        self.cloud_history.append((stamp,world.copy()))
        clusters=[]
        if len(world):
            tree=cKDTree(world);unseen=set(range(len(world)))
            while unseen:
                todo=[unseen.pop()];group=[]
                while todo:
                    i=todo.pop();group.append(i)
                    for j in tree.query_ball_point(world[i],.32):
                        if j in unseen:unseen.remove(j);todo.append(j)
                q=world[group]
                if len(q)<2:continue
                clusters.append({'center':(q.min(0)+q.max(0))/2,'extent':q.max(0)-q.min(0),'points':q})
        candidates=self.tracks;updated=[];used=set()
        for cluster in clusters:
            center=cluster['center'];best=None;distance=.85
            for j,t in enumerate(candidates):
                if j in used:continue
                d=np.linalg.norm(center-(t['center']+t['velocity']*(stamp-t['stamp'])))
                if d<distance:best=j;distance=d
            if best is None:
                self.serial+=1;t={**cluster,'id':self.serial,'velocity':np.zeros(2),'stamp':stamp,'hits':1,'history':deque([(stamp,center.copy())],maxlen=20)}
            else:
                used.add(best);old=candidates[best];dt=stamp-old['stamp'];history=old['history'];history.append((stamp,center.copy()))
                t={**cluster,'id':old['id'],'velocity':old['velocity'].copy(),'stamp':stamp,'hits':old['hits']+1,'history':history,'last_motion_stamp':old.get('last_motion_stamp',-100.),'motion_run':old.get('motion_run',0),'largest_extent':np.maximum(old.get('largest_extent',old['extent']),cluster['extent']),
                   'turn_planes':list(old.get('turn_planes',[])),'last_direction':old.get('last_direction'),'ever_confirmed':old.get('ever_confirmed',False)}
                if len(history)>2 and history[-1][0]-history[0][0]>.15:
                    velocity=(history[-1][1]-history[0][1])/(history[-1][0]-history[0][0]);speed=np.linalg.norm(velocity)
                    if speed>1.5:velocity*=1.5/speed
                    t['velocity']=.5*old['velocity']+.5*velocity
            displacement=np.linalg.norm(t['history'][-1][1]-t['history'][0][1]);span=t['history'][-1][0]-t['history'][0][0]
            translation=np.zeros(2);spread=10.
            if old_tree is not None:
                distances,indices=old_tree.query(cluster['points']);vectors=cluster['points']-old_surface[indices]
                translation=np.median(vectors,axis=0);spread=float(np.median(np.linalg.norm(vectors-translation,axis=1)))
                # Global surface matching avoids false velocity when a static
                # cluster splits/merges or its sampled bounding box changes.
                t['velocity']=.3*t['velocity']+.7*translation/.9
            moving_now=bool(t['hits']>=7 and span>=.6 and np.linalg.norm(translation)>.16 and spread<.20 and np.linalg.norm(t['velocity'])>.18 and np.max(t['extent'])<1.8)
            t['motion_run']=t.get('motion_run',0)+1 if moving_now else 0
            t['counterflow_confirmed']=bool(t['motion_run']>=3 and displacement>max(.45,.55*np.linalg.norm(t.get('largest_extent',t['extent']))))
            t['ever_confirmed']=bool(t.get('ever_confirmed') or t['counterflow_confirmed'])
            if moving_now:t['last_motion_stamp']=stamp
            t['moving']=moving_now or stamp-t.get('last_motion_stamp',-100.)<1.5
            if t['moving'] and len(t['history'])>=4:
                ta,pa=t['history'][-4];tb,pb=t['history'][-1];velocity=(pb-pa)/max(.01,tb-ta);speed=np.linalg.norm(velocity)
                if speed>1.5:velocity*=1.5/speed
                t['velocity']=.3*t['velocity']+.7*velocity
            if t['moving']:
                axis=observed_motion_axis(t)
                if axis is not None:
                    recent=list(t['history'])[-5:];times=np.array([time for time,_ in recent]);times-=times.mean()
                    positions=np.array([point for _,point in recent])@axis
                    speed=float(times@(positions-positions.mean())/max(1e-6,times@times))
                    t['velocity']=axis*np.clip(speed,-1.5,1.5)
            if t['moving'] and np.linalg.norm(t['velocity'])>.35:
                direction=t['velocity']/np.linalg.norm(t['velocity']);previous=t.get('last_direction')
                if previous is not None and previous@direction<-.65:
                    limit=max(p@previous for _,p in t['history']);planes=t.get('turn_planes',[])
                    planes=[(n,value) for n,value in planes if n@previous<.8];planes.append((previous.copy(),limit));t['turn_planes']=planes[-4:]
                t['last_direction']=direction.copy()
            updated.append(t)
        for j,t in enumerate(candidates):
            if j not in used and now-t['stamp']<.65:updated.append(t)
        self.tracks=updated


def astar(free,clearance,start,target,res,prior=None):
    shape=free.shape;start=tuple(start);target=tuple(target)
    if not (0<=start[0]<shape[0] and 0<=start[1]<shape[1]) or not free[start]:return None
    labels,_=label(free);component=labels==labels[start]
    if not (0<=target[0]<shape[0] and 0<=target[1]<shape[1]) or not component[target]:
        cells=np.argwhere(component)
        if not len(cells):return None
        # Closest reachable point to the requested local target. Unknown is not
        # filled using truth: turning in place observes new branches instead.
        target=tuple(cells[np.argmin(np.linalg.norm(cells-np.array(target),axis=1))])
    cost={start:0.};parent={};queue=[(0.,start)]
    while queue:
        _,u=heapq.heappop(queue)
        if u==target:break
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            v=(u[0]+dx,u[1]+dy)
            if not (0<=v[0]<shape[0] and 0<=v[1]<shape[1]) or not component[v]:continue
            if dx and dy and (not component[u[0]+dx,u[1]] or not component[u[0],u[1]+dy]):continue
            g=cost[u]+np.hypot(dx,dy)*(1+.08/max(.05,clearance[v]-.3)+(prior[v] if prior is not None else 0.))
            if g<cost.get(v,1e30):
                cost[v]=g;parent[v]=u;heapq.heappush(queue,(g+np.linalg.norm(np.array(v)-target),v))
    if target not in cost:return None
    path=[target]
    while path[-1]!=start:path.append(parent[path[-1]])
    return np.array(path[::-1])


def safe_path_target(path,pose,memory,lo,clearance,obstacle_clearance,minimum,obstacle_minimum):
    """Shrink lookahead when its chord would cut a nonconvex safe corridor.

    A* nodes can all be valid while a fixed long pursuit chord is invalid.
    Use the same visible reference in current control and predictive rollout.
    """
    nearest=int(np.argmin(np.linalg.norm(path-pose[:2],axis=1)));segment=path[nearest:]
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(segment,axis=0),axis=1))]
    for distance in [.38,.28,.20,.12,.06]:
        j=min(len(segment)-1,np.searchsorted(arc,distance));target=segment[j]
        length=np.linalg.norm(target-pose[:2]);n=max(2,int(np.ceil(length/.02))+1)
        samples=pose[:2]+(target-pose[:2])*np.linspace(0,1,n)[:,None];ij=memory.index(samples)-lo
        if np.any(ij<0) or np.any(ij>=clearance.shape):continue
        if np.all(clearance[ij[:,0],ij[:,1]]>=minimum) and np.all(obstacle_clearance[ij[:,0],ij[:,1]]>=obstacle_minimum):return target
    return segment[min(1,len(segment)-1)]


class SensorPlanner:
    def __init__(self,cfg):
        self.cfg=cfg;self.road=RoadMemory(cfg['camera'],head=cfg.get('ground_head'));self.tracker=ReturnTracker();self.mode='follow'
        self.pocket=None;self.counter_direction=None;self.counter_id=None;self.counter_seen=0.;self.counter_moving=0.;self.yield_not_before=0.;self.crossing=None;self.static_crossing_memory=[];self.last_path=np.empty((0,2));self.stats={'retreats':0,'waiting_steps':0,'proposal_steps':0,'overrides':0,'emergency_steps':0,'crossing_reservations':0,'static_holds_released':0}
        self.goal_history=[]
        self.counter_crosswise_since=None

    def refuge(self,free,clearance,lo,center,pose,direction,track,now):
        """Choose sensed connected paving outside the observed patrol corridor.

        The corridor comes exclusively from past local-goal inputs. A single
        tangent through a deviated robot pose is unsafe on a curved shared lane.
        """
        components,_=label(free);connected=components==components[tuple(center)] if free[tuple(center)] else np.zeros(free.shape,bool)
        cells=np.argwhere(free & connected & (clearance>.43));world=self.road.xy(cells+lo)
        if not len(world):return None
        progress=(world-pose[:2])@direction;distance=np.linalg.norm(world-pose[:2],axis=1)
        velocity=track['velocity'];speed=np.linalg.norm(velocity)
        if speed<.05:return None
        lateral=np.abs((world-track['center'])@np.array([-velocity[1],velocity[0]])/speed)
        corridor=cKDTree(np.asarray(self.goal_history)).query(world)[0]
        # Sampling the allowed goal history at <=6cm makes this bound
        # conservative by an additional 5cm over the desired corridor width.
        ok=(lateral>1.35)&(corridor>1.40)&(progress<1.8)&(distance<10.)
        if not np.any(ok):return None
        candidates=world[ok]
        futures=np.array([predicted_center(track,now-track['stamp']+dt) for dt in np.arange(0.,6.01,.25)])
        safe=np.min(np.linalg.norm(candidates[:,None,:]-futures[None,:,:],axis=2),axis=1)>1.6
        candidates=candidates[safe]
        if not len(candidates):return None
        score=np.linalg.norm(candidates-pose[:2],axis=1)+2*np.maximum(0,(candidates-pose[:2])@direction)
        return candidates[np.argmin(score)]

    def command(self,observation,goal,proposal=None):
        pose=np.asarray(observation['odometry']);now=observation['time'];goal=np.asarray(goal);goal_world=goal@rotation(pose[2]).T+pose[:2]
        if not self.goal_history:self.goal_history=[pose[:2].copy()]
        delta=goal_world-self.goal_history[-1];span=np.linalg.norm(delta)
        if span>=.06:
            previous=self.goal_history[-1].copy()
            self.goal_history.extend(previous+delta*f for f in np.linspace(0.,1.,int(np.ceil(span/.06))+1)[1:])
            self.goal_history=self.goal_history[-2500:]
        self.road.update(observation['rgb'],observation['camera_pose'],pose,observation['lidar_xyz'],observation['lidar_valid_mask'])
        self.tracker.update(observation['lidar_xyz'],observation['lidar_valid_mask'],pose,observation['scan_time'],now)
        lo,known,clearance=self.road.crop(pose);center=self.road.index(pose[:2])-lo;res=self.road.res
        radius=np.hypot(self.cfg['robot']['length'],self.cfg['robot']['width'])/2
        minimum=radius+.035;free=clearance>=minimum
        # Measured obstacle returns only; there is no static-world object list.
        points=self.tracker.points
        moving=[t for t in self.tracker.tracks if t.get('moving')]
        # Moving returns must not also be frozen into the static obstacle grid.
        # Otherwise a safe crossing maneuver is incorrectly blocked even after
        # the person will have moved away, forcing the robot to stop on-track.
        if len(points) and moving:
            keep=np.ones(len(points),bool)
            for t in moving:
                r=max(.35,float(np.linalg.norm(t['extent'])/2)+.15)
                keep&=np.linalg.norm(points-t['center'],axis=1)>r
            points=points[keep]
        if len(points):
            ij=self.road.index(points)-lo;valid=(ij>=0).all(1)&(ij<np.array(free.shape)).all(1);occupied=np.zeros(free.shape,bool);ij=ij[valid];occupied[ij[:,0],ij[:,1]]=True
            obst_clearance=distance_transform_edt(~occupied)*res;free&=obst_clearance>radius+.045
        else:obst_clearance=np.full(free.shape,10.)
        direction=(goal_world-pose[:2]);direction/=max(1e-6,np.linalg.norm(direction))
        if self.mode=='follow':
            threats=[]
            for t in moving:
                current=predicted_center(t,now-t['stamp']);delta=current-pose[:2]
                axis=observed_motion_axis(t);consistent=axis is None or abs(axis@direction)>.7
                if consistent and t.get('counterflow_confirmed') and t['velocity']@direction<-.65*np.linalg.norm(t['velocity']) and delta@direction>.8 and abs(np.cross(direction,delta))<1.4:threats.append((np.linalg.norm(delta),t))
            if threats:
                _,threat=min(threats,key=lambda q:q[0]);pocket=self.refuge(free,clearance,lo,center,pose,direction,threat,now)
                if pocket is not None:
                    self.pocket=pocket;self.mode='yield';self.counter_direction=direction.copy();self.counter_id=threat['id'];self.counter_seen=now;self.counter_moving=now;self.stats['retreats']+=1
                    self.counter_crosswise_since=None
                    self.yield_not_before=now+max(0.,(threat['center']-self.pocket)@direction)/max(.15,-threat['velocity']@direction)+5.
        if self.mode=='yield':
            track=next((t for t in self.tracker.tracks if t['id']==self.counter_id),None)
            if track is not None:
                current=predicted_center(track,now-track['stamp']);self.counter_seen=now
                if track.get('moving'):self.counter_moving=now
                futures=np.array([predicted_center(track,now-track['stamp']+dt) for dt in np.arange(0.,6.01,.25)])
                if np.min(np.linalg.norm(futures-self.pocket,axis=1))<1.4:
                    replacement=self.refuge(free,clearance,lo,center,pose,self.counter_direction,track,now)
                    if replacement is not None:self.pocket=replacement
                history=track['history'];stationary=bool(len(history)>5 and history[-1][0]-history[0][0]>1.4 and max(np.linalg.norm(p-history[-1][1]) for _,p in history)<.16)
                axis=observed_motion_axis(track);crosswise=axis is not None and abs(axis@self.counter_direction)<.55
                if crosswise:
                    if self.counter_crosswise_since is None:self.counter_crosswise_since=now
                else:self.counter_crosswise_since=None
                if self.counter_crosswise_since is not None and now-self.counter_crosswise_since>1.:self.mode='follow';self.pocket=None
                elif (current-pose[:2])@self.counter_direction < -1.6:self.mode='follow';self.pocket=None
                elif stationary and now-self.counter_moving>1.:self.mode='follow';self.pocket=None
                elif now>self.yield_not_before and now-self.counter_moving>2.:self.mode='follow';self.pocket=None
            elif now>self.yield_not_before and now-self.counter_seen>3.:self.mode='follow';self.pocket=None
        target=self.pocket if self.mode=='yield' else goal_world
        target_local=(target-pose[:2])@rotation(pose[2]);dist=np.linalg.norm(target_local)
        if self.mode=='follow' and np.linalg.norm(goal)<.17:return (0.,0.),{'mode':'terminal_stop'}
        was_committed=bool(self.crossing is not None and self.crossing['committed']);pending_crossing=False;observed_clear_candidate=False
        if self.crossing is not None:
            reservation=self.crossing;axis=reservation['axis'];perp=np.array([-axis[1],axis[0]])
            track=next((t for t in self.tracker.tracks if t['id']==reservation['id']),None)
            if track is None:
                # Packet loss can replace a local track identifier. Reassociate
                # from observed motion on the same crossing plane, not a
                # simulator identity, so a valid hold cannot become permanent.
                candidates=[t for t in moving if abs((t['center']-reservation['line'])@axis)<.35 and abs((t['center']-reservation['line'])@perp)<4.5]
                if candidates:
                    track=min(candidates,key=lambda t:abs((t['center']-reservation['line'])@axis));reservation['id']=track['id']
            if track is None:
                # A lost local identifier cannot hold the mission forever.
                # Release requires sustained positive visibility through the
                # crossing plus the complete crossing rollout below. Time
                # alone, black images and empty scans never release a hold.
                clear=now-reservation['last_seen']>1. and crossing_view_clear(observation,reservation['line'],axis)
                if clear:
                    reservation.setdefault('clear_since',observation['scan_time'])
                    observed_clear_candidate=observation['scan_time']-reservation['clear_since']>=.6
                    pending_crossing=bool(self.mode=='follow' and not reservation['committed'] and observed_clear_candidate)
                else:reservation.pop('clear_since',None)
            if track is not None:
                reservation.pop('clear_since',None)
                if not reservation['committed']:
                    # Refine during approach, then keep the waiting boundary
                    # stable. Short noisy fits must not alternate retreat and
                    # forward approach while the robot is already waiting.
                    if (reservation['line']-pose[:2])@axis<1.45:reservation['line_locked']=True
                    if not reservation.get('line_locked'):reservation['line']=crossing_intersection(track,reservation['line'],axis)
                remember_crossing_motion(reservation,track)
                position=predicted_center(track,now-track['stamp']);lateral=(position-reservation['line'])@perp;side_velocity=track['velocity']@perp
                if not reservation['committed'] and abs(lateral)>.5 and lateral*side_velocity<0:reservation['sign']=float(np.sign(lateral))
                side=lateral*reservation['sign'];along=(reservation['line']-pose[:2])@axis
                ready=along<1.2
                # Try a complete safe rollout just after the observed pass.
                # Waiting another metre wastes the gap without making the
                # subsequent4s crossing safer when the person can turn back.
                recent_pass=side < -.15 and abs(lateral)<1.45 and side_velocity*reservation['sign']<-.1
                far_clear=abs(lateral)>1.5*((max(0.,along)+1.05)/self.cfg['robot']['vmax']+.25)+.5 and lateral*side_velocity>0
                # Moving away now does not establish where a first, unobserved
                # reversal will happen. Wait for observed evidence or enough
                # distance for a bounded-speed return before crossing.
                pending_crossing=bool(self.mode=='follow' and not reservation['committed'] and ready and ((recent_pass and observed_turn_ahead(track,position)) or far_clear))
                reservation['last_seen']=now
                history=track.get('history',[])
                if len(history)>5 and history[-1][0]-history[0][0]>1.4 and now-reservation['created']>2.5:
                    stationary=max(np.linalg.norm(p-history[-1][1]) for _,p in history)<.35
                    if stationary and not track.get('counterflow_confirmed'):
                        self.static_crossing_memory.append((track['center'].copy(),now+15.));self.crossing=None;self.stats['static_holds_released']+=1
            if (pose[:2]-reservation['line'])@axis>1.05:self.crossing=None
        if self.mode=='follow' and self.crossing is None:
            self.static_crossing_memory=[(p,expiry) for p,expiry in self.static_crossing_memory if expiry>now]
            for track in moving:
                if not track.get('ever_confirmed',track.get('counterflow_confirmed')):continue
                if not track.get('counterflow_confirmed') and any(np.linalg.norm(track['center']-p)<.65 for p,_ in self.static_crossing_memory):continue
                velocity=track['velocity'];speed=np.linalg.norm(velocity);perp=np.array([-direction[1],direction[0]])
                delta=predicted_center(track,now-track['stamp'])-pose[:2];along=delta@direction;side=delta@perp;side_velocity=velocity@perp
                # Reserve the observed crossing corridor even while a person
                # moves away. Otherwise the robot enters the stopping margin
                # before the person turns and becomes an "incoming" target.
                if speed>.25 and abs(velocity@direction)<.6*speed and abs(side)<4.5 and abs(side_velocity)>.2:
                    line=crossing_intersection(track,pose[:2],direction);intersection=(line-pose[:2])@direction
                    if not .3<intersection<4.:continue
                    self.crossing={'id':track['id'],'line':line,'axis':direction.copy(),'sign':float(np.sign(side)),'committed':False,'last_seen':now,'created':now};self.stats['crossing_reservations']+=1;break
        crossing_backoff=False
        if self.mode=='follow' and self.crossing is not None and not self.crossing['committed']:
            reservation=self.crossing;along=(reservation['line']-pose[:2])@reservation['axis']
            if along<.85:reservation['backoff']=True
            if reservation.get('backoff'):
                if along>1.2 and observation['measured_twist'][0]<.15:reservation['backoff']=False
                else:
                    crossing_backoff=True;pending_crossing=False
                    target=reservation['line']-reservation['axis']*1.3
                    target_local=(target-pose[:2])@rotation(pose[2]);dist=np.linalg.norm(target_local)
        # Moving objects are checked in time-indexed rollouts below. Painting
        # their future swept region into a static map can itself cause unsafe
        # stopping on crossing lines, so A* uses measured static occupancy only.
        prior_cost=None
        if proposal is not None and self.mode=='follow' and not crossing_backoff and np.linalg.norm(goal)>.6:
            reference=np.asarray(proposal)[:,:2]@rotation(pose[2]).T+pose[:2]
            if np.isfinite(reference).all():
                xx,yy=np.ogrid[:free.shape[0],:free.shape[1]];near=((xx-center[0])**2+(yy-center[1])**2)*res**2<4.
                cells=np.argwhere(free&near);prior_cost=np.zeros(free.shape,np.float32)
                if len(cells):
                    distance=cKDTree(np.vstack([pose[:2],reference])).query(self.road.xy(cells+lo))[0]
                    prior_cost[cells[:,0],cells[:,1]]=.2*np.minimum(distance,1.)
                    self.stats['proposal_steps']+=1
        target_ij=self.road.index(target)-lo;path=astar(free,clearance,center,target_ij,res,prior_cost)
        if path is None:
            v_des=0.;w_des=0.;self.last_path=np.empty((0,2))
        else:
            self.last_path=self.road.xy(path+lo)
            point=safe_path_target(self.last_path,pose,self.road,lo,clearance,obst_clearance,minimum,radius+.045)
            near=(point-pose[:2])@rotation(pose[2]);angle=np.arctan2(near[1],near[0]);near_dist=np.linalg.norm(near)
            if near_dist<.02:
                if dist>.3:angle=np.arctan2(target_local[1],target_local[0]);v_des=0.
                else:v_des=min(.2,dist)
            else:v_des=min(.5,near_dist*1.5)*max(0.,np.cos(angle))**4
            w_des=float(np.clip(2.8*angle,-1,1))
            if abs(angle)>.45:v_des=0.
            if dist<.55:v_des=min(v_des,dist*.6)
            if self.mode=='yield' and dist<.22:v_des=0.;w_des=0.;self.stats['waiting_steps']+=1
        if self.crossing is None:pending_crossing=False
        hold=self.crossing if self.mode=='follow' and self.crossing is not None and not self.crossing['committed'] and not pending_crossing and not crossing_backoff else None
        if hold is not None:
            remaining=(hold['line']-pose[:2])@hold['axis']-.95
            v_des=min(v_des,float(np.sqrt(2*max(0.,remaining-.06))))
            if remaining<.12:v_des=0.;w_des=0.
        # Substep projection enforces measured road/obstacle clearance including
        # acceleration and braking. Truth labels are never consulted here.
        best=None;fallback=None;fallback_clearance=-1e9
        for candidate_v in [v_des,min(.25,v_des),min(.12,v_des),0.]:
            q=pose.copy();v,w=observation['measured_twist'];road_safe=True;actor_safe=True;minimum_actor=10.
            for step in range(50 if pending_crossing else 30):
                dt=.1
                if step==0:pv,pw=candidate_v,w_des
                elif len(self.last_path):
                    point=safe_path_target(self.last_path,q,self.road,lo,clearance,obst_clearance,minimum,radius+.045)
                    near=(point-q[:2])@rotation(q[2]);angle=np.arctan2(near[1],near[0]);pv=min(candidate_v,np.linalg.norm(near)*1.5)*max(0.,np.cos(angle))**4;pw=float(np.clip(2.8*angle,-1,1))
                    if abs(angle)>.45 or np.linalg.norm(q[:2]-self.last_path[-1])<.12:pv=0.
                else:pv,pw=0.,w_des
                if hold is not None:
                    remaining=(hold['line']-q[:2])@hold['axis']-.95
                    pv=min(pv,float(np.sqrt(2*max(0.,remaining-.06))))
                v+=np.clip(pv-v,-dt,dt);w+=np.clip(pw-w,-2*dt,2*dt);mid=q[2]+w*dt/2;q[:2]+=v*dt*np.array([np.cos(mid),np.sin(mid)]);q[2]=wrap(q[2]+w*dt)
                ij=self.road.index(q[:2])-lo
                if np.any(ij<0) or np.any(ij>=free.shape) or clearance[tuple(ij)]<minimum or obst_clearance[tuple(ij)]<radius+.035:road_safe=False;break
                for t in moving:
                    future=predicted_center(t,now-t['stamp']+(step+1)*dt);r=.65 if t['velocity']@direction<-.65*np.linalg.norm(t['velocity']) else .34
                    distance=np.linalg.norm(q[:2]-future)-radius-r
                    minimum_actor=min(minimum_actor,distance)
                    if distance<.07:actor_safe=False
            if road_safe and minimum_actor>fallback_clearance:fallback=(float(candidate_v),float(w_des));fallback_clearance=minimum_actor
            crossing_complete=not pending_crossing or (q[:2]-self.crossing['line'])@self.crossing['axis']>1.05
            if road_safe and actor_safe and crossing_complete:best=(float(candidate_v),float(w_des));break
        if pending_crossing:
            if best is not None and best[0]>=.35:self.crossing['committed']=True
            else:best=(0.,0.)
        if was_committed and self.crossing is not None and best is not None and best[0]<.3:
            reservation=self.crossing;along=(reservation['line']-pose[:2])@reservation['axis']
            if along>.8:
                reservation['committed']=False;track=next((t for t in moving if t['id']==reservation['id']),None)
                if track is not None:
                    perp=np.array([-reservation['axis'][1],reservation['axis'][0]]);side=(track['center']-reservation['line'])@perp
                    if abs(side)>.5:reservation['sign']=float(np.sign(side))
                best=(0.,0.)
        if best is None:
            # A moving obstacle may threaten a stationary body too. When no
            # predicted trajectory meets the margin, maximize separation while
            # staying inside measured static clearance, rather than freeze on
            # the crossing line. This is best effort, not a proof of safety.
            best=fallback if fallback is not None else (0.,0.);self.stats['emergency_steps']+=1
        if best[0]==0:self.stats['waiting_steps']+=1
        return best,{'mode':self.mode,'moving_tracks':len(moving),'pocket':self.pocket.tolist() if self.pocket is not None else None,'path_points':len(self.last_path),'v_des':v_des,'w_des':w_des,'crossing_hold':hold is not None,'crossing_backoff':crossing_backoff}
