"""Measured crossing reservations, independent of neural or grid search.

Only observed track histories/turns and allowed local goals are used. Crossing
release is a prediction under bounded measured motion, not a human guarantee.
"""
import numpy as np
from collections import deque
from navigation.sensor_planner import rotation,observed_motion_axis,crossing_intersection,remember_crossing_motion


class CrossingGuard:
    def __init__(self,hold_distance=.95,clear_distance=.85):
        self.hold_distance=hold_distance;self.clear_distance=clear_distance
        self.reservation=None;self.committed=False;self.commit_time=None
        self.waiting=False;self.distance=None;self.last_budget=None

    def update(self,tracks,pose,goal,now,measured_v):
        pose=np.asarray(pose);world_axis=rotation(pose[2])@np.asarray(goal)
        world_axis/=max(1e-6,np.linalg.norm(world_axis))
        active=[t for t in tracks if t.get('is_dynamic')]
        if self.reservation is not None:
            r=self.reservation;along=float((r['line']-pose[:2])@r['axis'])
            if along < -self.clear_distance:
                self.reservation=None;self.committed=False;self.commit_time=None
        if self.reservation is None:
            candidates=[]
            for track in active:
                velocity=track['velocity'];speed=np.linalg.norm(velocity)
                if speed<.2 or abs(velocity@world_axis)>.55*speed:continue
                motion_axis=observed_motion_axis(track)
                # A turning/partially occluded car can have a noisy transverse
                # instantaneous velocity. A crossing needs a measured spatial
                # corridor transverse to the patrol direction as well.
                if motion_axis is None or abs(motion_axis@world_axis)>.55:continue
                line=crossing_intersection(track,pose[:2],world_axis)
                along=float((line-pose[:2])@world_axis)
                normal=np.array([-world_axis[1],world_axis[0]])
                if .6<along<4.5 and abs((track['center']-line)@normal)<4.5:
                    candidates.append((along,track,line))
            if candidates:
                _,track,line=min(candidates,key=lambda q:q[0])
                self.reservation=dict(line=line.copy(),axis=world_axis.copy(),track_id=track['id'],last_seen=now)
        r=self.reservation
        if r is None:
            self.waiting=False;self.distance=None;self.last_budget=None;return
        normal=np.array([-r['axis'][1],r['axis'][0]])
        track=next((t for t in active if t['id']==r['track_id']),None)
        if track is None:
            nearby=[t for t in active if abs((t['center']-r['line'])@r['axis'])<.65 and abs(t['velocity']@r['axis'])<.5*np.linalg.norm(t['velocity'])]
            if nearby:
                track=min(nearby,key=lambda t:np.linalg.norm(t['center']-r.get('last_position',t['center'])))
                r['track_id']=track['id']
        along=float((r['line']-pose[:2])@r['axis']);self.distance=along
        required=max(0.,along+self.clear_distance)/.45+.65
        budget=0.
        if track is not None:
            r['last_seen']=now;r['last_position']=track['center'].copy()
            remember_crossing_motion(r,track)
            if not self.committed and along>1.45:
                refined=crossing_intersection(track,pose[:2],r['axis'])
                # Keep the fitted plane transverse to the reserved route axis.
                shift=float((refined-r['line'])@r['axis'])
                r['line']+=np.clip(shift,-.08,.08)*r['axis']
                along=float((r['line']-pose[:2])@r['axis']);self.distance=along
            side=float((track['center']-r['line'])@normal)
            rate=float(track['velocity']@normal)
            history=r.setdefault('sides',deque(maxlen=40))
            if not history or track['stamp']>history[-1][0]:history.append((track['stamp'],side))
            if abs(side)>.20:
                previous=r.get('side_sign')
                if previous is not None and previous*side<0:r['last_pass']=now
                r['side_sign']=np.sign(side)
            speeds=r.setdefault('speeds',deque(maxlen=20))
            if abs(rate)>.35:speeds.append(abs(rate))
            recent=list(history)[-4:]
            spatial_departing=len(recent)>=4 and (recent[-1][1]-recent[0][1])*side>.06
            # Release directly after an observed pass, never while slowing at
            # a distant endpoint where filtered velocity can lag a reversal.
            passed=now-r.get('last_pass',-1e6)<2.
            if passed and spatial_departing and side*rate>0 and .20<abs(side)<1.25 and now-track['stamp']<=.35:
                speed=max(abs(rate),float(np.median(speeds)) if speeds else abs(rate))+.10
                # An unobserved turnaround could happen now. Only an actually
                # observed reversal boundary permits additional departure time.
                budget=abs(side)/speed
                ahead=[]
                for n,limit in track.get('turn_planes',[]):
                    approach=track['velocity']@n
                    if approach>.05 and limit-track['center']@n>=-.05:
                        t=max(0.,(limit-track['center']@n)/approach)
                        turn=track['center']+track['velocity']*t
                        ahead.append(t+abs((turn-r['line'])@normal)/speed)
                if ahead:budget=min(ahead)
        self.last_budget=float(budget)
        if self.committed:
            # If the planner never got moving before the line, re-arm; never
            # remain permanently released on a stale once-safe observation.
            if along>.80 and abs(measured_v)<.03 and now-self.commit_time>1.:
                self.committed=False
        if not self.committed and budget>required:
            self.committed=True;self.commit_time=now
        self.waiting=not self.committed

    def cap_command(self,v,w,amax=1.):
        if not self.waiting or self.distance is None:return float(v),float(w)
        remaining=self.distance-self.hold_distance-.025
        cap=np.sqrt(2*amax*max(0.,remaining))
        return float(min(v,cap)),float(w if cap>.03 else 0.)
