"""Independent observed-free-space support for compact-object translation.

No return is never free. A positive witness is a past valid return BEYOND a
current 3-D point, with adjacent measured rays conservatively bounding it.
Timestamped world poses have already been applied by the sensor adapter.
"""
from collections import deque
import numpy as np
from scipy.spatial import cKDTree


def compact_surface_observable(points):
    """A new object needs multi-level, non-collinear measured surface support.

    This is an observability gate for the declared human/cart population, not
    a classifier for tiny/flat moving objects. Sparse returns remain obstacles.
    Existing confirmed tracks do not lose identity merely from this gate.
    """
    q=np.asarray(points)
    if len(q)<6 or np.ptp(q[:,2])<.08:return False
    singular=np.linalg.svd(q-q.mean(0),compute_uv=False)/np.sqrt(len(q))
    return bool(len(singular)>=2 and singular[1]>.035)


def fresh_motion_confirmation(state,stamp,positive,window=.35):
    """Require two distinct current scans, not retransmissions of one packet."""
    times=[t for t in state.get('free_witness_times',[]) if 0<=stamp-t<=window]
    if positive and (not times or stamp>times[-1]):times.append(stamp)
    state['free_witness_times']=times
    return len(times)>=2


class SurfaceMotionMemory:
    def __init__(self):self.history=deque(maxlen=20)

    def add(self,stamp,world,origin):
        if self.history and stamp<=self.history[-1]['stamp']:return
        vectors=np.asarray(world)-origin;distance=np.linalg.norm(vectors,axis=1)
        valid=np.isfinite(vectors).all(1)&(distance>.05)
        vectors=vectors[valid];distance=distance[valid]
        if len(distance)<4:return
        direction=vectors/distance[:,None]
        self.history.append(dict(stamp=stamp,origin=np.array(origin).copy(),directions=direction,
                                 ranges=distance,tree=None))

    def support(self,points,stamp):
        candidates=[h for h in self.history if .45<=stamp-h['stamp']<=1.35]
        selected=[]
        for delay in [.55,.85,1.15]:
            if not candidates:break
            h=min(candidates,key=lambda h:abs(stamp-h['stamp']-delay))
            if all(h is not q for q in selected):selected.append(h)
        if len(selected)<2 or len(points)<2:return dict(confirmed=False,count=0,fraction=0.,views=len(selected))
        votes=np.zeros(len(points),int)
        for h in selected:
            if h['tree'] is None:h['tree']=cKDTree(h['directions'])
            vectors=points-h['origin'];distance=np.linalg.norm(vectors,axis=1)
            direction=vectors/np.maximum(distance[:,None],1e-6)
            angular,ids=h['tree'].query(direction,k=4)
            perpendicular=angular*distance[:,None]
            nearby=perpendicular<.10
            # Use the nearest observed blocker among neighboring rays; a thin
            # foreground silhouette must not be erased by a farther neighbor.
            nearest=np.min(np.where(nearby,h['ranges'][ids],np.inf),axis=1)
            free=(perpendicular[:,0]<.065)&np.isfinite(nearest)&(nearest>distance+.18)
            votes+=free
        count=int(np.sum(votes>=2));fraction=count/len(points)
        return dict(confirmed=bool(count>=max(2,int(np.ceil(.10*len(points))))),
                    count=count,fraction=float(fraction),views=len(selected))
