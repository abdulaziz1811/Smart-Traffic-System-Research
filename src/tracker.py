"""
SORT Tracker
=============
Kalman filter + Hungarian assignment + IoU gating.
Ref: Bewley et al., ICIP 2016.
"""

from typing import List
import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou(a, b):
    x1,y1 = max(a[0],b[0]), max(a[1],b[1])
    x2,y2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/union if union>0 else 0.

def _iou_mat(a, b):
    m = np.zeros((len(a),len(b)), np.float32)
    for i in range(len(a)):
        for j in range(len(b)):
            m[i,j] = _iou(a[i], b[j])
    return m


class KalmanBoxTracker:
    _nid = 1
    def __init__(self, bb):
        self.id = KalmanBoxTracker._nid; KalmanBoxTracker._nid += 1
        self.x = np.zeros((7,1))
        self.x[:4] = self._z(bb).reshape(4,1)
        self.F = np.eye(7); self.F[0,4]=self.F[1,5]=self.F[2,6]=1.
        self.H = np.zeros((4,7)); np.fill_diagonal(self.H,1.)
        self.P = np.eye(7)*10.; self.P[4:,4:]*=1000.
        self.Q = np.eye(7)*0.01; self.R = np.eye(4)
        self.age=0; self.hits=0; self.streak=0; self.miss=0

    @staticmethod
    def _z(b):
        w,h = b[2]-b[0], b[3]-b[1]
        return np.array([b[0]+w/2, b[1]+h/2, w*h, w/max(h,1e-6)])

    @staticmethod
    def _bb(z):
        cx,cy,s,r = z.flatten()[:4]
        w=np.sqrt(max(s*r,1.)); h=s/max(w,1e-6)
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2])

    def predict(self):
        if self.x[6]+self.x[2]<=0: self.x[6]=0.
        self.x = self.F@self.x
        self.P = self.F@self.P@self.F.T + self.Q
        self.age+=1
        if self.miss>0: self.streak=0
        self.miss+=1
        return self._bb(self.x)

    def update(self, bb):
        self.miss=0; self.hits+=1; self.streak+=1
        z = self._z(bb).reshape(4,1)
        y = z - self.H@self.x
        S = self.H@self.P@self.H.T + self.R
        K = self.P@self.H.T@np.linalg.inv(S)
        self.x += K@y; self.P = (np.eye(7)-K@self.H)@self.P

    def bbox(self): return self._bb(self.x)


class SORTTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age=max_age; self.min_hits=min_hits; self.iou_th=iou_threshold
        self.trackers: List[KalmanBoxTracker] = []; self.frame=0

    @classmethod
    def from_config(cls, cfg):
        t = cfg["tracking"]
        return cls(t["max_age"], t["min_hits"], t["iou_threshold"])

    def update(self, dets):
        """dets: (N,5) [x1,y1,x2,y2,score] → returns (M,5) [x1,y1,x2,y2,id]"""
        self.frame += 1
        # Predict
        preds, dead = [], []
        for i,t in enumerate(self.trackers):
            p = t.predict()
            if np.any(np.isnan(p)): dead.append(i)
            else: preds.append(p)
        for i in reversed(dead): self.trackers.pop(i)
        preds = np.array(preds) if preds else np.empty((0,4))

        # Associate
        db = dets[:,:4] if len(dets) else np.empty((0,4))
        matched, um_d, um_t = [], set(range(len(db))), set(range(len(preds)))
        if len(db) and len(preds):
            cost = _iou_mat(db, preds)
            row, col = linear_sum_assignment(-cost)
            for r,c in zip(row, col):
                if cost[r,c] >= self.iou_th:
                    matched.append((r,c)); um_d.discard(r); um_t.discard(c)

        for d,t in matched: self.trackers[t].update(dets[d,:4])
        for d in um_d: self.trackers.append(KalmanBoxTracker(dets[d,:4]))

        # Collect & prune
        res, alive = [], []
        for t in self.trackers:
            if t.miss <= self.max_age:
                alive.append(t)
                if t.streak >= self.min_hits or self.frame <= self.min_hits:
                    res.append(np.append(t.bbox(), t.id))
        self.trackers = alive
        return np.array(res) if res else np.empty((0,5))
