"""Anonymous, provisional jersey-colour groups; never roster identity."""
from collections import Counter
import cv2
import numpy as np


class JerseyTeams:
    def __init__(self):
        self.samples, self.centers, self.votes = [], None, {}
        self.sample_tracks=Counter()
        self.frames=0
        self.fit_quality=None

    def feature(self, frame, box):
        height, width = frame.shape[:2]
        a,b,c,d = box
        x1,x2 = max(0,int(a+.2*(c-a))),min(width,int(c-.2*(c-a)))
        y1,y2 = max(0,int(b+.2*(d-b))),min(height,int(b+.55*(d-b)))
        crop=frame[y1:y2,x1:x2]
        if crop.size<30:
            return None
        hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
        keep=~((hsv[:,:,0]>=28)&(hsv[:,:,0]<=95)&(hsv[:,:,1]>45))
        keep &= hsv[:,:,2]>30
        if keep.sum()<8:
            return None
        # Downweight brightness: shadows should not become the dominant team cue.
        return np.median(cv2.cvtColor(crop,cv2.COLOR_BGR2LAB)[keep],axis=0)*np.array([.35,1.,1.])

    def fit(self):
        data=np.array(self.samples,dtype=float)
        bins=np.floor(data/12).astype(int)
        counts=Counter(map(tuple,bins))
        modes=[]
        for key,n in counts.most_common():
            if n<3:continue
            center=np.median(data[np.all(bins==key,axis=1)],axis=0)
            if not modes or np.linalg.norm(center-modes[0])>=25:
                modes.append(center)
            if len(modes)==2:break
        if len(modes)<2:return
        a,b=modes
        centers=np.array([a,b])
        for _ in range(15):
            labels=np.linalg.norm(data[:,None,:]-centers[None,:,:],axis=2).argmin(axis=1)
            if min((labels==k).sum() for k in range(2))<max(4,len(data)*.2):
                return
            centers=np.array([np.median(data[labels==k],axis=0) for k in range(2)])
        if np.linalg.norm(centers[0]-centers[1])<25:
            return
        distances=np.linalg.norm(data-centers[labels],axis=1)
        if np.median(distances)>.5*np.linalg.norm(centers[0]-centers[1]):return
        self.centers=centers[np.argsort(centers[:,0])]
        self.fit_quality=dict(samples=len(data),tracks=len(self.sample_tracks),separation=round(float(np.linalg.norm(centers[0]-centers[1])),2))

    def annotate(self, frame, rows):
        self.frames+=1
        for row in rows:
            if row['object_type'] not in {'player','person'}:
                continue
            feature=self.feature(frame,row['bbox'])
            if feature is None:
                continue
            if self.centers is None:
                identity=row.get('track_id')
                if not identity or self.sample_tracks[identity]>=8:continue
                self.samples.append(feature);self.sample_tracks[identity]+=1
                if len(self.samples)>=32 and len(self.sample_tracks)>=4 and self.frames>=8:
                    self.fit()
                    if self.centers is None:
                        # Bounded warmup; new identities can still contribute.
                        self.samples=self.samples[-256:]
                continue
            distance=np.linalg.norm(self.centers-feature,axis=1)
            label=int(distance.argmin()); margin=abs(distance[0]-distance[1])/max(distance.max(),1)
            if distance[label]>60 or margin<.25:
                continue
            identity=row.get('track_id')
            if not identity:
                continue
            votes=self.votes.setdefault(identity,Counter());votes[label]+=1
            winner,n=votes.most_common(1)[0]
            if sum(votes.values())<3 or n/sum(votes.values())<.7:
                continue
            row.update(team=f'kit-{chr(65+winner)}',team_assignment_source='unsupervised_jersey_colour',
                       team_assignment_confidence=round(float(margin),3),team_review_status='unreviewed')
