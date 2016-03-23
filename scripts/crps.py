from IPython.utils import io
import numpy as np
import sys
import math
import re
import csv

distFile = sys.argv[1]
gtFile = sys.argv[2]

dataV = np.transpose(np.loadtxt(gtFile, delimiter=",",skiprows=1)).astype('float')

ids, t_esv, t_edv= dataV
smallest_id=100000
with open(distFile, 'r') as csvfile:
   distsCV=csv.reader(csvfile)

   labels=[]
   dists=[]
   skip=True
   for row in distsCV:
      if skip:
         skip=False
         continue
      labels.append( row[0] )
      m = re.match(r'(\d+)_\w+',row[0])
      id = int(m.group(1))
      if id<smallest_id:
         smallest_id=id
      dists.append( [float(n) for n in row[1:]] )


maxVol=600

def eval_dist(dist,actual_V):
   #print 'eval_dist'
   score=0
   #assert(dist[0]==0)
   #assert(dist[-1]==1.0)
   for x in xrange(600):
     assert(x==0 or dist[x-1]<=dist[x])
     H=0
     if x>= actual_V:
       H=1.0
     score+=pow(dist[x]-H,2)
   return score/600.0
#trainDist_sys=[0]*600
#trainDist_dias=[0]*600
accumScore=0
for r in range(len(labels)):
  mSys = re.match(r'(\d+)_Systole',labels[r])
  if mSys:
     id = int(mSys.group(1))-smallest_id
     accumScore+=eval_dist(dists[r],t_esv[id])
  else:
     mDias = re.match(r'(\d+)_Diastole',labels[r])
     id = int(mDias.group(1))-smallest_id
     accumScore+=eval_dist(dists[r],t_edv[id])
  
print 'CRPS: '+str(accumScore/(0.0+len(labels)))


