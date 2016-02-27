from IPython.utils import io
import numpy as np
import sys

accFile = sys.argv[1]
volFile = sys.argv[2]
outFile = sys.argv[3]

# calculate some error metrics to evaluate actual vs. predicted EF values obtained from FCN model
data = np.transpose(np.loadtxt(accFile, delimiter=',')).astype('float')
ids, actual_edv, actual_esv, predicted_edv, predicted_esv = data
actual_ef = (actual_edv - actual_esv) / actual_edv
actual_ef_std = np.std(actual_ef)
actual_ef_median = np.median(actual_ef)
predicted_ef = (predicted_edv - predicted_esv) / predicted_edv # potential of dividing by zero, where there is no predicted EDV value
nan_idx = np.isnan(predicted_ef)
actual_ef = actual_ef[~nan_idx]
predicted_ef = predicted_ef[~nan_idx]
MAE = np.mean(np.abs(actual_ef - predicted_ef))
RMSE = np.sqrt(np.mean((actual_ef - predicted_ef)**2))
print 'Mean absolute error (MAE) for predicted EF: {:0.4f}'.format(MAE)
print 'Root mean square error (RMSE) for predicted EF: {:0.4f}'.format(RMSE)
print 'Standard deviation of actual EF: {:0.4f}'.format(actual_ef_std)
print 'Median value of actual EF: {:0.4f}'.format(actual_ef_median)

esv_mean = np.mean(actual_esv)
edv_mean = np.mean(actual_edv)

dias_k = actual_edv.sum()/predicted_edv.sum()
sys_k = actual_esv.sum()/predicted_esv.sum()
k = (dias_k+sys_k)/2
print 'predicted k = '+str(k)

dataV = np.transpose(np.loadtxt(volFile, delimiter=',')).astype('float')
ids, predicted_edv, predicted_esv = dataV

validStart=501
validEnd=700
used=[False]*(validEnd-validStart+1)

maxVol=600
out = open(outFile,'w')
out.write('Id')
for v in xrange(maxVol):
    out.write(',P'+str(v))
out.write('\n')
for i in xrange(ids.shape[0]):
    used[int(ids[i])-validStart]=True
    id = str(int(ids[i]))
    esv = int(round(predicted_esv[i]*k))
    edv = int(round(predicted_edv[i]*k))
    out.write(id+'_Systole')
    step=False
    for v in xrange(maxVol):
        if step:
            out.write(',1.0')
        elif v == esv:
            step=True
            out.write(',1.0')
        else:
            out.write(',0.0')
    out.write('\n')
    
    out.write(id+'_Diastole')
    step=False
    for v in xrange(maxVol):
        if step:
            out.write(',1.0')
        elif v == edv:
            step=True
            out.write(',1.0')
        else:
            out.write(',0.0')
    out.write('\n')


for i in range(validStart,validEnd+1):
  if not used[i-validStart]:
    id = str(i)
    esv = int(round(esv_mean))
    edv = int(round(edv_mean))
    out.write(id+'_Systole')
    step=False
    for v in xrange(maxVol):
        if step:
            out.write(',1.0')
        elif v == esv:
            step=True
            out.write(',1.0')
        else:
            out.write(',0.0')
    out.write('\n')
    
    out.write(id+'_Diastole')
    step=False
    for v in xrange(maxVol):
        if step:
            out.write(',1.0')
        elif v == edv:
            step=True
            out.write(',1.0')
        else:
            out.write(',0.0')
    out.write('\n')
      
