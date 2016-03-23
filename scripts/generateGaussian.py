from IPython.utils import io
import numpy as np
import sys
import math

trainFile = sys.argv[1]
validateFile = sys.argv[2]
#volFile = "/home/yunan/anaconda2/scripts/accuracy_caffe.csv" #sys.argv[2]
#outFile = "/home/yunan/anaconda2/scripts/out.txt" #sys.argv[3]

# calculate some error metrics to evaluate actual vs. predicted EF values obtained from FCN model
#data = np.transpose(np.loadtxt(accFile, delimiter=',')).astype('float')
#ids, actual_edv, actual_esv, predicted_edv, predicted_esv = data
#actual_ef = (actual_edv - actual_esv) / actual_edv
#actual_ef_std = np.std(actual_ef)
#actual_ef_median = np.median(actual_ef)
#predicted_ef = (predicted_edv - predicted_esv) / predicted_edv # potential of dividing by zero, where there is no predicted EDV value
#nan_idx = np.isnan(predicted_ef)
#actual_ef = actual_ef[~nan_idx]
#predicted_ef = predicted_ef[~nan_idx]
#MAE = np.mean(np.abs(actual_ef - predicted_ef))
#RMSE = np.sqrt(np.mean((actual_ef - predicted_ef)**2))
#print 'Mean absolute error (MAE) for predicted EF: {:0.4f}'.format(MAE)
#print 'Root mean square error (RMSE) for predicted EF: {:0.4f}'.format(RMSE)
#print 'Standard deviation of actual EF: {:0.4f}'.format(actual_ef_std)
#print 'Median value of actual EF: {:0.4f}'.format(actual_ef_median)

#esv_mean = np.mean(actual_esv)
#edv_mean = np.mean(actual_edv)

dias_k = 1.0 #actual_edv.sum()/predicted_edv.sum()
sys_k = 1.0 #actual_esv.sum()/predicted_esv.sum()
#k = (dias_k+sys_k)/2
#print 'predicted k = '+str(k)

#for i in xrange(ids.shape[0]):

#    id = str(int(ids[i]))
#    sys_k += actual_esv[i]/predicted_esv[i]
#    dias_k += actual_edv[i]/predicted_edv[i]

#dias_k /=ids.shape[0]
#sys_k /=ids.shape[0]
#print 'predicted dias_k = '+str(dias_k)
#print 'predicted sys_k = '+str(sys_k)

#predicted_esv_adj = predicted_esv*sys_k
#MAEsv = np.mean(np.abs(actual_esv - predicted_esv_adj))
#RMSEsv = np.sqrt(np.mean((actual_esv - predicted_esv_adj)**2))
#print 'Mean absolute error (MAE) for predicted systole vol: {:0.4f}'.format(MAEsv)
#print 'Root mean square error (RMSE) for predicted systole vol: {:0.4f}'.format(RMSEsv)
#predicted_edv_adj = predicted_edv*dias_k
#MAEdv = np.mean(np.abs(actual_edv - predicted_edv_adj))
#RMSEdv = np.sqrt(np.mean((actual_edv - predicted_edv_adj)**2))
#print 'Mean absolute error (MAE) for predicted diastole vol: {:0.4f}'.format(MAEdv)
#print 'Root mean square error (RMSE) for predicted diastole vol: {:0.4f}'.format(RMSEdv)

dataV = np.transpose(np.loadtxt(trainFile, delimiter=",")).astype('float')

ids, t_edv, t_esv, p_edv, p_esv = dataV

s_dev = 0.0
d_dev = 0.0
count = 0.0

for v in xrange(ids.size):
    if ((p_edv[v] != 0) & (p_esv[v] != 0)):
	multiplier = (t_esv[v]+ t_edv[v] )/ (p_esv[v] + p_edv[v])
	s_dev = s_dev + (t_esv[v] - p_esv[v]*multiplier)*(t_esv[v] - p_esv[v]*multiplier)
	d_dev = d_dev + (t_edv[v] - p_edv[v]*multiplier)*(t_edv[v] - p_edv[v]*multiplier) 
	count = count + 1  
s_dev = s_dev / count
d_dev = d_dev / count
print s_dev
print d_dev

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

out = open("train_guasDist.csv",'w')
out.write('Id')
for v in xrange(maxVol):
    out.write(',P'+str(v))
out.write('\n')

accumScore=0
for i in xrange(ids.size):
    id = str(int(ids[i]))
    if ((p_edv[i] != 0) & (p_esv[i] != 0)):

	multiplier = 1 #(t_esv[i]+ t_edv[i] )/ (p_esv[i] + p_edv[i])
	
        dist_sys=[0]*maxVol
        out.write(id+'_Systole')
        for v in xrange(maxVol):
            dist_sys[v]=0.5*(1 + math.erf( (v - p_esv[i]*multiplier)/(math.sqrt(s_dev*2))))
	    outstring = ',' + str(dist_sys[v])
            out.write(outstring)
        out.write('\n')
    
        dist_dias=[0]*maxVol
        out.write(id+'_Diastole')
        for v in xrange(maxVol):
            dist_dias[v]=(0.5)*(1 + math.erf((v - p_edv[i]*multiplier)/(math.sqrt(d_dev*2))))
            out.write(',')
	    out.write(str(dist_dias[v]))
        out.write('\n')
        
        accumScore+=eval_dist(dist_sys,t_esv[i])
        accumScore+=eval_dist(dist_dias,t_edv[i])
out.close()
print 'CRPS on test data: '+str(accumScore/(2*ids.size))



dataV = np.transpose(np.loadtxt(validateFile, delimiter=",")).astype('float')

ids, validate_t_edv, validate_t_esv, p_edv, p_esv = dataV
validateDist_sys=[0]*600
validateDist_dias=[0]*600

out = open("validate_guasDist.csv",'w')
out.write('Id')
for v in xrange(maxVol):
    out.write(',P'+str(v))
out.write('\n')

accumScore=0
for i in xrange(ids.size):
    id = str(int(ids[i]))
    if ((p_edv[i] != 0) & (p_esv[i] != 0)):

	multiplier = 1 #(t_esv[i]+ t_edv[i] )/ (p_esv[i] + p_edv[i])
	
        dist_sys=[0]*maxVol
        out.write(id+'_Systole')
        for v in xrange(maxVol):
            dist_sys[v]=0.5*(1 + math.erf( (v - p_esv[i]*multiplier)/(math.sqrt(s_dev*2))))
	    outstring = ',' + str(dist_sys[v])
            out.write(outstring)
        out.write('\n')
    
        dist_dias=[0]*maxVol
        out.write(id+'_Diastole')
        for v in xrange(maxVol):
            dist_dias[v]=(0.5)*(1 + math.erf((v - p_edv[i]*multiplier)/(math.sqrt(d_dev*2))))
            out.write(',')
	    out.write(str(dist_dias[v]))
        out.write('\n')
        
        accumScore+=eval_dist(dist_sys,validate_t_esv[i])
        accumScore+=eval_dist(dist_dias,validate_t_edv[i])
out.close()
print 'CRPS on validation data: '+str(accumScore/(2*ids.size))
