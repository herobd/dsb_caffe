import dicom, lmdb, cv2, re, sys
import os, fnmatch, shutil, subprocess
from IPython.utils import io
import numpy as np
np.random.seed(1234)
import matplotlib.pyplot as plt
import warnings
import pickle
import operator
import json

with open('SETTINGS.json', 'r') as f:
    settings = json.load(f)

if len(sys.argv)>1:
  mode=sys.argv[1]
else:
  mode='validate'

if len(sys.argv)>2:
  prefix=sys.argv[2]#or pickl
else:
  prefix=settings['MODEL_NAME']
load =  '_seg.pkl' in prefix
loadFile=prefix
if load:
  prefix=re.match(r'.*/(.*)_seg\.pkl',loadFile).group(1)
  print 'loading, prefix = '+prefix
if len(sys.argv)>3:
  gpu_num=int(sys.argv[3])

iteration=15000
if len(sys.argv)>4:
  iteration = int(sys.argv[4])

CAFFE_ROOT = settings['CAFFE_ROOT'] #"/home/brianld/bleeding_caffe"
caffe_path = os.path.join(CAFFE_ROOT, "python")
if caffe_path not in sys.path:
    sys.path.insert(0, caffe_path)

import caffe

class Dataset(object):
    dataset_count = 0

    def __init__(self, directory, subdir):
        # deal with any intervening directories
        while True:
            subdirs = next(os.walk(directory))[1]
            if len(subdirs) == 1:
                directory = os.path.join(directory, subdirs[0])
            else:
                break

        slices = []
        for s in subdirs:
            m = re.match('sax_(\d+)', s)
            if m is not None:
                slices.append(int(m.group(1)))

        slices_map = {}
        first = True
        times = []
        for s in slices:
            files = next(os.walk(os.path.join(directory, 'sax_%d' % s)))[2]
            offset = None

            for f in files:
                m = re.match('IM-(\d{4,})-(\d{4})\.dcm', f)
                if m is not None:
                    if first:
                        times.append(int(m.group(2)))
                    if offset is None:
                        offset = int(m.group(1))
                else:
                  m = re.match('IM-(\d{4,})-(\d{4})-(\d{4})\.dcm', f)
                  if m is not None:
			nothin='nothin'
                      
            first = False
            slices_map[s] = offset

        self.directory = directory
        self.time = sorted(times)
        self.slices = sorted(slices)
        self.slices_map = slices_map
        Dataset.dataset_count += 1
        self.name = subdir

    def _filename(self, s, t):
        return os.path.join(self.directory,
                            'sax_%d' % s,
                            'IM-%04d-%04d.dcm' % (self.slices_map[s], t))

    def _read_dicom_image(self, filename):
        d = dicom.read_file(filename)
        img = d.pixel_array.astype('int')
        #img = np.expand_dims(img, axis=0)
        #img = np.concatenate((img,img,img),axis=0)
        return img

    def _read_all_dicom_images(self):
        f1 = self._filename(self.slices[0], self.time[0])
        d1 = dicom.read_file(f1)
        (x, y) = d1.PixelSpacing
        (x, y) = (float(x), float(y))
        f2 = self._filename(self.slices[1], self.time[0])
        d2 = dicom.read_file(f2)

        # try a couple of things to measure distance between slices
        try:
            dist = np.abs(d2.SliceLocation - d1.SliceLocation)
        except AttributeError:
            try:
                dist = d1.SliceThickness
            except AttributeError:
                dist = 8  # better than nothing...

        self.images = np.array([[self._read_dicom_image(self._filename(d, i))
                                 for i in self.time]
                                for d in self.slices])
        self.dist = dist
        self.area_multiplier = x * y

    def load(self):
        self._read_all_dicom_images()

#############

MEAN_VALUE = 77
THRESH = 0.5
net=None
def calc_all_areas(images):
    (num_images, times, _, _) = images.shape
    print 'calc_all_areas'
    all_probs = [{} for i in range(times)]
    all_masks = [{} for i in range(times)]
    all_areas = [{} for i in range(times)]
    for i in range(times):
        for j in range(num_images):
            # print 'Calculating area for time %d and slice %d...' % (i, j)
            img = images[j][i]
#
            if 'skip' in prefix and 'vgg16' in prefix:
                for dim in range(0,2):
                    if img.shape[dim]%16!=0:
                       dif=img.shape[dim] - 16*int(img.shape[dim]/16)
                       cut=dif/2
                       extraCut=dif%2
                       if dim==0:
                          img=img[extraCut+cut:-cut,:]
                       else:
                          img=img[:,extraCut+cut:-cut]
#
            in_ = np.expand_dims(img, axis=0)
            
            in_ -= np.array([MEAN_VALUE])
            in_ = np.concatenate((in_,in_,in_),axis=0)
            #print 'hey hey, here is shape: '+str(in_.shape)
            net.blobs['data'].reshape(1, *in_.shape)
            net.blobs['data'].data[...] = in_
            net.forward()
            prob = net.blobs['prob'].data
            obj = prob[0][1]
            preds = np.where(obj > THRESH, 1, 0)
            all_probs[i][j]=obj
            all_masks[i][j] = preds
            all_areas[i][j] = np.count_nonzero(preds)
    f = open('seg_results/'+prefix+'_seg.pkl','w')
    if f:
        pickle.dump(all_probs,f)
        f.close()
    else:
        print 'error opening: '+'seg_results/'+prefix+'_seg.pkl'
    return all_masks, all_areas #, all_probs

def calc_all_areas_from_probs(images,all_probs):
    (num_images, times, _, _) = images.shape
    print 'calc_all_areas_FROM_PROBS'
    all_masks = [{} for i in range(times)]
    all_areas = [{} for i in range(times)]
    for i in range(times):
        for j in range(num_images):

            obj = all_probs[i][j]
            preds = np.where(obj > THRESH, 1, 0)
            all_masks[i][j] = preds
            all_areas[i][j] = np.count_nonzero(preds)
    return all_masks, all_areas

def calc_total_volume(areas, area_multiplier, dist):
    slices = np.array(sorted(areas.keys()))
    modified = [areas[i] * area_multiplier for i in slices]
    vol = 0
    for i in slices[:-1]:
        a, b = modified[i], modified[i+1]
        subvol = (dist/3.0) * (a + np.sqrt(a*b) + b)
        vol += subvol / 1000.0  # conversion to mL
    return vol

#def calc_accumulative_distributions(probs):
#    slices = np.array(sorted(probs.keys()))
    

def segment_dataset(dataset):
    # shape: num slices, num snapshots, rows, columns
    print 'Calculating areas...'
    all_masks, all_areas = calc_all_areas(dataset.images)
    print 'Calculating volumes...'
    area_totals = [calc_total_volume(a, dataset.area_multiplier, dataset.dist)
                   for a in all_areas]
    print 'Calculating EF...'
    edv_idx,edv = max(enumerate(area_totals), key=operator.itemgetter(1))
    esv_idx,esv = min(enumerate(area_totals), key=operator.itemgetter(1))
    ef = (edv - esv) / edv
    print 'Done, EF is {:0.4f}'.format(ef)
    #dias_k = np.sum(actual_dv)/np.num(edv)
    #sys_k = np.sum(actual_sv)/np.sum(esv)
    #k=(dias_k+sys_k)/2.0;
    #print 'calulated k = ' + str(k)   
    #dias_probs = [all_probs[i] for i in edv_idx]
    #sys_probs = [all_probs[i] for i in esv_idx]
    #dataset.dias_dist = calc_accumulative_distributions(dias_probs)
    #dataset.sys_dist = calc_accumulative_distributions(sys_probs)
 
    dataset.edv = edv
    dataset.esv = esv
    dataset.ef = ef

def segment_dataset_load(dataset,loadFile):
    # shape: num slices, num snapshots, rows, columns
    f=open(loadFile,'r')
    all_probs = pickle.load(f)
    f.close()
    print 'Calculating areas...'
    all_masks, all_areas = calc_all_areas_from_probs(dataset.images,all_probs)
    print 'Calculating volumes...'
    area_totals = [calc_total_volume(a, dataset.area_multiplier, dataset.dist)
                   for a in all_areas]
    print 'Calculating EF...'
    edv_idx,edv = max(enumerate(area_totals), key=operator.itemgetter(1))
    esv_idx,esv = min(enumerate(area_totals), key=operator.itemgetter(1))
    ef = (edv - esv) / edv
    print 'Done, EF is {:0.4f}'.format(ef)
    #dias_k = np.sum(actual_dv)/np.num(edv)
    #sys_k = np.sum(actual_sv)/np.sum(esv)
    #k=(dias_k+sys_k)/2.0;
    #print 'calulated k = ' + str(k)   
    #dias_prob = all_probs[edv_idx]
    #sys_prob = all_probs[esv_idx]
    #dataset.dias_dist = calc_accumulative_distributions(dias_prob,dataset.area_multiplier,dataset.dist,edv)
    #dataset.sys_dist = calc_accumulative_distributions(sys_prob,dataset.area_multiplier,dataset.dist,esv)

    dataset.edv = edv
    dataset.esv = esv
    dataset.ef = ef
###############
#%%time
#prefix2='dag'
# We capture all standard output from IPython so it does not flood the interface.
#if True:
with io.capture_output() as captured:
    # edit this so it matches where you download the DSB data
    if mode=='train':
      studies_dir =  settings['TRAIN_DATA_PATH']
    else:
      studies_dir =  settings['TEST_DATA_PATH'] #os.path.join(DATA_PATH, mode)
    DATA_PATH = os.path.join(studies_dir,'..') #'/scratch/cardiacMRI/'
    #print 'init net'
    if not load:
      caffe.set_mode_gpu()
      caffe.set_device(gpu_num)
      if mode=='train':
        weightsLoc = str('data/sunnybrook_training/network/'+prefix+'_iter_'+str(iteration)+'.caffemodel')
      else:
        weightsLoc = str(settings['MODEL_PATH'])
      prototxtLoc=str(os.path.join(settings['NET_DEFS'],prefix+'_deploy.prototxt'))
      print 'creating net with '+prototxtLoc+' and '+weightsLoc
      net = caffe.Net(prototxtLoc, weightsLoc, caffe.TEST)

    print 'DICOM dir is '+studies_dir
    studies = next(os.walk(studies_dir))[1]
    #print 'load csv'
    if mode=='train':
        labels = np.loadtxt(os.path.join(DATA_PATH, 'train.csv'), delimiter=',',
                        skiprows=1)

        label_map = {}
        for l in labels:
            label_map[l[0]] = (l[2], l[1])
        accuracy_csv = open(os.path.join(settings['OUT_PATH'],'train_volumes_'+prefix+'.csv'), 'w')
    else:
        accuracy_csv = open(os.path.join(settings['OUT_PATH'],mode+'_volumes_'+prefix+'.csv'), 'w')
    #if os.path.exists('output'):
    #    shutil.rmtree('output')
    #os.mkdir('output')


    for s in studies:
        dset = Dataset(os.path.join(studies_dir, s), s)
        print 'Processing dataset %s...' % dset.name
        try:
            dset.load()
            if mode=='train':
                (edv, esv) = label_map[int(dset.name)]
                if load:
                  segment_dataset_load(dset,loadFile)
                else:
                  segment_dataset(dset)
                accuracy_csv.write('%s,%f,%f,%f,%f\n' %
                               (dset.name, edv, esv, dset.edv, dset.esv))
            else:
                if load:
                  segment_dataset_load(dset,loadFile)
                else:
                  segment_dataset(dset)
                accuracy_csv.write('%s,%f,%f\n' %
                               (dset.name, dset.edv, dset.esv))
                
        except Exception as e:
            print '***ERROR***: Exception %s thrown by dataset %s' % (str(e), dset.name)

    accuracy_csv.close()

# We redirect the captured stdout to a log file on disk.
# This log file is very useful in identifying potential dataset irregularities that throw errors/exceptions in the code.
with open(prefix+'_logs.txt', 'w') as f:
    f.write(captured.stdout)
