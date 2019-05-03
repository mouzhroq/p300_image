import os
import csv
import sys

import             numpy as np
import             scipy as sp
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from skimage.morphology import rectangle
from skimage.morphology import square
import skimage.filters.rank as rk

import scipy.io as spio

from scipy import signal



subject   = 's16-00-02-A'
store_mod = '02-A'

args =  sys.argv
print 'Argument List:', args

try:
    subject   = args[1]
    store_mod = args[2]
except:
    print "Running default subject"


print "processing subject: %s, modificador %s"%(subject,store_mod)


#Init paths
#=============
home      = 'C:\\Users\\Mouzhroq\\Desktop\\Nueva carpeta\\Nueva carpeta\\Servicio\\'
mainpath  = home+"bci\\src\\"
pthnpath  = mainpath+'python/'
folder    = 'scenario_screen'

sys.path.append( pthnpath )


import bci_tools as tools

#set behavior
#============
info = tools.load_manifest(mainpath+folder+'/manifest.csv')

PDGM = {  'nrows'  :int( info['nrows'  ] ),
          'ncols'  :int( info['ncols'  ] ),
          'ntrials':int( info['ntrials'] )   }

SR       = float( info['sr' ] )
LEPT     = int( info['lep'    ] )
LEPS     = int(  np.ceil(LEPT*SR/1000)  )
PWFS     = tools.get_filter_spef( info['passbands'] )
coef     = np.array(   [   signal.butter(2, 2*band  /SR, 'band') for band in PWFS  ]   )

#Load data from file 
#=======
#data = np.load(  '%s%s-%s.npz'%(info['storepath'],subject,store_mod)  )

data = np.load(  '%s%s-%s.npz'%(info['storepath'],subject[:-5],store_mod)  )
eeg  = data['data']
eeg  = tools.parallel_filter( eeg,coef[0] )

epochs = tools.get_epochs( eeg,LEPS,data['index_stim'] )


ntarg, nstim   = data['target_ids'].shape[0], data['single_ids'].shape[0]
mask           = data['target_ids'][np.newaxis].T == data['single_ids'].reshape(ntarg,nstim/ntarg)
mask           = mask.flatten()

ep_target  = epochs[ mask] 
ep_ntarget = epochs[~mask]
t          = np.linspace(0,LEPT,LEPS)
x,m,y=ep_target.shape

im=np.array([1,2,3,4,5,6,7,8])

tar=np.array([])
for i in range (0,x):
    tar=np.append(tar,ep_target[i,im,:])

ntar=np.array([])
for i in range (0,x*11):
    ntar=np.append(ntar,ep_ntarget[i,im,:])


tar=tar.reshape(x,8,512)
ntar=ntar.reshape(x*11,8,512)
kernel=rectangle(5,20)
#kernel=square(5)
#i=np.nonzero(tar.mean(axis=0)==np.max(tar.mean(axis=0)))



plt.figure()
for i in range (0,8):
    plt.subplot(8,5,(5*i)+1)
    plt.axis('off')
    plt.imshow(tar[:,i,:]/abs(tar[:,i,:]).max(),cmap='magma')
    #plt.axvline(i,linewidth=4, color='y')
    plt.subplot(8,5,(5*i)+2)
    plt.axis('off')
    plt.imshow(rk.mean(tar[:,i,:]/abs(tar[:,i,:]).max(),kernel),cmap='magma')
    #plt.axvline(i,linewidth=4, color='y')
    plt.subplot(8,5,(5*i)+3)
    plt.axis('off')
    plt.imshow(rk.median(tar[:,i,:]/abs(tar[:,i,:]).max(),kernel),cmap='magma')
    #plt.axvline(i,linewidth=4, color='y')
    plt.subplot(8,5,(5*i)+4)
    plt.axis('off')
    plt.imshow(rk.gradient(tar[:,i,:]/abs(tar[:,i,:]).max(),kernel),cmap='magma')
    #plt.axvline(i,linewidth=4, color='y')
    plt.subplot(8,5,(5*i)+5)
    plt.axis('off')
    plt.imshow(rk.entropy(tar[:,i,:]/abs(tar[:,i,:]).max(),kernel),cmap='magma')
    #plt.axvline(i,linewidth=4, color='y')

plt.show(block=False)  
plt.savefig('C:\\Users\\Mouzto\\Servicio\\s16-00-02-A-tar.png')     
