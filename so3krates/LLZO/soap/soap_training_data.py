import ase.io
import os
import numpy as np
import dscribe

from dscribe.descriptors import SOAP, EwaldSumMatrix
from dscribe.kernels import REMatchKernel, AverageKernel
from sklearn.preprocessing import normalize
import pickle

import argparse
parser = argparse.ArgumentParser(description='control for creating SOAP kernels',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-part','--part', type=int, help='which part of the kernel')
parser.add_argument('-bs','--batchsize', type=int, help='how many structures per batch',default=10)
args = parser.parse_args()


root_directory="/scratch/dy3/jy8620/LLZO/llzo_12_44/MD_testing_warmup"
R, F, E, z, pbc, unit_cell, idx_i, idx_j, node_mask = [[] for _ in range(9)]

j=0
features=[]
all_systems=[]
for i in range(26):
    i=i+1
    vasprun = root_directory + '/vasprun_'+str(i)+'.xml'
    for atoms in ase.io.read(vasprun, format='vasp-xml',index=':'):
        all_systems.append(atoms.__dict__['_calc'].__dict__['results']['energy'])
        _atomic_numbers = list(set(atoms.__dict__['arrays']['numbers']))

        desc = SOAP(species=_atomic_numbers,r_cut=5, n_max=7, l_max=6, sigma=0.1, periodic=True, sparse=False)
        #feature = desc.create(atoms['atoms'], positions=atoms['indicies'])
        feature = desc.create(atoms)
        feature = normalize(feature)
        features.append(feature)
        j+=1
        print(j)

re = REMatchKernel(metric="cosine", alpha=0.6, threshold=1e-6, gamma=2)
print("Building Kernel")

start=args.part*args.batchsize
end=(args.part+1)*args.batchsize

if start>len(all_systems):
    raise Exception("more than what we have in the database")

if end>len(all_systems):
    end = len(all_systems)
_kernel = re.create(x=features[start:end],y=features)
print("kernel build: " + str(np.shape(_kernel)))

pickle.dump([_kernel, all_systems[start:end]], open('kernel_part_'+str(args.part)+'.bp', 'wb'))

