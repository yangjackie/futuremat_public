import ase.io
from ase import Atoms
import os
import numpy as np
import dscribe
from dscribe.descriptors import SOAP
from dscribe.kernels import REMatchKernel
from sklearn.preprocessing import normalize
import pickle
import h5py
import argparse
parser = argparse.ArgumentParser(description='control for creating SOAP kernels',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-part','--part', type=int, help='which part of the kernel')
parser.add_argument('-bs','--batchsize', type=int, help='how many structures per batch',default=10)
args = parser.parse_args()


R, F, E, z, pbc, unit_cell, idx_i, idx_j, node_mask = [[] for _ in range(9)]
features=[]
all_systems=[]

#fingerprints for the ab initio data
abinitio_data_dir='/scratch/dy3/jy8620/LLZO/llzo_12_44/MD_testing_warmup/MD_320K'
vasprun=abinitio_data_dir+'/vasprun.xml'
j=0
for atoms in ase.io.read(vasprun, format='vasp-xml', index='::10'): #dont need to take all the frames, consecutive frames are too similar to each other
    cell = atoms.get_cell()
    all_systems.append({'energy':atoms.__dict__['_calc'].__dict__['results']['energy'],'ab_initio':True})
    atomic_numbers = list(set(atoms.__dict__['arrays']['numbers']))

    desc = SOAP(species=atomic_numbers, r_cut=5, n_max=7, l_max=6, sigma=0.1, periodic=True, sparse=False)
    # feature = desc.create(atoms['atoms'], positions=atoms['indicies'])
    feature = desc.create(atoms)
    feature = normalize(feature)
    features.append(feature)
    j += 1
    print('ab initio frame No.:\t'+str(j))

#fingerprints for the MLFF MD data
mlff_md_data='/scratch/dy3/jy8620/LLZO/llzo_12_44/MD_testing_warmup/train2/ckpt_dir/learning_curve_1000_longer_train_2/module/trajectory.h5'
traj = h5py.File(mlff_md_data)
#<KeysViewHDF5 ['atomic_numbers', 'forces', 'kinetic_energy', 'positions', 'potential_energy', 'temperature', 'velocities']>

#200000 frames
species='Li56O96La24Zr16'
#cells=[[12.9727012899999998,0.0,0.0],[0.0,12.9727012899999998,0.0],[0.0,0.0,12.9727012899999998]]
#atoms=Atoms(species,positions[0],cell=cells)

total_num_frames = len(traj['positions'])
for i in range(0,total_num_frames,10):
    positions = traj['positions'][i][0]
    all_systems.append({'energy': float(traj['potential_energy'][i][0]), 'ab_initio': False})
    atoms=Atoms(species,positions,cell=cell)
    desc = SOAP(species=atomic_numbers, r_cut=5, n_max=7, l_max=6, sigma=0.1, periodic=True, sparse=False)
    # feature = desc.create(atoms['atoms'], positions=atoms['indicies'])
    feature = desc.create(atoms)
    feature = normalize(feature)
    features.append(feature)

    print('ML-MD frame No.:\t' + str(i))

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