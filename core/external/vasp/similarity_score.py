"""
Module containing functions to measure the structrual similarities among MD trajectory frames using the SOAP-REMatch kernel.
"""
from dscribe.descriptors import SOAP
from dscribe.kernels import REMatchKernel

import argparse
from pymatgen.core.trajectory import Trajectory
from ase.io.vasp import read_vasp_xdatcar,read_vasp
from ase.io import read
from sklearn.preprocessing import normalize
from core.external.vasp.anharmonic_score import AnharmonicScore
import matplotlib.pyplot as plt
import os

from matplotlib import rc

rc('text', usetex=True)

import matplotlib.pylab as pylab

params = {'legend.fontsize': '15',
          'figure.figsize': (7, 6),
          'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)

trajectory = Trajectory.from_file("./vasprun_100K_correct.xml")
trajectory.write_Xdatcar(filename='XDATCAR_temp')
images = read_vasp_xdatcar('XDATCAR_temp',index=None)
print("total number of frames: "+str(len(images)))
os.remove('./XDATCAR_temp')


reference_frame = read_vasp('./POSCAR_equ')
desc = SOAP(species=[55,50,82,53], rcut=6.0, nmax=9, lmax=9, sigma=0.3, periodic=True, crossover=True, sparse=False)
ref_features = desc.create(reference_frame)
ref_features = normalize(ref_features)

re = REMatchKernel(metric="linear", alpha=1, threshold=1e-6, gamma=1)

similarities = []
for i,image in enumerate(images):
    image_features = desc.create(image)
    image_features = normalize(image_features)
    re_kernel = re.create([image_features, ref_features])
    print(i,re_kernel[0][1])
    similarities.append(re_kernel[0][1])

scorer = AnharmonicScore(md_frames='./vasprun_100K_correct.xml', ref_frame='./POSCAR_equ', atoms=None)
sigmas, _ = scorer.structural_sigma(return_trajectory=True)

fig, ax1 = plt.subplots(figsize=(7,5))

color = '#1fbfb8'
ax1.set_xlabel('$t$ (fs)')
ax1.set_ylabel('$\\sigma(t)$', color=color)
ax1.plot(range(len(sigmas)), sigmas, '.-', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = '#031163'
ax2.set_ylabel('kernel similarity', color=color)  # we already handled the x-label with ax1
ax2.plot(range(len(sigmas)), similarities, '.-', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.savefig("soap_100K.pdf")
