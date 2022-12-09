from phonopy.interface.calculator import read_crystal_structure, write_crystal_structure
from phonopy import Phonopy
import phonopy
import os
import numpy as np

supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


if os.path.isfile('./CONTCAR_nospin'):
    unitcell, _ = read_crystal_structure('./CONTCAR_nospin', interface_mode='vasp')
elif os.path.isfile('./CONTCAR'):
    unitcell, _ = read_crystal_structure('./CONTCAR', interface_mode='vasp')

phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix)
write_crystal_structure('POSCAR_super', phonon.supercell, interface_mode='vasp')

phonon = phonopy.load(unitcell_filename='./POSCAR_super',force_constants_filename='./force_constants.hdf5')

supercells=[]
for i in range(100):
    temperature=np.random.normal(300,35)
    print('frame: '+str(i)+' '+str(temperature))
    phonon.generate_displacements(temperature=temperature,number_of_snapshots=1)
    supercells.append(phonon.supercells_with_displacements[-1])

if not os.path.exists('thermal_displacements'):
    os.mkdir('thermal_displacements')

os.chdir('thermal_displacements')

for i,frame in enumerate(supercells):
    os.mkdir('config-'+str(i))
    os.chdir('config-'+str(i))
    write_crystal_structure('POSCAR',frame)
    os.chdir("..")

os.chdir('..')