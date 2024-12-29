from ase.io.extxyz import read_extxyz
import os
import glob
import sys
import numpy as np

all_xyzs = glob.glob('*.xyz')
R, pbc, unit_cell, system_name, z = [[] for _ in range(5)]

for i_xyz, xyz in enumerate(all_xyzs):

    sys.stdout.write(
        "Structure: " + str(i_xyz + 1) + "/" + str(len(all_xyzs)) + '\r')
    if i_xyz != len(all_xyzs) - 1:
        sys.stdout.flush()

    for item in read_extxyz(open(xyz, 'r')):
        atoms = item

    R.append(atoms.get_positions())
    pbc.append(atoms.get_pbc())
    unit_cell.append(atoms.get_cell())
    z.append(atoms.get_atomic_numbers())
    system_name.append(xyz.replace('.xyz',''))

R, z, pbc, unit_cell, system_name = np.array(R), np.array(z), np.array(pbc), np.array(unit_cell), np.array(system_name)
np.savez('all_data_set_11_45', R=R, z=z, pbc=pbc, unit_cell=unit_cell, system_name=system_name)