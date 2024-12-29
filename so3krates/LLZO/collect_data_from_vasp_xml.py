from typing import Optional
  
import ase
import ase.io
import os
import numpy as np


mix = True

root_directory = "/scratch/dy3/jy8620/LLZO/pure/MD_set/"

if not mix:
    temps = [200, 300, 400, 500, 600, 700]
else:
    temps = [200, 300, 400]

if mix:
    R, F, E, z, pbc, unit_cell, idx_i, idx_j, node_mask = [[] for _ in range(9)]

for temp in temps:

    if not mix:
        R, F, E, z, pbc, unit_cell, idx_i, idx_j, node_mask = [[] for _ in range(9)]

    folder_name = 'MD_' + str(temp)
    folder_path = os.path.join(root_directory, folder_name)

    vasprun = folder_path + '/vasprun.xml'
    for atoms in ase.io.read(vasprun, format='vasp-xml', index=':'):
        R.append(atoms.get_positions())
        F.append(atoms.get_forces())
        E.append(atoms.get_potential_energy())
        z.append(atoms.get_atomic_numbers())
        pbc.append(atoms.get_pbc())
        unit_cell.append(atoms.get_cell())
        node_mask.append([True for _ in range(len(R[-1]))])

    if not mix:
        R, F, E, z, pbc, unit_cell, node_mask = np.array(R), np.array(F), np.array(E), np.array(z), np.array(
            pbc), np.array(unit_cell), np.array(node_mask)

        np.savez(folder_name, R=R, F=F, E=E, z=z, pbc=pbc, unit_cell=unit_cell, node_mask=node_mask)

if mix:
    R, F, E, z, pbc, unit_cell, node_mask = np.array(R), np.array(F), np.array(E), np.array(z), np.array(pbc), np.array(
        unit_cell), np.array(node_mask)
    np.savez('all_data_2_3_4_MD', R=R, F=F, E=E, z=z, pbc=pbc, unit_cell=unit_cell, node_mask=node_mask)
