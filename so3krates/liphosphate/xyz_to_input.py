from ase.io.extxyz import read_extxyz
import numpy as np

R, F, E, z, pbc, unit_cell, idx_i, idx_j, node_mask = [[] for _ in range(9)]

counter = 1
for atoms in read_extxyz(open('./li3po4-joint-together.xyz','r'),index=slice(0,50000,1)):
    R.append(atoms.get_positions())
    F.append(atoms.get_forces())
    E.append(atoms.get_potential_energy())
    z.append(atoms.get_atomic_numbers())
    pbc.append(atoms.get_pbc())
    unit_cell.append(atoms.get_cell())
    node_mask.append([True for _ in range(len(R[-1]))])
    print("Finished frame No. "+str(counter)+'  \t Energy:'+str(E[-1])+' eV')

    counter+=1

print("Total number of valid data: "+str(len(R)))
R, F, E, z, pbc, unit_cell, node_mask = np.array(R), np.array(F), np.array(E), np.array(z), np.array(pbc), np.array(unit_cell), np.array(node_mask)

np.savez('li-phosphate-MD-data', R=R, F=F, E=E, z=z, pbc=pbc, unit_cell=unit_cell, node_mask=node_mask)