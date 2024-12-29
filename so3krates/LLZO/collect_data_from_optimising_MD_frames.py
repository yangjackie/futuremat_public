import os
import zipfile
import glob
import ase
import ase.io
import numpy as np
import sys

pwd = os.getcwd()
all_zips = glob.glob("*.zip")

total_number_of_data = 0

R, F, E, z, pbc, unit_cell, idx_i, idx_j, node_mask = [[] for _ in range(9)]

temperatures=[]

for izip, zip in enumerate(list(sorted(all_zips))):
    with zipfile.ZipFile(zip) as zf:
        with open("./vasprun_temp.xml", 'wb') as f:
            f.write(zf.read(zip.replace('.zip','')+"/vasprun.xml"))
    f.close()

    #R, F, E, z, pbc, unit_cell, idx_i, idx_j, node_mask = [[] for _ in range(9)]
    atoms=ase.io.read("./vasprun_temp.xml", format='vasp-xml', index=0)
    #for atoms in out:
    R.append(atoms.get_positions())
    #sys.stdout.write(str(R[-1]))
    F.append(atoms.get_forces())
    E.append(atoms.get_potential_energy())
    z.append(atoms.get_atomic_numbers())
    pbc.append(atoms.get_pbc())
    unit_cell.append(atoms.get_cell())
    node_mask.append([True for _ in range(len(R[-1]))])

    sys.stdout.write("Structure: "+str(izip+1)+"/"+str(len(all_zips))+" Cumulative total number of data points: "+str(len(R))+' '+'energy: '+str(E[-1])+'\r')
    if izip!=len(all_zips)-1:
        sys.stdout.flush()
    #total_number_of_data += len(R)

    os.remove("./vasprun_temp.xml")

#filename=zip.replace('.zip','')+'.npz'
filename='opt_data_cLLZO_12_44_minus_ten_frame.npz'
np.savez(filename, R=R, F=F, E=E, z=z, pbc=pbc, unit_cell=unit_cell, node_mask=node_mask)

