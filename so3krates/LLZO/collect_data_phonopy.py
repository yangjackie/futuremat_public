"""
Script that collect the VASP calculation output into the .npx format needed as the input for the so3krates network.

The training data are prepared in the following way:
* For the base material, say LLZO, a phonopy calculation is first performed to obtain its force constant.
* Then at a chosen temperature, use phonopy to generate up to N (e.g. 2000) randomly displaced structures, with the
    displacements chosen from the thermal distributions of the phonon DOS determined from the force costants.
* Each structure is then put through VASP to perform a single point calculation to determine the corresponding
    energies and atomic forces.
"""
import os
import subprocess
import os
import tarfile
import ase.io
import ase
import numpy as np
import shutil

# Define the directory containing the folders with disp_* names
root_dir = "/scratch/dy3/jy8620/LLZO/pure/"

# Iterate through the folders, in this case, each folder corresponding to results from sampling the thermally displaced
# structures at a chosen temperature
mix_all=False

if mix_all:
    R, F, E, z, pbc, unit_cell, idx_i, idx_j, node_mask = [[] for _ in range(9)]

for folder_name in list(sorted(os.listdir(root_dir))):
    if folder_name.startswith("disp_") and os.path.isdir(os.path.join(root_dir, folder_name)): #e.g. disp_200K
        # Construct the full path to the folder
        folder_path = os.path.join(root_dir, folder_name)

        # Find all .tar.gz files in the folder
        tar_files = [file_name for file_name in os.listdir(folder_path) if file_name.endswith(".tar.gz")]

        # These are the information that we need to gather, initialising an empty placeholder here.
        if not mix_all:
            R, F, E, z, pbc, unit_cell, idx_i, idx_j, node_mask = [[] for _ in range(9)]

        #looping through the dataset to collect result from individual structure
        for tar_count, tar_file in enumerate(tar_files):
            tar_file_path = os.path.join(folder_path, tar_file)
            print(f"{tar_count}/{len(tar_files)}, Processing {tar_file_path}...")

            _path = os.path.join(folder_path, tar_file[:-7])

            # Extract the gz file, each calculation folder is assumed to be tar-ed up after the calculation completes
            # in order to reduce the total number of files on the scratch dir.
            with tarfile.open(tar_file_path, 'r') as tar:
                tar.extractall(path=folder_path)

            # Read the OUTCAR file if it exists
            try:
                outcar_file = os.path.join(_path, 'OUTCAR')
                if os.path.exists(outcar_file):
                    atoms = ase.io.read(os.path.join(_path, 'OUTCAR'), format='vasp-out')
                    # Process the contents of OUTCAR as needed

                    all_forces = np.ndarray.flatten(atoms.get_forces())

                    if max(all_forces)<20:
                        R.append(atoms.get_positions())
                        F.append(atoms.get_forces())
                        E.append(atoms.get_potential_energy())
                        z.append(atoms.get_atomic_numbers())
                        pbc.append(atoms.get_pbc())
                        unit_cell.append(atoms.get_cell())

                        #not quite sure about what this mask mean, but this works for the moment.
                        node_mask.append([True for _ in range(len(R[-1]))])
                        print(f"Finished processing {tar_file_path}")
                    else:
                        print('skipp this, force too large!')
            except:
                print('Error parsing OUTCAR, skipped')

            # Clean up the extracted files if needed
            try:
                shutil.rmtree(os.path.join(_path))
            except:
                pass



        #everything needs to be in numpy array!
        if not mix_all:
            print("Total number of valid data: "+str(len(R)))
            R, F, E, z, pbc, unit_cell, node_mask = np.array(R), np.array(F), np.array(E), np.array(z), np.array(
                pbc), np.array(unit_cell), np.array(node_mask)

if mix_all:
    R, F, E, z, pbc, unit_cell, node_mask = np.array(R), np.array(F), np.array(E), np.array(z), np.array(pbc), np.array(unit_cell), np.array(node_mask)
    np.savez('all_data_2_3_4', R=R, F=F, E=E, z=z, pbc=pbc, unit_cell=unit_cell, node_mask=node_mask)
