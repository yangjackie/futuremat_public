"""
Module contains codes that set up bulk perovskite (space group $Pm\bar{3}m$) structures
given the chemical composition of ABC3. The module utilize the crystal builder from ASE
package and write a POSCAR for each crystal structure in a seperate folder. There are two
part of the procedure:

(1) In the first step, structures generated in the $Pm\bar{3}m$ space group will be optimized
    utilising the function .geometry_optimisation.default_symmetry_preserving_optimisation().
    This can be done by submitting the default_symmetry_preserving_optimisation task to the PBS
    queue with the myqueue command line option.

(2) In the second step, optimized structure for each perovskite will be read in from individual folder, and ten randomly
    distorted structures will be generated from this optimised structures. Then .geometry_optimisation.default_structural_optimisation()
    should be called to perform full optimisations to drive each distorted structure to a new nearby local minima.

"""

import os
import random
from ase.spacegroup import crystal as ase_crystal
from ase import Atom
from ase.db import connect
from core.models.lattice import Lattice
from core.resources.crystallographic_space_groups import CrystallographicSpaceGroups

from core.models.crystal import Crystal

from core.internal.builders.crystal import map_ase_atoms_to_crystal, build_supercell
from core.dao.vasp import VaspWriter, VaspReader

# define the chemistry of the perovskites that we are interested in

A_site_list = [['Li', 'Na', 'K', 'Rb', 'Cs'], ['Li', 'Na', 'K', 'Rb', 'Cs'], ['Mg', 'Ca', 'Sr', 'Ba'],
               ['Li', 'Na', 'K', 'Rb', 'Cs']]
B_site_list = [['Pb', 'Sn', 'Ge'], ['V', 'Ta', 'Nb'], ['Ti', 'Zr'], ['V', 'Ta', 'Nb']]
C_site_list = [['F', 'Cl', 'Br', 'I'], ['F', 'Cl', 'Br', 'I'], ['O', 'S', 'Se', 'Te'], ['O', 'S', 'Se', 'Te']]


# give a random initial unit cell length
a = 5.0


def make_starting_bulk_strutures():
    cwd = os.getcwd()
    for i in range(len(A_site_list)):
        for j in range(len(A_site_list[i])):
            for k in range(len(B_site_list[i])):
                for l in range(len(C_site_list[i])):
                    atoms = ase_crystal([A_site_list[i][j], B_site_list[i][k], C_site_list[i][l]],
                                        basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
                                        spacegroup=221,
                                        cellpar=[a, a, a, 90, 90, 90],
                                        size=(1, 1, 1))
                    atoms = map_ase_atoms_to_crystal(atoms)

                    wd = cwd + '/relax_Pm3m/' + A_site_list[i][j] + B_site_list[i][k] + C_site_list[i][l] + '_Pm3m' + '/'
                    if not os.path.exists(wd):
                        os.makedirs(wd)

                    os.chdir(wd)
                    vasp_writer = VaspWriter()
                    vasp_writer.write_structure(atoms)
                    os.chdir(cwd)


def make_distorted_structures_from_optimised():
    cwd = os.getcwd()
    for i in range(len(A_site_list)):
        for j in range(len(A_site_list[i])):
            for k in range(len(B_site_list[i])):
                for l in range(len(C_site_list[i])):
                    for c in range(10): #make 10 randomly distorted crystal structures

                        #TODO - ideally this should be get from a database!
                        crystal = VaspReader(
                            input_location=cwd + '/' + A_site_list[i][j] + B_site_list[i][k] + C_site_list[i][
                                l] + '_Pm3m/CONTCAR').read_POSCAR()

                        crystal = randomise_crystal(crystal)

                        wd = cwd + '/relax_randomized/' + A_site_list[i][j] + B_site_list[i][k] + C_site_list[i][
                            l] + '_full_relax_rand_' + str(c) + '/'
                        if not os.path.exists(wd):
                            os.makedirs(wd)

                        os.chdir(wd)
                        vasp_writer = VaspWriter()
                        vasp_writer.write_structure(crystal)
                        os.chdir(cwd)


def randomise_crystal(crystal):
    a = crystal.lattice.a * (1 + random.randrange(-100, 100) / 900)
    b = crystal.lattice.b * (1 + random.randrange(-100, 100) / 900)
    c = crystal.lattice.c * (1 + random.randrange(-100, 100) / 900)
    alpha = crystal.lattice.alpha * (1 + random.randrange(-100, 100) / 900)
    beta = crystal.lattice.beta * (1 + random.randrange(-100, 100) / 900)
    gamma = crystal.lattice.gamma * (1 + random.randrange(-100, 100) / 900)
    lattice = Lattice(a,b,c,alpha,beta,gamma)
    asymmetric_unit = crystal.asymmetric_unit
    for _i in range(len(crystal.asymmetric_unit)):
        for _j in range(len(crystal.asymmetric_unit[_i].atoms)):
            asymmetric_unit[_i].atoms[_j].scaled_position.x = crystal.asymmetric_unit[_i].atoms[
                                                                          _j].scaled_position.x * (
                                                                              1 + random.randrange(
                                                                          -100, 100) / 900)
            asymmetric_unit[_i].atoms[_j].scaled_position.y = crystal.asymmetric_unit[_i].atoms[
                                                                          _j].scaled_position.y * (
                                                                              1 + random.randrange(
                                                                          -100, 100) / 900)
            asymmetric_unit[_i].atoms[_j].scaled_position.z = crystal.asymmetric_unit[_i].atoms[
                                                                          _j].scaled_position.z * (
                                                                              1 + random.randrange(
                                                                          -100, 100) / 900)
    new_crystal = Crystal(lattice,asymmetric_unit,space_group=CrystallographicSpaceGroups.get(1))
    return new_crystal


def prepare_property_calculation_folder(db=None,property='phonon',supercell=[1,1,1]):
    cwd = os.getcwd()
    db = connect(cwd + '/2dpv.db')
    print("Successfully connected to the database")
    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:
                    system_name = a + b + c
                    uid = system_name + '3_pm3m'
                    print(uid)
                    row = db.get(selection=[('uid', '=', uid)])
                    crystal = row.toatoms()
                    crystal = map_ase_atoms_to_crystal(crystal)
                    if property is 'phonon':
                        crystal = build_supercell(crystal, expansion=supercell)
                        wd = cwd + '/relax_Pm3m/' + system_name + '_Pm3m' + '/phonon_G/'
                    if not os.path.exists(wd):
                        os.makedirs(wd)
                    os.chdir(wd)
                    vasp_writer = VaspWriter()
                    vasp_writer.write_structure(crystal)
                    os.chdir(cwd)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='bulk library set up routine',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--perovskite", action='store_true')
    parser.add_argument("--distort", action='store_true')
    parser.add_argument("--db", type=str, default=os.getcwd() + '/2dpv.db',
                        help="Name of the database that contains the results of the screenings.")
    parser.add_argument("--setup_gamma_phonon", action='store_true',
                        help="Setup calculation folders for Gamma point phonon")
    parser.add_argument("--randomize", action='store_true',
                        help="Setup a randomized crystal")
    args = parser.parse_args()

    if args.perovskite:
        make_starting_bulk_strutures()
    if args.distort:
        make_distorted_structures_from_optimised()
    if args.setup_gamma_phonon:
        prepare_property_calculation_folder(db=args.db,property='phonon')
    if args.randomize:
        reader = VaspReader(input_location='./CONTCAR_opt')
        crystal = reader.read_POSCAR()
        crystal = randomise_crystal(crystal)
        VaspWriter().write_structure(crystal,'POSCAR')


