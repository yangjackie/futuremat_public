"""
The 2D library module, which builds slab models of two-dimensional perovskite structures with
different (a) orientations, (b) thicknesses and (c) surface terminations. Structures that
are built will be written to a directory with the corresponding set ups for VASP calculations.
"""

from ase.io import read, write
from ase.build import cut, stack, mx2, add_vacuum, rotate
from ase.io.vasp import write_vasp
from ase.db import connect

from core.dao.vasp import *
from core.internal.builders.crystal import *
import os

from twodPV.elements import A_site_list, B_site_list, C_site_list

#thicknesses = [3, 5, 7, 9]
thicknesses = [3]
orientation_dict = {'100': {'a': (1, 1, 0), 'b': (-1, 1, 0),
                            'origio': {'AO': (0.25, 0.25, 0), 'BO2': (0, 0, 0.25)}},
                    '111': {'a': (1, 1, 0), 'b': (-1, 0, 1),
                            'origio': {'AO3': (0, 0, 0), 'B': (0, 0, 0.25)}},
                    '110': {'a': (1, 1, 0), 'b': (0, 0, 1),
                            'origio': {'O2': (0.05, 0, 0), 'ABO': (0, 0, 0)}}}


def setup_two_d_structure_folders(orientation=None, termination=None, db=None):
    cwd = os.getcwd()
    print(cwd)
    A_site_list=[['Sr']]
    B_site_list=[['Ti']]
    C_site_list=[['O']]
    thicknesses = [11,13,15,17,19,21]

    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:
                    system_name = a + b + c
                    uid = system_name + '3_pm3m'
                    print(uid)
                    row = db.get(selection=[('uid', '=', uid)])
                    crystal = row.toatoms()

                    for thick in thicknesses:
                        print(a, b, c, orientation_dict[orientation]['a'], orientation_dict[orientation]['b'], thick)
                        slab = cut(crystal,
                                   a=orientation_dict[orientation]['a'],
                                   b=orientation_dict[orientation]['b'],
                                   nlayers=thick,
                                   origo=orientation_dict[orientation]['origio'][termination])
                        add_vacuum(slab, 80)

                        slab_wd = cwd + '/slab_' + str(orientation) + '_' + str(
                            termination) + '_small/' + a + b + c + "_" + str(thick) + '/'

                        if not os.path.exists(slab_wd):
                            os.makedirs(slab_wd)

                        os.chdir(slab_wd)
                        if orientation in ['110', '111']:
                            rotate(slab, slab.cell[0], [1, 0, 0], slab.cell[1], [0, 1, 0])

                        write_vasp('POSCAR', slab, vasp5=True, sort=True)
                        os.chdir(cwd)


def prepare_property_calculation_folder(db=None, property='phonon', supercell=[1, 1, 1], orientation=None,
                                        termination=None):
    cwd = os.getcwd()
    db = connect(cwd + '/2dpv.db')
    print("Successfully connected to the database")
    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:
                    system_name = a + b + c
                    for thick in thicknesses:
                        uid = system_name + '3_' + str(orientation) + "_" + str(termination) + "_" + str(thick)

                        row = db.get(selection=[('uid', '=', uid)])

                        crystal = row.toatoms()
                        crystal = map_ase_atoms_to_crystal(crystal)
                        base_dir = cwd + '/slab_' + str(orientation) + '_' + str(termination) + '_small/'
                        work_dir = base_dir + system_name + "_" + str(thick)
                        print(work_dir)
                        if property is 'phonon':
                            if supercell != [1, 1, 1]:
                                crystal = build_supercell(crystal, expansion=supercell)
                            wd = work_dir + '/phonon_G/'

                            if not os.path.exists(wd):
                                os.makedirs(wd)

                            os.chdir(wd)
                            vasp_writer = VaspWriter()
                            vasp_writer.write_structure(crystal)
                            os.chdir(cwd)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='two-d library set up routine',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--orient", type=str, default='100',
                        help='Orientations of the two-d perovskite slabs')
    parser.add_argument("--termination", type=str, default='AO',
                        help='Surface termination type of the two-d slab')
    parser.add_argument("--setup_twod", action='store_true')
    parser.add_argument("--setup_gamma_phonon", action='store_true',
                        help="Setup calculation folders for Gamma point phonon")
    parser.add_argument("--db", type=str, default=os.getcwd() + '/2dpv.db',
                        help="Name of the database that contains the results of the screenings.")
    args = parser.parse_args()

    if os.path.exists(args.db):
        args.db = connect(args.db)
    else:
        raise Exception("Database " + args.db + " does not exists, cannot proceed!")

    if args.setup_twod:
        setup_two_d_structure_folders(orientation=args.orient, termination=args.termination, db=args.db)
    if args.setup_gamma_phonon:
        prepare_property_calculation_folder(db=args.db, property='phonon', orientation=args.orient,
                                            termination=args.termination)
