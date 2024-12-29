import argparse, shutil, os

from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.transformations.standard_transformations import PerturbStructureTransformation
from core.models import Crystal
from core.resources.crystallographic_space_groups import CrystallographicSpaceGroups
from core.utils.zipdir import ZipDir
from core.internal.builders.crystal import map_pymatgen_IStructure_to_crystal
from core.calculators.vasp import Vasp
from core.dao.vasp import VaspWriter
from core.models.lattice import Lattice
from ase.io.extxyz import read_extxyz
from ase.io import read
import random


def optimise_extended_xyz(extxyz: str = None,
                          gpu_run: bool = False):
    dirname = extxyz.replace('.xyz','')

    if os.path.exists(dirname + '.zip'):
        #this needs to be further implmented to do secondary minimisation
        return 0
    else:
        for c in read_extxyz(open(extxyz,'r')):
            crystal = c

    pwd = os.getcwd()
    if not os.path.exists('./'+dirname):
        os.mkdir('./'+dirname)

    os.chdir('./'+dirname)

    crystal = map_pymatgen_IStructure_to_crystal(crystal)
    VaspWriter().write_structure(crystal,filename='POSCAR_orig')

    #this is a hack, using the api from Pymatgen. somehow my own version didnt work anymore. problem in
    #converting fractional coordinates when structure is mapped from pymatgen IStructure
    #need to randomly displace the atoms to get some more unique optimisation trajectories
    crystal = Structure.from_file('POSCAR_orig')
    transformer = PerturbStructureTransformation(min_distance=0.01, distance=0.025)
    randomised_crystal = transformer.apply_transformation(crystal)
    randomised_crystal.to_file('POSCAR_rand')
    randomised_crystal = read('POSCAR_rand')
    randomised_crystal = map_pymatgen_IStructure_to_crystal(randomised_crystal)
    randomised_crystal.gamma_only = True  # DO NOT DELETE THIS!!!


    optimisation_set = {'ISPIN': 1, 'PREC': "Accurate", 'IALGO': 38, 'NCORE': 32, 'ENCUT': 500, 'IBRION': 1, 'ISIF': 0, 'NSW': 400,
                     'LCHARG': True, 'LWAVE': True, 'use_gw': True, 'Gamma_centered': True, 'MP_points': [1, 1, 1],
                     'clean_after_success': False, 'LREAL': 'False', 'executable': 'vasp_gam', 'gpu_run': gpu_run}
    vasp = Vasp(**optimisation_set)
    vasp.set_crystal(randomised_crystal)
    vasp.execute()

    files = ['CHG', 'CHGCAR', 'LOCPOT', 'EIGENVAL', 'IBZKPT', 'PCDAT', 'POTCAR', 'WAVECAR', 'DOSCAR', 'OUTCAR',
             'PROCAR', 'KPOINTS']
    for f in files:
        try:
            os.remove(f)
        except OSError:
            pass

    os.chdir(pwd)
    ZipDir(dirname, dirname + '.zip')
    shutil.rmtree(dirname, ignore_errors=True)
    return 0

def optimise_frames_from_md_trajectory(part: int = 0,
                                       batch_size: int = 20,
                                       gpu_run: bool = False,
                                       gap: int = 10):
    # load all the production run xml file

    all_vasp_runs = []
    all_structures = []
    for v in ['vasprun.xml']:
        if os.path.exists('./' + v):
            all_vasp_runs.append(v)

    for v in all_vasp_runs:
        vasprun = Vasprun(v)
        trajectory = vasprun.get_trajectory()
        all_structures += [trajectory.get_structure(i) for i in range(len(trajectory.coords))]

    if (part + 1) * batch_size > len(all_structures):
        raise Exception("over the length of the trajectory, quit")

    for i in range(batch_size * part, batch_size * (part + 1), gap):
        pwd = os.getcwd()
        folder = 'frame_' + str(i)
        if os.path.exists(pwd + '/' + folder + '.zip'): continue
        try:
            os.mkdir(folder)
        except:
            pass
        os.chdir(pwd + '/' + folder)

        this_structure = map_pymatgen_IStructure_to_crystal(all_structures[i])
        this_structure.gamma_only = True  # DO NOT DELETE THIS!!!
        single_pt_set = {'ISPIN': 1, 'PREC': "Normal", 'IALGO': 38, 'NPAR': 32, 'ENCUT': 500, 'IBRION': 1, 'ISIF': 0,
                         'NSW': 300,
                         'LCHARG': True, 'LWAVE': True, 'use_gw': True, 'Gamma_centered': True, 'MP_points': [1, 1, 1],
                         'clean_after_success': False, 'LREAL': 'False', 'executable': 'vasp_gam', 'gpu_run': gpu_run}
        vasp = Vasp(**single_pt_set)
        vasp.set_crystal(this_structure)
        vasp.execute()

        files = ['CHG', 'CHGCAR', 'LOCPOT', 'EIGENVAL', 'IBZKPT', 'PCDAT', 'POTCAR', 'WAVECAR', 'DOSCAR', 'OUTCAR',
                 'PROCAR', 'KPOINTS']
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass

        os.chdir(pwd)
        ZipDir(folder, folder + '.zip')
        shutil.rmtree(folder, ignore_errors=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='workflow control for high-throughput DFT ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--opt", action='store_true', help='perform initial structural optimization')
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--gap", type=int, default=20)
    parser.add_argument("--gpu", action='store_true',
                        help='specify whether the calculation is to be run on the GPU node')
    parser.add_argument("--extxyz", type=str, default=None, help='Name of the xyz file containing the structure to be optimised')
    args = parser.parse_args()

    if args.extxyz:
        optimise_extended_xyz(extxyz=args.extxyz,gpu_run=args.gpu)
    else:
        optimise_frames_from_md_trajectory(part=args.part, batch_size=args.batch_size, gpu_run=args.gpu, gap=args.gap)
