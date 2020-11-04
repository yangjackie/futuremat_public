import warnings
import logging
import math
import numpy as np

from pymatgen.core.structure import IStructure as pymatstructure
import settings

from core.internal.builders.crystal import map_pymatgen_IStructure_to_crystal, expand_to_P1_strucutre
from core.models.element import pbe_pp_choices
from core.models import *
from core.dao.abstract_io import *
from core.models.lattice import Lattice

logger = logging.getLogger("futuremat.core.dao.vasp")

default_ionic_optimisation_set = {
    'SYSTEM': 'futuremat',
    'PREC': 'Normal',
    'EDIFF': 1E-5,
    'EDIFFG': -0.01,
    'IBRION': 2,
    'POTIM': 0.1,
    'ISIF': 0,
    'NSW': 800,
    'IALGO': 38,
    'IALGO': 38,
    'ISYM': 0,
    'ADDGRID': '.TRUE.',
    'ISMEAR': 0,
    'SIGMA': 0.05,
    'LREAL': 'AUTO',
    'LWAVE': '.FALSE.',
    'LCHARG': '.FALSE.',
    'LVTOT': '.FALSE.',
    'LMAXMIX': 6,
    'AMIN': 0.01
}

default_static_calculation_set = {
    'SYSTEM': 'futuremat',
    'PREC': 'Normal',
    'INIWAV': 1,
    'ENCUT': 600,
    'EDIFF': 1E-5,
    'EDIFFG': -0.01,
    'IBRION': -1,
    'ISIF': 0,
    'NSW': 0,
    'IALGO': 38,
    'ISYM': 0,
    'ADDGRID': '.TRUE.',
    'ISMEAR': 0,
    'SIGMA': 0.05,
    'LREAL': 'AUTO',
    'LWAVE': '.FALSE.',
    'LCHARG': '.TRUE.',
    'LMAXMIX': 6,
    'NPAR': 28,
    'LORBIT': 11,
    'NELM': 500,
    'AMIN': 0.01,
    'LVTOT': '.TRUE.',
    'ISPIN': 2
}

"""
Default set of keywords for switching between different density functionals, adopted from
https://github.com/jkitchin/vasp/blob/master/vasp/vasp_core.py
"""
xc_defaults = {'lda': {'pp': 'LDA'},
               # GGAs
               'gga': {'pp': 'GGA'},
               'pbe': {'pp': 'PBE'},
               'revpbe': {'pp': 'LDA', 'gga': 'RE'},
               'rpbe': {'pp': 'LDA', 'gga': 'RP'},
               'am05': {'pp': 'LDA', 'gga': 'AM'},
               'pbesol': {'pp': 'LDA', 'gga': 'PS'},
               # Meta-GGAs
               'tpss': {'pp': 'PBE', 'metagga': 'TPSS'},
               'revtpss': {'pp': 'PBE', 'metagga': 'RTPSS'},
               'm06l': {'pp': 'PBE', 'metagga': 'M06L'},
               # vdW-DFs
               'optpbe-vdw': {'pp': 'LDA', 'gga': 'OR', 'luse_vdw': True,
                              'aggac': 0.0},
               'optb88-vdw': {'pp': 'LDA', 'gga': 'BO', 'luse_vdw': True,
                              'aggac': 0.0, 'param1': 1.1 / 6.0,
                              'param2': 0.22},
               'optb86b-vdw': {'pp': 'LDA', 'gga': 'MK', 'luse_vdw': True,
                               'aggac': 0.0, 'param1': 0.1234,
                               'param2': 1.0},
               'vdw-df2': {'pp': 'LDA', 'gga': 'ML', 'luse_vdw': True,
                           'aggac': 0.0, 'zab_vdw': -1.8867},
               'beef-vdw': {'pp': 'PBE', 'gga': 'BF', 'luse_vdw': True,
                            'zab_vdw': -1.8867, 'lbeefens': True},
               # hybrids
               'pbe0': {'pp': 'LDA', 'gga': 'PE', 'lhfcalc': True},
               'hse03': {'pp': 'LDA', 'gga': 'PE', 'lhfcalc': True,
                         'hfscreen': 0.3},
               'hse06': {'pp': 'LDA', 'gga': 'PE', 'lhfcalc': True,
                         'hfscreen': 0.2},
               'b3lyp': {'pp': 'LDA', 'gga': 'B3', 'lhfcalc': True,
                         'aexx': 0.2, 'aggax': 0.72,
                         'aggac': 0.81, 'aldac': 0.19},
               'hf': {'pp': 'PBE', 'lhfcalc': True, 'aexx': 1.0,
                      'aldac': 0.0, 'aggac': 0.0}}


class VaspReader(FileReader):
    def __init__(self, input_location=None, file_content=None):
        super(self.__class__, self).__init__(input_location=input_location,
                                             file_content=file_content)

    def read_INCAR(self):
        incar = {}
        for line in self.file_content:
            splitted = line.split('=')
            incar[splitted[0]] = splitted[1]
        return incar

    def get_free_energies(self):
        if 'OSZICAR' in self.input_location:
            return self.get_free_energies_from_oszicar()
        elif 'OUTCAR' in self.input_location:
            return self.get_free_energies_from_outcar()

    def get_rpa_frequency_dependent_dielectric_tensor_from_outcar(self):
        import cmath
        dielectric_tensor = {}
        start_read = None
        for line_no, line in enumerate(self.file_content):
            if ('INVERSE MACROSCOPIC DIELECTRIC TENSOR (including local field effects in RPA (Hartree))' in line) or (
                    'MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects in DFT)' in line):
                start_read = line_no

            if (start_read is not None):
                if 'w=' in line:
                    energy = float(line.split()[1])
                    _xrow = [float(i) for i in self.file_content[line_no + 1].split()]
                    _yrow = [float(i) for i in self.file_content[line_no + 2].split()]
                    _zrow = [float(i) for i in self.file_content[line_no + 3].split()]
                    dielectric_tensor[energy] = {'xx': complex(_xrow[0], _xrow[1]),
                                                 'xy': complex(_xrow[2], _xrow[3]),
                                                 'xz': complex(_xrow[4], _xrow[5]),
                                                 'yy': complex(_yrow[2], _yrow[3]),
                                                 'yz': complex(_yrow[4], _yrow[5]),
                                                 'zz': complex(_zrow[4], _zrow[5])}
                if 'cpu time' in line:
                    start_read = None
        return dielectric_tensor

    def get_free_energies_from_oszicar(self):
        """
        Given the location of the OSZICAR file, read all the free energies from each optimisation step.

        :return list free_energies: A list containing the free energies for all the geometry optimisation steps
            completed thus far.
        """
        free_energies = []
        for line in self.file_content:
            if 'd E' in line:
                free_energies.append(float(line.split()[2]))
        return free_energies

    def get_free_energies_from_outcar(self):
        """
        Given the location of the OUTCAR file, read all the free energies from each optimisation step.

        :return list free_energies: A list containing the free energies for all the geometry optimisation steps
            completed thus far.
        """
        free_energies = []
        for line in self.file_content:
            if 'free  energy   TOTEN' in line:
                free_energies.append(float(line.split()[4]))
        return free_energies

    def get_vibrational_eigenfrequencies_from_outcar(self):
        """
        Given the OUTCAR from a phonon calculation, read in all the phonon frequencies.

        :return list phonon_frequencies: A list of all the Gamma point phonon frequencies, given in the unit of THz.
        """
        phonon_frequencies = []
        for line in self.file_content:
            if ('f' in line) and ('=' in line) and ('THz' in line) and ('cm-1' in line) and ('meV' in line):
                if 'f/i' not in line:
                    phonon_frequencies.append(float(line.split()[3]))
                else:
                    freq = float(line.split()[2])
                    phonon_frequencies.append(complex(0, freq))  # imaginary frequencies, instabilities!
        return phonon_frequencies

    def read_XDATCAR(self):
        _counter = 0
        _frame_counter = 0
        _lvs = []
        _atoms = []
        all_frames = []
        _atom_counter = 0
        for line in self.file_content:

            if _counter in [2, 3, 4]:  # read in values of the lattice vectors from the second to forth line of the file
                _lvs.append(cVector3D(float(line.split()[0]), float(line.split()[1]), float(line.split()[2])))

            if _counter == 4:
                lattice_vectors = cMatrix3D(_lvs[0], _lvs[1], _lvs[2])

            if _counter == 5:
                _atomic_labels = line.split()

            if _counter == 6:
                _num_species = [int(x) for x in line.split()]

                _atomic_labels = [_num_species[i] * _atomic_labels[i].split() for i in range(len(_num_species))]
                _atomic_labels = [item for sublist in _atomic_labels for item in sublist]

            if _counter > 7:
                if 'Direct configuration' not in line:
                    if _counter == len(self.file_content) - 1:
                        break
                    # read in the atomic coordinates
                    coord = cVector3D(float(line.split()[0]), float(line.split()[1]), float(line.split()[2]))
                    _atoms.append(Atom(label=_atomic_labels[_atom_counter], scaled_position=coord))
                    _atom_counter += 1
                else:
                    from core.models.lattice import Lattice
                    crystal = Crystal(lattice=Lattice.from_lattice_vectors(lattice_vectors),
                                      asymmetric_unit=[Molecule(atoms=_atoms)],
                                      space_group=CrystallographicSpaceGroups.get(1))
                    all_frames.append(crystal)
                    _frame_counter += 1
                    _atom_counter = 0
                    _atoms = []
            _counter += 1
        print("Total number of frames " + str(len(all_frames)))
        return all_frames

    def read_POSCAR(self):
        """
        Method to read in a crystal structure from VASP POSCAR or CONTCAR file. The crystal being read in will
        be in the P1 settings.

        :return: A crystal structure object
        """
        _counter = 0
        _lvs = []
        _atoms = []
        for line in self.file_content:

            if _counter == 1:
                scaling_factor = float(line.split()[0])

            if _counter in [2, 3, 4]:  # read in values of the lattice vectors from the second to forth line of the file
                _lh = [float(l) * scaling_factor for l in line.split()]
                _lvs.append(cVector3D(_lh[0], _lh[1], _lh[2]))

            if _counter == 4:
                lattice_vectors = cMatrix3D(_lvs[0], _lvs[1], _lvs[2])

            if _counter == 5:
                _atomic_labels = line.split()

            if _counter == 6:
                _num_species = [int(x) for x in line.split()]

                _atomic_labels = [_num_species[i] * _atomic_labels[i].split() for i in range(len(_num_species))]
                _atomic_labels = [item for sublist in _atomic_labels for item in sublist]

            if _counter == 7:
                _coord_type = line

            if _counter > 7:
                if _counter <= sum(_num_species) + 7:
                    # read in the atomic coordinates
                    coord = cVector3D(float(line.split()[0]), float(line.split()[1]), float(line.split()[2]))
                    if 'cartesian' in _coord_type.lower():
                        _atoms.append(Atom(label=_atomic_labels[_counter - 8], position=coord))
                    elif 'direct' in _coord_type.lower():
                        _atoms.append(Atom(label=_atomic_labels[_counter - 8], scaled_position=coord))
            _counter += 1

        from core.models.lattice import Lattice
        lattice = Lattice.from_lattice_vectors(lattice_vectors)
        lattice.lattice_vectors = lattice_vectors

        # logger.warning("===                                WARNING                                           ===")
        # logger.warning("=== Setting Lattice Vector as originally provided in the POSCAR to prevent rotation! ===")
        # logger.warning("=== Check this is what you wanted and it gives back exactly the same thing as wanted!===")

        crystal = Crystal(lattice=lattice,
                          asymmetric_unit=[Molecule(atoms=_atoms)],
                          space_group=CrystallographicSpaceGroups.get(1))  # all vasp input assumes P1

        return crystal

    def read_CHGCAR(self):
        # this routine is adapted from MacroDensity Package with modifications
        crystal = self.read_POSCAR()
        _counter = 0
        _chg_start = len(crystal.all_atoms()) + 9

        # read all the values of the densities into a plane list
        potential = []
        for line in self.file_content:
            if _counter > len(crystal.all_atoms()) + 7:
                if _counter == _chg_start:
                    NGX, NGY, NGZ = [int(x) for x in line.split()]
                elif (_counter > _chg_start) and (_counter <= _chg_start + int(math.ceil(NGX * NGY * NGZ / 5))):
                    for _l in line.split():
                        potential.append(float(_l))
            _counter += 1
        # put this into a matrix form
        l = 0
        potential_grid = np.zeros(shape=(NGX, NGY, NGZ))
        # if charge:
        #    point_volume = 1.0 / (NGX * NGY * NGZ)
        # else:
        #    point_volume = 1
        for k in range(NGZ):
            for j in range(NGY):
                for i in range(NGX):
                    potential_grid[i, j, k] = potential[l]  # * point_volume
                    l += 1
        return potential_grid, crystal




class VaspWriter(object):

    def write_INCAR(self, filename='INCAR',
                    default_options=default_ionic_optimisation_set,
                    **kwargs):
        default_options.update(kwargs)

        incar = open(filename, 'w')
        try:
            incar.write("SYSTEM=" + str(default_options['SYSTEM']) + '\n')
        except KeyError:
            incar.write("SYSTEM=furturemat\n")

        keylist = default_options.keys()
        keylist = list(sorted(keylist))

        for key in keylist:
            if str(default_options[key]).lower() == 'true':
                default_options[key] = '.true.'
            elif str(default_options[key]).lower() == 'false':
                default_options[key] = '.false.'
            if key not in ['frame', 'kwargs', 'self', 'SYSTEM', 'filename']:
                incar.write(str(key).upper() + ' = ' + str(default_options[key]).upper() + '\n')
        incar.close()

    def write_potcar(self, crystal, filename='POTCAR', sort=True, unique=True, magnetic=False, use_GW=False):
        if settings.functional is not None:
            if settings.functional.lower() == 'pbe':
                pass
            else:
                raise NotImplementedError("Current implementation will only concatenate PAW pseudopotential for PBE!")
        else:
            raise Exception("Please specify in the default configuration where to find VASP PAW pseudopotential files!")

        if isinstance(crystal, pymatstructure):
            crystal = map_pymatgen_IStructure_to_crystal(crystal)

        if not magnetic:
            _all_atoms = crystal.all_atoms(unique=unique)
            _all_atom_label = [i.clean_label for i in _all_atoms]
            all_atom_label = []

            if unique:
                for l in _all_atom_label:
                    if l not in all_atom_label:
                        all_atom_label.append(l)
            else:
                all_atom_label = _all_atom_label
        else:
            all_atom_label = [k[0] for k, v in crystal.mag_group]

        if not use_GW:
            potcars = [settings.vasp_pp_directory + '/' + pbe_pp_choices[e] + '/POTCAR' for e in
                       all_atom_label]
        else:
            logger.info("Using GW version of the pseudopotential requested!")
            potcars = []
            import os
            for e in all_atom_label:
                if os.path.exists(settings.vasp_pp_directory + '/' + pbe_pp_choices[e] + '_GW/'):
                    potcars.append(settings.vasp_pp_directory + '/' + pbe_pp_choices[e] + '_GW/POTCAR')
                else:
                    potcars.append(settings.vasp_pp_directory + '/' + pbe_pp_choices[e] + '/POTCAR')

        with open(filename, 'w') as outfile:
            for fn in potcars:
                logger.info("Getting pseudopotential " + fn)
                with open(fn) as infile:
                    for line in infile:
                        if ('Zr' in fn) and ('VRHFIN' in line): line = '   VRHFIN =Zr: 4s4p5s4d\n'
                        outfile.write(line)

    def write_structure(self, crystal, filename='POSCAR', direct=False, sort=True, magnetic=False):
        """Method to write VASP position (POSCAR/CONTCAR) files.

        Adopted from ASE with modifications
        """
        if crystal.space_group.index != 1:
            crystal = expand_to_P1_strucutre(crystal)

        if isinstance(filename, str):
            f = open(filename, 'w')
        else:  # Assume it's a 'file-like object'
            f = filename

        # Write atom positions in scaled or cartesian coordinatesq

        # Get all atoms and corresponding symbols
        all_unqiue_labels = None
        if not magnetic:
            all_atoms = crystal.all_atoms(sort=sort)
            all_atom_label = [i.clean_label for i in all_atoms]
            # all_unqiue_labels = list(set([i.clean_label for i in all_atoms]))
            if sort:
                all_unqiue_labels = list(sorted(set(all_atom_label), key=all_atom_label.index))
            else:
                all_unqiue_labels = all_atom_label

            # Create a list sc of (symbol, count) pairs
            label_count = [0 for _ in all_unqiue_labels]
            for i, label in enumerate(all_unqiue_labels):
                for atom in all_atoms:
                    if atom.clean_label == label:
                        label_count[i] += 1

            if not sort:
                label_count = [1 for _ in all_atom_label]
        else:
            all_unqiue_labels = []
            label_count = []
            all_atoms = []
            for k, v in crystal.mag_group:
                _v = list(v)
                all_unqiue_labels.append(k[0])
                label_count.append(len(_v))
                for a in _v:
                    all_atoms.append(a)

        # Create the label
        label = 'Atomistic Systems created by FUTUREMAT'

        # for i, l in enumerate(all_unqiue_labels):
        #    label += l + '_' + str(label_count[i]) + '_'
        f.write(label + '\n')

        # Write unitcell in real coordinates and adapt to VASP convention
        # for unit cell
        # ase Atoms doesn't store the lattice constant separately, so always
        # write 1.0.
        f.write('%19.16f\n' % 1.0)

        latt_form = ' %21.16f'
        for row in range(3):
            f.write(' ')
            for column in range(3):
                f.write(latt_form % crystal.lattice.lattice_vectors.get(row, column))
            f.write('\n')

        # write out the atomic symbol and count for each atom
        for l in all_unqiue_labels:
            f.write('%5s' % l)
        f.write('\n')

        for count in label_count:
            f.write('%5i' % count)
        f.write('\n')

        if direct:
            f.write('Direct\n')
        else:
            f.write('Cartesian\n')

        # print 'In vasp writer, how many atoms '+str(len(all_atoms))

        for iatom, atom in enumerate(all_atoms):
            for i in range(3):
                if direct:
                    f.write(' %19.16f' % atom.scaled_position[i])
                else:
                    f.write(' %19.16f' % atom.position[i])
            f.write('\n')

        if type(filename) == str:
            f.close()

    def write_KPOINTS(self, crystal, filename='KPOINTS', K_points=None, grid=0.025, molecular=False,
                      gamma_centered=False, kpoint_str=None):
        kpoint_file = open(filename, 'w')

        if kpoint_str is None:
            kpoint_file.write("KPOINTS created by Entdecker Program\n")
            kpoint_file.write('0\n')
            if not gamma_centered:
                kpoint_file.write('Monkhorst-Pack\n')
            else:
                kpoint_file.write('Gamma\n')

            if K_points is not None:
                kpoint_file.write(str(K_points[0]) + ' ' + str(K_points[1]) + ' ' + str(K_points[2]) + '\n')
            else:
                from core.utils.kpoints import kpoints_from_grid
                kpoints = kpoints_from_grid(crystal, grid=grid, molecular=molecular)
                if [int(k) for k in kpoints] == [1, 1, 1]:
                    crystal.gamma_only = True
                kpoint_file.write(str(int(kpoints[0])) + ' ' + str(int(kpoints[1])) + ' ' + str(int(kpoints[2])) + '\n')
            kpoint_file.write("0 0 0")
        else:
            kpoint_file.write(kpoint_str)

        # TODO - write K-POINT paths for bandstructure calculations

        kpoint_file.close()


def prepare_potcar(poscar_file,gw=False):
    crystal = VaspReader(input_location=poscar_file).read_POSCAR()
    VaspWriter().write_potcar(crystal, use_GW=gw)
    # write out the crystal again to get rid of the atom re-ordering problem?
    VaspWriter().write_structure(crystal)


def prepare_kpoints(poscar_file, MP_points, grid):
    crystal = VaspReader(input_location=poscar_file).read_POSCAR()
    VaspWriter().write_KPOINTS(crystal, K_points=MP_points, grid=grid, molecular=False)


def convert_xml_to_pickle():
    import pickle
    from pymatgen.io.vasp.outputs import Vasprun
    vasprun = Vasprun('./vasprun.xml')
    pickle.dump(vasprun.as_dict(), open('./vasprun.p', 'wb'))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='cmd utils for VASP IO',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--genpot", action='store_true')
    parser.add_argument("--gen_mp_k", action='store_true')
    parser.add_argument("--mp_points", type=str, default=None,
                        help='string of comma separated integer denoting number of K-points along each reciprocal space direction.')
    parser.add_argument("--mp_grid", type=float, default=0.025, help='grid spacing for generating MP Kpoints')
    parser.add_argument('--convert_xml', action='store_true')
    parser.add_argument('--gw', action='store_true')
    args = parser.parse_args()

    if args.genpot:
        prepare_potcar("./POSCAR", gw=args.gw)

    if args.gen_mp_k:
        prepare_kpoints("./POSCAR", MP_points=args.mp_points, grid=args.mp_grid)

    if args.convert_xml:
        convert_xml_to_pickle()
