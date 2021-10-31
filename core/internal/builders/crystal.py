from pymatgen import Composition
from pymatgen.alchemy.filters import RemoveDuplicatesFilter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure

from core.models import cMatrix3D
from core.models.lattice import Lattice as fLattice
from core.models.crystal import Crystal
from core.models.molecule import Molecule
from core.models.atom import Atom
from core.models.element import element_name_dict
from core.models.vector3d import cVector3D
from core.resources.crystallographic_space_groups import CrystallographicSpaceGroups

import copy
import numpy as np


def expand_to_P1_strucutre(crystal):
    expanded_mol_list = []

    for mol in crystal.asymmetric_unit:
        for op in crystal.space_group.full_symmetry:
            new_mol = Molecule()
            new_mol.atoms = [op.transform_atom(a) for a in mol.atoms]
            expanded_mol_list.append(new_mol)

    return Crystal(lattice=crystal.lattice,
                   asymmetric_unit=expanded_mol_list,
                   space_group=CrystallographicSpaceGroups.get(1))


def build_supercell(crystal, expansion=[1, 1, 1]):
    """
    Method to make a supercell using the input crystal as the primitive crystal structure. Given the
    transformation specified by a list of three integers `[n_{x},n_{y},n_{z}]`, a super cell with cell
    lengths `[n_{x}a,n_{y}b,n_{z}c]` will be built.

    :param crystal: Input crystal structure.
    :param expansion: A list of three integers specifying how big the supercell will be.
    :return: crystal: A fully constructed crystal object with new lattice parameters.
    :rtype: :class:`.Crystal`
    """
    crystal_in_p1_setting = expand_to_P1_strucutre(crystal)

    lattice = fLattice(a=crystal.lattice.a * expansion[0],
                       b=crystal.lattice.b * expansion[1],
                       c=crystal.lattice.c * expansion[2],
                       alpha=crystal.lattice.alpha,
                       beta=crystal.lattice.beta,
                       gamma=crystal.lattice.gamma)

    asymmetric_unit = [x for x in crystal_in_p1_setting.asymmetric_unit]

    for n_x in range(expansion[0]):
        for n_y in range(expansion[1]):
            for n_z in range(expansion[2]):

                tr_vec = crystal.lattice.lattice_vectors.get_row(0).vec_scale(n_x) + \
                         crystal.lattice.lattice_vectors.get_row(1).vec_scale(n_y) + \
                         crystal.lattice.lattice_vectors.get_row(2).vec_scale(n_z)

                if (n_x == 0) and (n_y == 0) and (n_z == 0):
                    pass
                else:
                    for mol in crystal_in_p1_setting.asymmetric_unit:
                        new_atoms = [Atom(label=atom.label, position=atom.position + tr_vec) for atom in mol.atoms]
                        asymmetric_unit.append(Molecule(atoms=new_atoms))

    return Crystal(lattice=lattice, asymmetric_unit=asymmetric_unit, space_group=CrystallographicSpaceGroups.get(1))


def map_pymatgen_IStructure_to_crystal(structure):
    """
    Given a Pymatgen IStructure object, map it to a crystal structure in our internal model

    :param atoms: An input Pymatgen IStructure object
    :return: a fully constructed crystal structure in P1 setting
    """

    lattice = fLattice(0, 0, 0, 0, 0, 0)
    lv = structure.lattice.__dict__['_matrix']

    lattice.lattice_vectors = cMatrix3D(cVector3D(lv[0][0], lv[0][1], lv[0][2]),
                                        cVector3D(lv[1][0], lv[1][1], lv[1][2]),
                                        cVector3D(lv[2][0], lv[2][1], lv[2][2]))
    return Crystal(lattice=lattice,
                   asymmetric_unit=[Molecule(atoms=[
                       Atom(label=element_name_dict[structure.atomic_numbers[i]], position=structure.cart_coords[i]) for
                       i in range(len(structure.atomic_numbers))])],
                   space_group=CrystallographicSpaceGroups.get(1))


def map_to_ase_atoms(crystal):
    """
    Given a crystal structure, map it to the ASE atoms model so we can use functionalities in ASE to
    manipulate things.

    :param crystal: Input crystal structure
    :return: A fully constructed ASE `atoms` model.
    """
    from ase.atoms import Atoms

    # Get all atoms and corresponding symbols
    crystal = expand_to_P1_strucutre(crystal)
    all_atoms = crystal.all_atoms()
    all_unqiue_labels = list(set([i.clean_label for i in all_atoms]))

    # Create a list sc of (symbol, count) pairs
    label_count = [0 for _ in all_unqiue_labels]
    for i, label in enumerate(all_unqiue_labels):
        for atom in all_atoms:
            if atom.clean_label == label:
                label_count[i] += 1

    symbol_line = ''
    for i in range(len(all_unqiue_labels)):
        symbol_line += all_unqiue_labels[i] + str(label_count[i])

    return Atoms(symbols=symbol_line,
                 scaled_positions=[a.scaled_position.to_numpy_array() for a in all_atoms],
                 pbc=True,
                 cell=[[crystal.lattice.lattice_vectors.get(m, n) for n in range(3)] for m in range(3)])


def map_ase_atoms_to_crystal(atoms):
    """
    Given an ASE atoms object, map it to a crystal structure in our internal model

    :param atoms: An input ASE atoms object
    :return: a fully constructed crystal structure in P1 setting
    """

    # this is dumb, but anyway, prevents re-orienting things when the lattice is constructed
    _lv = atoms.get_cell().array
    _a = cVector3D(*_lv[0])
    _b = cVector3D(*_lv[1])
    _c = cVector3D(*_lv[2])
    _lv = cMatrix3D(_a, _b, _c)
    lattice = fLattice.from_lattice_vectors(_lv)
    lattice.lattice_vectors = _lv

    return Crystal(lattice=lattice,
                   asymmetric_unit=[Molecule(atoms=[
                       Atom(label=atoms.get_chemical_symbols()[i],
                            scaled_position=cVector3D(*atoms.get_scaled_positions()[i])) for
                       i in range(len(atoms.get_chemical_symbols()))])],
                   space_group=CrystallographicSpaceGroups.get(1))


def map_to_pymatgen_Structure(crystal):
    """
    Given a crystal structure, map it to the pymatgen IStructure model so we can use functionalities in pymatgen to
    manipulate things.

    :param crystal: Input crystal structure
    :return: A fully constructed pymatgen `structure` model.
    """

    # Get all atoms and corresponding symbols
    crystal = expand_to_P1_strucutre(crystal)
    all_atoms = crystal.all_atoms()

    return Structure(lattice=[[crystal.lattice.lattice_vectors.get(m, n) for n in range(3)] for m in range(3)],
                     species=[a.label for a in all_atoms],
                     coords=[a.scaled_position.to_numpy_array() for a in all_atoms],
                     coords_are_cartesian=False)


class SubstitutionalSolidSolutionBuilder(object):

    def __init__(self, primitive_cell, supercell_size=[1, 1, 1], atom_to_substitute='H', atom_substitute_to='F',
                 number_of_substitutions=1, write_vasp=False, prefix=None, throttle=5, max_structures=None):
        self.primitive_cell = map_to_pymatgen_Structure(primitive_cell)
        self.primitive_cell.make_supercell(supercell_size)
        self.sc_size = supercell_size
        self.supercells = [self.primitive_cell]
        self.atom_to_substitute = atom_to_substitute
        self.atom_substitute_to = atom_substitute_to
        self.number_of_substitutions = number_of_substitutions
        self.write_vasp = write_vasp
        self.prefix = prefix
        self.substituted_sites = None
        self.throttle = throttle
        self.max_structures = max_structures #retain up to this number of structures in each run

    def make_one_substitution(self, supercell):
        sga = SpacegroupAnalyzer(supercell)
        symm_structure = sga.get_symmetrized_structure()
        substitution_site_coords = []

        for equiv_site in list(symm_structure.equivalent_sites):
            if self.atom_to_substitute == equiv_site[0].species.__str__().replace('1', ''):
                substitution_site_coords.append(equiv_site[0].coords)

        substituted_structures = [copy.deepcopy(supercell) for _ in substitution_site_coords]

        for i, struct in enumerate(substituted_structures):
            for no, site in enumerate(struct.__dict__['_sites']):
                if np.array_equal(site.__dict__['_coords'], substitution_site_coords[i]):
                    site.__dict__["_species"] = Composition(self.atom_substitute_to)

        return substituted_structures

    def _write_vasp_files(self, supercell, dir_name):
        import os
        cwd = os.getcwd()
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        os.chdir(cwd + '/' + dir_name)
        from pymatgen.io.vasp import Poscar
        Poscar(supercell.get_sorted_structure()).write_file('POSCAR')
        os.chdir(cwd)

    def make_all_substituted_configurations(self):
        substituted_count = 0
        while substituted_count < self.number_of_substitutions:
            self.substituted_sites = []
            unique = 0
            _list_holder = copy.deepcopy(self.supercells)
            self.supercells = []

            if substituted_count < 2:
                for k, supercell in enumerate(_list_holder):
                    for extra_doped_supercell in self.make_one_substitution(supercell):
                        self.supercells.append(extra_doped_supercell)

                        #if self.write_vasp:
                        #    dir_name = "SC_" + str(self.sc_size[0]) + "_" + str(self.sc_size[1]) + "_" + str(
                        #        self.sc_size[2]) + '_' + self.prefix + '_' + str(substituted_count + 1) + "_str_" + str(
                        #        len(self.supercells))
                        #    self._write_vasp_files(extra_doped_supercell, dir_name=dir_name)
                        #    unique += 1
            else:
                import random

                filter = RemoveDuplicatesFilter(symprec=1e-5)

                extra_doped_supercells = []

                doped_sites = []

                if len(_list_holder) > self.throttle:
                    _list_holder = random.choices(_list_holder, k=self.throttle)

                for counter, supercell in enumerate(_list_holder):
                    extra_doped_supercells += self.make_one_substitution(supercell)

                    for s in extra_doped_supercells:
                        __site_no = []
                        for no, site in enumerate(s.__dict__['_sites']):
                            if site.__dict__["_species"] == Composition(self.atom_substitute_to):
                                __site_no.append(no)
                        __site_no = list(sorted(__site_no))
                        if __site_no not in doped_sites:
                            doped_sites.append(__site_no)
                            self.supercells.append(s)
                    print(len(self.supercells))

            if self.max_structures is not None:
                if len(self.supercells)>self.max_structures:
                    import random
                    self.supercells=random.sample(self.supercells,self.max_structures)

            if self.write_vasp:
                for i, cell in enumerate(self.supercells):
                    dir_name = "SC_" + str(self.sc_size[0]) + "_" + str(self.sc_size[1]) + "_" + str(
                        self.sc_size[2]) + '_' + self.prefix + '_' + str(substituted_count + 1) + "_str_" + str(i)
                    self._write_vasp_files(cell, dir_name=dir_name)
                    unique += 1

            print("number of dopants:" + str(substituted_count + 1) + ' number of configurations:' + str(unique))

            substituted_count += 1

        #self.supercells = [map_pymatgen_IStructure_to_crystal(s) for s in self.supercells]

        return self.supercells


def deform_crystal_by_lattice_expansion_coefficients(crystal, def_fraction=[0.0,0.0,0.0]):
    _new_asym_unit = []
    for mol in crystal.asymmetric_unit:
        _new_atoms = [Atom(label=atom.label, scaled_position=atom.scaled_position) for atom in mol.atoms]
        _new_asym_unit.append(Molecule(atoms=_new_atoms))

    # make a new lattice
    lattice = crystal.lattice.scale_by_lattice_expansion_coefficients(def_fraction)

    return Crystal(lattice=lattice, asymmetric_unit=_new_asym_unit, space_group=crystal.space_group)
