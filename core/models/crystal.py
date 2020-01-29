''' Crystals '''
from __future__ import division, absolute_import, print_function
import numpy as np
import copy

from core.models.space_group import SpaceGroup
from core.resources.crystallographic_space_groups import CrystallographicSpaceGroups


class Crystal(object):
    ''' The Crystal structure class

    The Crystal has a lattice, an asymmetric_unit and a space_group.
    It is a central container for CSPy

    :param lattice: The lattice defining the cell dimensions
    :type lattice: :class:.Lattice
    :param asymmetric_unit: A list of molecules in the asymmetric unit (symmetry independant)
    :type asymmetric_unit: list of :class:.Molecule
    :param space_group: The symmetry of the crystal
    :type space_group: :class:`.SpaceGroup`
    '''

    def __init__(self, lattice, asymmetric_unit, space_group):
        self.lattice = lattice

        self.asymmetric_unit = asymmetric_unit

        # auto set up the reverse link
        #TODO - need to realign molecule at origin and axis! To be consistent for treating molecular crystals!!!
        for molecule in self.asymmetric_unit:
            molecule._crystal = self
        self.lattice._crystal = self

        if isinstance(space_group, int):
            space_group = CrystallographicSpaceGroups.get(space_group)
        self.space_group = space_group

        ## set up the crystal for each atoms in the crystal
        self.__link_crystal_to_atoms()

        # check if the MP-k point for this crystal contains only gamma point
        self.gamma_only = False

    def copy(self):
        return self.__class__(self.lattice.copy(), [m.copy() for m in self.asymmetric_unit], self.space_group)

    def __reduce__(self):
        if isinstance(self.space_group, SpaceGroup):
            _space_group = self.space_group.index
        else:
            _space_group = self.space_group
        return (self.__class__, (self.lattice, self.asymmetric_unit, _space_group,))

    def __link_crystal_to_atoms(self):
        for molecule in self.asymmetric_unit:
            for atom in molecule.atoms:
                atom.crystal = self

    def all_atoms(self, sort=True, unique=True):
        """
        Method to retrieve all atoms in the asymmetric unit of the crystal as a flattened list.
        This is useful, for example, when trying to retrieve all atoms in the unit cell
        for the crystal structure to be written out in the input file for DFT calculations,
        after the crystal has been converted to P1.
        :return: A list of all atoms in the asymmetric unit.
        """
        all_atoms = []
        for mol in self.asymmetric_unit:
            for atom in mol.atoms:
                all_atoms.append(atom)
        if not unique:
            if sort:
                all_atoms.sort(key=lambda x: x.label, reverse=False)
            if not sort:
                pass
        else:
            _unique_atoms=[]
            for a in all_atoms:
                if a not in _unique_atoms:
                    _unique_atoms.append(a)
            all_atoms=_unique_atoms

        return all_atoms

    def all_atoms_count_dictionaries(self):
        """
        Method to retrieve all atoms in the asymmetric unit of the crystal and returns
        a dictionary for the number of each element in the asymmetric unit.

        :return: A dictionary of number of each element in the asymmetric units
        """
        labels = [x.label for x in self.all_atoms()]
        return dict((x,labels.count(x)) for x in set(labels))

    @property
    def mag_group(self):
        keyfunc = lambda a: (a.label, a.magmom)
        from itertools import groupby
        all_atoms = []

        for mol in self.asymmetric_unit:
            for atom in mol.atoms:
                all_atoms.append(atom)
        all_atoms = sorted(all_atoms, key=keyfunc)
        self.__mag_group = groupby(all_atoms, keyfunc)
        return self.__mag_group
