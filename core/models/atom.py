'''
The Atom Class
==============

'''
from __future__ import division, absolute_import, print_function
import re
from core.models.element import *
from core.models.vector3d import cVector3D


class Atom(object):
    def __init__(self, label=None, position=None, scaled_position=None,crystal=None, magmom=None):
        """
        Initialise an atom
        """
        self.label = label
        self.position = position
        self.scaled_position = scaled_position
        self.crystal = crystal
        self.magmom = magmom
        self.__set_element()

    @property
    def position(self):
        if self.__position is None:
            if self.__scaled_position and self.crystal:
                self.__position = self.__scaled_position.vec_mat_mul(self.get_lattice().lattice_vectors)
        return self.__position

    @position.setter
    def position(self, position):
        if isinstance(position, list):
            self.__position = cVector3D(*position)
        else:
            self.__position = position

    @property
    def scaled_position(self):
        if not self.__scaled_position:
            if self.__position and self.crystal:
                self.__scaled_position = self.position.vec_mat_mul(self.get_lattice().inv_vectors)
        return self.__scaled_position

    @scaled_position.setter
    def scaled_position(self, scaled_position):
        if isinstance(scaled_position, list):
            self.__scaled_position = cVector3D(*scaled_position)
        else:
            self.__scaled_position = scaled_position

    @property
    def clean_label(self):
        self.cl = self.label
        self.cl = self.cl.replace(" ", "")  # turns " C123_A" to "C123_A"
        self.cl = re.sub('\d+', " ", self.cl)  # turns "C123_A" to "C   _A"
        self.cl = re.sub("[\(\_].*", " ", self.cl)  # turns "C   _A" to C    A"
        self.cl = self.cl.split()[0]  # returns "C"
        self.cl = self.cl.capitalize()
        return self.cl

    def get_lattice(self):
        """
        Retrieve the crystal lattice information from the molecule class
        """
        return self.crystal.lattice

    def __set_element(self):
        """
        Set the atoms elemental properties based on the original label
        Returned as a named tuple currently containing:
        :param symbol: atomic symbol
        :type symbol: string
        :param atomic_number: atomic number
        :type atomic_number: int
        :type vdw_radius: the vdw radius
        :param atomic_number: float
        """
        _symbol = ''.join(c for c in self.label.strip()[:2] if c.isalpha() is True)
        self.element = element_dict[_symbol]

    def is_transition_metal(self):
        return self.clean_label in transition_metals

    def is_rare_earth(self):
        return self.clean_label in rare_earth_metals
