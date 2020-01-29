from core.models.vector3d import cVector3D

class Molecule(object):
    def __init__(self, atoms=None, crystal=None):
        self.name = None

        if atoms is None:
            self.atoms = []
        else:
            self.atoms = atoms

        for atom in self.atoms:
            atom._molecule = self

        if crystal is not None:
            for atom in self.atoms:
                atom.set_crystal(crystal)

    def copy(self):
        return self.__class__([a.copy() for a in self.atoms])

    def __reduce__(self):
        return (self.__class__, (self.atoms,))
