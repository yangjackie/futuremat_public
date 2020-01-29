from core.models import Atom
from core.models.vector3d import cVector3D
from fractions import Fraction


class SpaceGroup(object):
    def __init__(self,
                 index=None,
                 name=None,
                 lattice_system=None,
                 lattice_centering=None,
                 inversion=None,
                 symmetry=None,
                 asymmetric_unit=None,
                 unique_axis=None):
        self.index = index
        self.name = name
        self.lattice_system = lattice_system
        self.lattice_centering = lattice_centering
        self.inversion = inversion
        self.symmetry = symmetry
        self.asymmetric_unit = asymmetric_unit
        self.unique_axis = unique_axis
        self.non_centering_symmetry = []

        self.full_symmetry = []
        self.__compute_full_symmetry()

    @property
    def identity(self):
        return SymmetryOperation('x,y,z')

    def __append_identity(self):
        if 'x,y,z' not in [i.operation_string for i in self.symmetry]:
            self.symmetry = [self.identity] + self.symmetry

    def __add_inversion_symmetry(self):
        for op in self.symmetry:
            self.non_centering_symmetry.append(op)

        if isinstance(self.inversion,CentroSymmetric):
            for i in range(len(self.non_centering_symmetry)):
                op = self.non_centering_symmetry[i]
                self.non_centering_symmetry.append(op.inversion())

    def __add_centering_symmetry(self):
        if self.lattice_centering:
            for i in range(len(self.full_symmetry)):
                op = self.full_symmetry[i]
                centering_ops = self.lattice_centering.transform(op)
                self.full_symmetry += centering_ops

    def __compute_full_symmetry(self):
        self.__append_identity()
        self.__add_inversion_symmetry()
        self.full_symmetry = [op for op in self.non_centering_symmetry]
        self.__add_centering_symmetry()
        return self.full_symmetry


class SymmetryOperation(object):
    def __init__(self, operation_string):
        """
        Initialize a symmetry operation object from a string representation of symmetry operation.

        :param operation_string: A string (as read in from res/cif file) representing a symmetry operation
            for a space group.
        """
        self.operation_string = operation_string.lower()
        self.operation_function = None
        self.__set_operation_function()

    def __set_operation_function(self):
        """
        Convert the string form of the symmetry operation into the form of a mathematical
         function that can be directly applied to a vector to transform a point to a
         symmetry related point.
        """
        if self.operation_function is not None:
            return self.operation_function
        else:
            self.operation_function = symm_eval

    def transform_scaled_position(self, data):
        """
        Applying this symmetry operation on a 3D coordinate to transform it to a symmetry-related
        position in the crystal.

        :param data: A vector (:class:`entdecker.core.models.vector3d.cVector3D`)
                    representing the fractional coordinates on which the symmetry
                    operation will be applied upon.
        :return: Symmetry transformed vector.
        """
        return self.operation_function(prepare_operation(self.operation_string), data)

    def transform_atom(self, atom):
        return Atom(label=atom.label, scaled_position=self.transform_scaled_position(atom.scaled_position))

    def inversion(self):
        func = lambda x: "-1*(%s),-1*(%s),-1*(%s)" % tuple(x.split(","))
        return self.__class__(func(self.operation_string))


class Symmetry(object):
    @staticmethod
    def get(value):
        if value is '' or value == 'UNKNOWN':
            return []
        return [SymmetryOperation(v) for v in value.split(';')]


class Inversion(object):
    # TODO - this needs to be fixed later to make it consistent with the Centering class!
    YES = True
    NO = False
    UNKNOWN = None


class InversionFactory(object):
    @staticmethod
    def construct(latt):
        if int(latt) > 0:
            return CentroSymmetric()
        else:
            return NonCentroSymmetric()


class CentroSymmetric(Inversion):
    @staticmethod
    def transform(op):
        func = lambda x: "-1*(%s),-1*(%s),-1*(%s)" % tuple(x.split(","))
        return func(op)


class NonCentroSymmetric(Inversion):
    @staticmethod
    def transform(op):
        return op


class Centering(object):
    def __init__(self, letter, additional_lattice_points):
        self.letter = letter
        self.additional_lattice_points = additional_lattice_points

    def transform(self, op):
        additional_ops = []
        for point in self.additional_lattice_points:
            func = lambda x: "{0}+{3},{1}+{4},{2}+{5}".format(*(x.split(",") + list(point)))
            additional_ops.append(op.__class__(func(op.operation_string)))
        return additional_ops

    @classmethod
    def primitive(cls):
        return cls('P', [])

    @classmethod
    def body_centered(cls):
        return cls('I', [(0.5, 0.5, 0.5)])

    @classmethod
    def hexagonal(cls):
        return cls('H', [(Fraction(2, 3), Fraction(1, 3), 0.0),
                         (Fraction(1, 3), Fraction(2, 3), 0.0)])

    @classmethod
    def rhombohedral(cls):
        return cls('R', [(Fraction(2, 3), Fraction(1, 3), Fraction(1, 3)),
                         (Fraction(1, 3), Fraction(2, 3), Fraction(2, 3))])

    @classmethod
    def face_centered(cls):
        return cls('F', [(0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)])

    @classmethod
    def base_centered_A(cls):
        return cls('A', [(0.0, 0.5, 0.5)])

    @classmethod
    def base_centered_B(cls):
        return cls('B', [(0.5, 0.0, 0.5)])

    @classmethod
    def base_centered_C(cls):
        return cls('C', [(0.5, 0.5, 0.0)])

    @classmethod
    def construct(cls, latt):
        """
        Given the LATT directive in a res file, return the corresponding centered lattice type.

        :param latt: the absolute integer value specified in LATT
        :return: corrected centered lattice type
        """
        latt = abs(latt)
        if latt == 1:
            return cls.primitive()
        elif latt == 2:
            return cls.body_centered()
        elif latt == 3:
            # default setting from reading in a res file is Rhombohedral
            return cls.rhombohedral()
        elif latt == 4:
            return cls.face_centered()
        elif latt == 5:
            return cls.base_centered_A()
        elif latt == 6:
            return cls.base_centered_B()
        elif latt == 7:
            return cls.base_centered_C()

    def get_LATT_code(self):
        if self.letter == 'P':
            return 1
        elif self.letter == 'I':
            return 2
        elif self.letter == 'R':
            return 3
        elif self.letter == 'F':
            return 4
        elif self.letter == 'A':
            return 5
        elif self.letter == 'B':
            return 6
        elif self.letter == 'C':
            return 7


class AsymmetricUnit(object):
    UNKNOWN = [[0, 1.00], [0, 1.00], [0, 1.00]]
    FULL = [[0, 1.00], [0, 1.00], [0, 1.00]]
    HALF_X = [[0, 0.50], [0, 1.00], [0, 1.00]]
    HALF_Y = [[0, 1.00], [0, 0.50], [0, 1.00]]
    HALF_Z = [[0, 1.00], [0, 1.00], [0, 0.50]]
    QUART_Y = [[0, 1.00], [0, 0.25], [0, 1.00]]
    HALF_X_QUART_Y = [[0, 0.50], [0, 0.25], [0, 1.00]]
    HALF_XZ = [[0, 0.50], [0, 1.00], [0, 0.50]]
    HALF_XY = [[0, 0.50], [0, 0.50], [0, 1.00]]
    EIGHT_Z = [[0, 1.00], [0, 1.00], [0, 0.125]]


class UniqueAxis(object):
    UNKNOWN = -1
    NA = -1
    X = 0
    Y = 1
    Z = 2


def symm_eval(s, data):
    x, y, z = data.x, data.y, data.z
    out = list(map(eval, s.split(",")))
    return cVector3D(out[0], out[1], out[2])


def prepare_operation(s):
    ''' Cleans up a string of a symmetry operation to be used in eval or exec

    :param s: Input string e.g. "x,y,z+1/2"
    :type s: string

    :rtype: string
    '''
    tmp = s.replace("1/4", "1.0/4.0")
    tmp = tmp.replace("1/2", "1.0/2.0")
    tmp = tmp.replace("3/4", "3.0/4.0")
    tmp = tmp.replace("1/3", "1.0/3.0")
    tmp = tmp.replace("2/3", "2.0/3.0")
    tmp = tmp.replace("1/6", "1.0/6.0")
    tmp = tmp.replace("5/6", "5.0/6.0")
    return tmp.replace(" ", "").lower()
