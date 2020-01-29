from libc.math cimport sin, cos
from math import pi,pow
from vector3d cimport cVector3D
from matrix3d cimport cMatrix3D

cdef class Lattice:
    cdef double parameters[6], _vectors[3][3], _inv_vectors[3][3]
    cdef public object _crystal
    cdef object vecs, ivecs, __lattice_vectors
    cdef str lattice_system

    def __init__(self, a, b, c, alpha, beta, gamma, crystal=None):
        """
        Initialise a lattice object by given the six lattice parameters.
        This is also the default initializer for a triclinic unit cell.

        :param a: Cell length a given in Angstorms
        :param b: Cell length b given in Angstorms
        :param c: Cell length c given in Angstorms
        :param alpha: Cell angle alpha given in Degrees
        :param beta: Cell angle beta given in Degrees
        :param gamma: Cell angle gamma given in Degrees
        :param crystal: (Optional) The crystal to which this lattice belongs to
        """
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._crystal = crystal

        self.__lattice_vectors = False

        self.vecs = False
        self.ivecs = False


    cpdef copy(self):
        return self.__class__(self.a, self.b, self.c, self.alpha, self.beta, self.gamma)

    def __reduce__(self):
        return (self.__class__, (self.a, self.b, self.c, self.alpha, self.beta, self.gamma))

    cpdef set_by_index(self, i, value):
        self.parameters[i] = value

    def __triclinic_setting(self):
        return (self.a != self.b) and \
               (self.b != self.c) and \
               (self.a != self.c) and \
               (self.alpha != self.beta) and \
               (self.beta != self.gamma) and \
               (self.gamma != self.alpha)

    def __monoclinic_setting(self):
        return (self.a != self.b) and \
               (self.b != self.c) and \
               (self.a != self.c) and \
               (self.alpha == 90) and \
               (self.beta != 90) and \
               (self.gamma == 90)

    def __orthorombic_setting(self):
        return (self.a != self.b) and \
               (self.b != self.c) and \
               (self.a != self.c) and \
               (self.alpha == 90) and \
               (self.beta == 90) and \
               (self.gamma == 90)

    def __tetragonal_setting(self):
        return (self.a == self.b) and \
               (self.b != self.c) and \
               (self.alpha == 90) and \
               (self.beta == 90) and \
               (self.gamma == 90)

    def __rhombohedral_setting(self):
        return (self.a == self.b) and \
               (self.b == self.c) and \
               (self.alpha != 90) and \
               (self.beta != 90) and \
               (self.gamma != 90)

    def __hexagonal_setting(self):
        return (self.a == self.b) and \
               (self.b != self.c) and \
               (self.alpha == 90) and \
               (self.beta == 90) and \
               (self.gamma == 120)

    def __cubic_setting(self):
        return (self.a == self.b) and \
               (self.b == self.c) and \
               (self.alpha == 90) and \
               (self.beta == 90) and \
               (self.gamma == 90)

    property lattice_system:
        def __get__(self):
            if self.__triclinic_setting():
                return 'TRICLINIC'
            elif self.__monoclinic_setting():
                return 'MONOCLINIC'
            elif self.__orthorombic_setting():
                return 'ORTHOROMBIC'
            elif self.__tetragonal_setting():
                return 'TETRAGONAL'
            elif self.__rhombohedral_setting():
                return 'RHOMBOHEDRAL'
            elif self.__hexagonal_setting():
                return 'HEXAGONAL'
            elif self.__cubic_setting():
                return 'CUBIC'
            else:
                raise Exception("Lattice Parameters non--compatiable with default lattice system settings.")

    @classmethod
    def cubic(cls, a=100):
        return cls(a, a, a, 90, 90, 90)

    @classmethod
    def hexagonal(cls, a=10, c=20):
        if (a == c):
            raise Exception("Lattice setting incompatible with Hexagonal settings!")
        return cls(a, a, c, 90, 90, 120)

    @classmethod
    def rhombohedral(cls, a=50, alpha=10, beta=20, gamma=30):
        if (alpha == 90) or (beta == 90) or (gamma == 90):
            raise Exception("Lattice setting incompatible with Rhombohedral settings!")
        else:
            return cls(a, a, a, alpha, beta, gamma)

    @classmethod
    def trigonal(cls, a=10, alpha=10):
        assert alpha != 90
        return cls(a, a, a, alpha, alpha, alpha)

    @classmethod
    def tetragonal(cls, a=50, c=20):
        if (a == c):
            raise Exception("Lattice setting incompatible with Hexagonal settings!")
        return cls(a, a, c, 90, 90, 90)

    @classmethod
    def orthorhombic(cls, a=10, b=20, c=30):
        if (a == b) or (a == c) or (b == c):
            raise Exception("Lattice setting incompatible with Orthorhombic settings!")
        return cls(a, b, c, 90, 90, 90)

    @classmethod
    def monoclinic(cls, a=10, b=20, c=30, beta=50):
        if beta == 90:
            raise Exception("Lattice setting incompatible with monoclinic settings!")
        return cls(a, b, c, 90, beta, 90)

    @classmethod
    def triclinic(cls, a=10, b=20, c=30, alpha=40, beta=50, gamma=60):
        return cls(a, b, c, alpha, beta, gamma)

    @classmethod
    def from_lattice_vectors(cls, lattice_vectors):
        a=lattice_vectors.get_row(0)
        b=lattice_vectors.get_row(1)
        c=lattice_vectors.get_row(2)
        a_norm=a.l2_norm()
        b_norm=b.l2_norm()
        c_norm=c.l2_norm()
        alpha=b.angle(c)
        beta=c.angle(a)
        gamma=a.angle(b)
        return cls(a_norm,b_norm,c_norm,alpha,beta,gamma)

    def scale_by_volume_fraction(self,vol_fraction):
        #work out how much each component of the lattice vector needs to be scaled.
        frac = pow(1.0+vol_fraction,1.0/6.0)
        v1 = self.lattice_vectors.get_row(0)#[0],self._vectors[0][1],self._vectors[0][2])
        v2 = self.lattice_vectors.get_row(1)#[0],self._vectors[1][1],self._vectors[1][2])
        v3 = self.lattice_vectors.get_row(2)#[0],self._vectors[2][1],self._vectors[2][2])
        v1 = v1.vec_scale(frac)
        v2 = v2.vec_scale(frac)
        v3 = v3.vec_scale(frac)
        new_latt_mat = cMatrix3D(v1,v2,v3)
        return Lattice.from_lattice_vectors(new_latt_mat)

    def scale_by_lattice_expansion_coefficients(self,def_fraction):
        v1 = self.lattice_vectors.get_row(0)#[0],self._vectors[0][1],self._vectors[0][2])
        v2 = self.lattice_vectors.get_row(1)#[0],self._vectors[1][1],self._vectors[1][2])
        v3 = self.lattice_vectors.get_row(2)#[0],self._vectors[2][1],self._vectors[2][2])
        v1 = v1.vec_scale(1+def_fraction[0])
        v2 = v2.vec_scale(1+def_fraction[1])
        v3 = v3.vec_scale(1+def_fraction[2])
        new_latt_mat = cMatrix3D(v1,v2,v3)
        return Lattice.from_lattice_vectors(new_latt_mat)

    property parameter_dict:
        def __get__(self):
            return {'a': self.a,
                    'b': self.b,
                    'c': self.c,
                    'alpha': self.alpha,
                    'beta': self.beta,
                    'gamma': self.gamma}

    property lattice_parameters:
        def __get__(self):
            return list(self.parameters)
        def __set__(self, values):
            self.parameters = values

    property a:
        def __get__(self):
            return self.parameters[0]
        def __set__(self, value):
            self.parameters[0] = value
    property b:
        def __get__(self):
            return self.parameters[1]
        def __set__(self, value):
            self.parameters[1] = value
    property c:
        def __get__(self):
            return self.parameters[2]
        def __set__(self, value):
            self.parameters[2] = value
    property alpha:
        def __get__(self):
            return self.parameters[3]
        def __set__(self, value):
            self.parameters[3] = value
    property beta:
        def __get__(self):
            return self.parameters[4]
        def __set__(self, value):
            self.parameters[4] = value
    property gamma:
        def __get__(self):
            return self.parameters[5]
        def __set__(self, value):
            self.parameters[5] = value

    property angles:
        def __get__(self):
            return self.alpha, self.beta, self.gamma
        def __set__(self, angles):
            self.alpha = angles[0]
            self.beta = angles[1]
            self.gamma = angles[2]

    property lengths:
        def __get__(self):
            return self.a, self.b, self.c
        def __set__(self, lengths):
            self.a = lengths[0]
            self.b = lengths[1]
            self.c = lengths[2]

    property volume:
        def __get__(self):
            return self.angle_component_of_volume * self.length_component_of_volume

    property length_component_of_volume:
        def __get__(self):
            return self.a * self.b * self.c

    property angle_component_of_volume:
        def __get__(self):
            d2r = pi / 180.0
            VStar = (1.0 + 2.0 * cos(self.alpha * d2r)
                     * cos(self.beta * d2r)
                     * cos(self.gamma * d2r)
                     - cos(self.alpha * d2r) ** 2
                     - cos(self.beta * d2r) ** 2
                     - cos(self.gamma * d2r) ** 2) ** 0.5
            return VStar

    property lattice_vectors:
        def __get__(self):
            if (self.__lattice_vectors is None) or (not self.__lattice_vectors):
                return self.vectors
            else:
                return self.__lattice_vectors
        def __set__(self, lv):
            if lv is None:
                pass
            elif isinstance(lv,cMatrix3D):
                self.__lattice_vectors = lv

    property vectors:
        def __get__(self):
            cdef double d2r, Az, Ay, Ax
            cdef cVector3D A, B, C
            cdef cVector3D tmp
            if self.vecs:
                return self.vecs
            d2r = pi / 180.0
            C = cVector3D(0.0, 0.0, self.c)
            B = cVector3D(0.0, 0.0, 1.0).rotated(self.alpha * d2r, cVector3D(-1.0, 0.0, 0.0))
            B = B._c_vec_scale(self.b)
            Az = self.a * cos(self.beta * d2r)
            Ay = self.a * (cos(self.gamma * d2r) - cos(self.beta * d2r) * cos(self.alpha * d2r)) / sin(self.alpha * d2r)
            Ax = self.a * self.angle_component_of_volume / sin(self.alpha * d2r)
            A = cVector3D(Ax, Ay, Az)
            self._vectors[0][0] = A.x
            self._vectors[0][1] = A.y
            self._vectors[0][2] = A.z
            self._vectors[1][0] = B.x
            self._vectors[1][1] = B.y
            self._vectors[1][2] = B.z
            self._vectors[2][0] = C.x
            self._vectors[2][1] = C.y
            self._vectors[2][2] = C.z
            self.vecs = cMatrix3D(A, B, C)
            return self.vecs
        def __set__(self,vecs):
            self.vecs = vecs

    property inv_lattice_vectors:
        def __get__(self):
            return self.inv_vectors

    property inv_vectors:
        def __get__(self):
            if self.ivecs:
                return self.ivecs
            m = self.vectors
            cdef double det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) - \
                              m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + \
                              m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            cdef double invdet = 1 / det
            self._inv_vectors[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * invdet
            self._inv_vectors[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invdet
            self._inv_vectors[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invdet
            self._inv_vectors[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invdet
            self._inv_vectors[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invdet
            self._inv_vectors[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * invdet
            self._inv_vectors[2][0] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * invdet
            self._inv_vectors[2][1] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * invdet
            self._inv_vectors[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * invdet
            self.ivecs = cMatrix3D(
                cVector3D(self._inv_vectors[0][0], self._inv_vectors[0][1], self._inv_vectors[0][2]).copy(),
                cVector3D(self._inv_vectors[1][0], self._inv_vectors[1][1], self._inv_vectors[1][2]).copy(),
                cVector3D(self._inv_vectors[2][0], self._inv_vectors[2][1], self._inv_vectors[2][2]).copy())
            return self.ivecs

    @staticmethod
    def anisotropic_grid(n_a, n_b, n_c):
        """
        Make a grid of integer-valued lattice point, defining how far to translate (in fractional space)
        along each given lattice vector. This is anisotrpic version which allows different lenghts
        of translations along each direction.

        :param int n_a: How far along the :math:`a` lattice direction to go.
        :param int n_b: How far along the :math:`b` lattice direction to go.
        :param int n_c: How far along the :math:`c` lattice direction to go.
        :return: A nested list of all the translation vectors in fractional sapce.
        """
        return [cVector3D(*[i, j, k]) for i in range(-1 * n_a, n_a + 1) for j in range(-1 * n_b, n_b + 1) for k in
                range(-1 * n_c, n_c + 1)]

    @staticmethod
    def isotropic_grid(n):
        """
        Make a grid of integer-valued lattice point, defining how far to translate (in fractional space)
        along each given lattice vector. This is the isotrpic version which the same length will be translated
        along each direction

        :param n: How far along each lattice direction to go.
        :return: A nested list of all the translation vectors in fractional sapce.
        """
        return Lattice.anisotropic_grid(n, n, n)

