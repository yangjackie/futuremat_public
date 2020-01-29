from vector3d cimport cVector3D
import cython

cdef class cMatrix3D:
    def __init__(self, cVector3D row1, cVector3D row2, cVector3D row3):
        self.data[0][0] = row1.xyz[0]
        self.data[0][1] = row1.xyz[1]
        self.data[0][2] = row1.xyz[2]
        self.data[1][0] = row2.xyz[0]
        self.data[1][1] = row2.xyz[1]
        self.data[1][2] = row2.xyz[2]
        self.data[2][0] = row3.xyz[0]
        self.data[2][1] = row3.xyz[1]
        self.data[2][2] = row3.xyz[2]

    def __iter__(self):
        for item in [cVector3D(self.data[0][0], self.data[0][1], self.data[0][2]),
                     cVector3D(self.data[1][0], self.data[1][1], self.data[1][2]),
                     cVector3D(self.data[2][0], self.data[2][1], self.data[2][2])]:
            yield item

    def determinant(self):
        return self._c_det()

    def inverse(self):
        return self._c_inverse()

    def dot_vec(self, cVector3D vec):
        """
        Method to perform a right-dot product with a vector
        :param vec:
        :return:
        """
        return self._c_dot_vec(vec)

    def transpose(self):
        return self._c_transpose()

    @classmethod
    def identity(cls, diag=1.0):
        return cls(cVector3D(diag, 0., 0.),
                   cVector3D(0., diag, 0.),
                   cVector3D(0., 0., diag))

    @classmethod
    def zeros(cls):
        return cls.identity(diag=0.)

    def get(self, row, column):
        return self.data[row][column]

    def __getitem__(self, row):
        return self.data[row]

    def get_row(self, row):
        return cVector3D(self.data[row][0], self.data[row][1], self.data[row][2])

    cdef double _c_det(cMatrix3D self):
        cdef double det = self.data[0][0] * (self.data[1][1] * self.data[2][2] - self.data[2][1] * self.data[1][2]) - \
                          self.data[0][1] * (self.data[1][0] * self.data[2][2] - self.data[1][2] * self.data[2][0]) + \
                          self.data[0][2] * (self.data[1][0] * self.data[2][1] - self.data[1][1] * self.data[2][0])
        return det

    cdef cMatrix3D _c_inverse(cMatrix3D self):
        cdef double invdet = 1. / self._c_det()

        cdef cMatrix3D minv = cMatrix3D(cVector3D(0, 0, 0), cVector3D(0, 0, 0), cVector3D(0, 0, 0))
        minv.data[0][0] = (self.data[1][1] * self.data[2][2] - self.data[2][1] * self.data[1][2]) * invdet
        minv.data[0][1] = (self.data[0][2] * self.data[2][1] - self.data[0][1] * self.data[2][2]) * invdet
        minv.data[0][2] = (self.data[0][1] * self.data[1][2] - self.data[0][2] * self.data[1][1]) * invdet
        minv.data[1][0] = (self.data[1][2] * self.data[2][0] - self.data[1][0] * self.data[2][2]) * invdet
        minv.data[1][1] = (self.data[0][0] * self.data[2][2] - self.data[0][2] * self.data[2][0]) * invdet
        minv.data[1][2] = (self.data[1][0] * self.data[0][2] - self.data[0][0] * self.data[1][2]) * invdet
        minv.data[2][0] = (self.data[1][0] * self.data[2][1] - self.data[2][0] * self.data[1][1]) * invdet
        minv.data[2][1] = (self.data[2][0] * self.data[0][1] - self.data[0][0] * self.data[2][1]) * invdet
        minv.data[2][2] = (self.data[0][0] * self.data[1][1] - self.data[1][0] * self.data[0][1]) * invdet

        return minv

    cdef cVector3D _c_dot_vec(cMatrix3D self, cVector3D vec):
        cdef cVector3D out_vec = cVector3D(0.0, 0.0, 0.0)
        out_vec.xyz[0] = self.data[0][0]*vec.xyz[0] + self.data[0][1]*vec.xyz[1] + self.data[0][2]*vec.xyz[2]
        out_vec.xyz[1] = self.data[1][0]*vec.xyz[0] + self.data[1][1]*vec.xyz[1] + self.data[1][2]*vec.xyz[2]
        out_vec.xyz[2] = self.data[2][0]*vec.xyz[0] + self.data[2][1]*vec.xyz[1] + self.data[2][2]*vec.xyz[2]
        return out_vec

    cdef cMatrix3D _c_transpose(self):
        return cMatrix3D(cVector3D(self.data[0][0], self.data[1][0], self.data[2][0]),
                         cVector3D(self.data[0][1], self.data[1][1], self.data[2][1]),
                         cVector3D(self.data[0][2], self.data[1][2], self.data[2][2]))

    def __repr__(self):
        return str([cVector3D(self.data[0][0], self.data[0][1], self.data[0][2]),
                cVector3D(self.data[1][0], self.data[1][1], self.data[1][2]),
                cVector3D(self.data[2][0], self.data[2][1], self.data[2][2])])
