from vector3d cimport cVector3D

cdef class cMatrix3D:
    cdef double data[3][3]
    cdef int index
    cdef double _c_det(cMatrix3D self)
    cdef cMatrix3D _c_inverse(cMatrix3D self)
    cdef cVector3D _c_dot_vec(cMatrix3D self, cVector3D vec)
    cdef cMatrix3D _c_transpose(cMatrix3D self)