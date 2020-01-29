from matrix3d cimport cMatrix3D

cdef class cVector3D:
    cdef double xyz[3]
    cdef int index
    cdef cVector3D _c_copy(cVector3D self)
    cdef cVector3D _c_add(cVector3D self, cVector3D other)
    cdef cVector3D _c_sub(cVector3D self, cVector3D other)
    cdef cVector3D _c_vec_scale(cVector3D self, double factor)
    cdef double _c_dot(cVector3D self, cVector3D other)
    cdef double _c_l2_norm(cVector3D self)
    cdef cVector3D _c_cross(cVector3D self,cVector3D other)
    cdef cVector3D _c_vec_mat_mul(cVector3D self, cMatrix3D m)
    cdef cVector3D _c_rotated(cVector3D self, double angle, cVector3D about)
    cdef double _c_angle(cVector3D self, cVector3D other)
    cdef cVector3D _c_normalise(cVector3D self)
