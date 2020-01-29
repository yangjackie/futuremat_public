import math

def kpoints_from_grid(crystal, grid=0.04, molecular=False):
    """
    Based on the length of lattice vectors and a desired k-point spacing, the number of k points per direction to full filled the requiremente is given as :math:`1/(\mbox{grid}\cdot a)`.

    :param float grid: k-point spacing, default set to be 0.04
    :return list k_point: A list of number of k-points to be sampled along each cell direction.
    """
    k_points = []
    if not molecular:
        for vector in crystal.lattice.lattice_vectors:
            length = vector.l2_norm()
            aux = math.ceil(1.0 / (grid * length))
            k_points.append(aux)
    else:
        k_points=[1.0,1.0,1.0]

    return k_points