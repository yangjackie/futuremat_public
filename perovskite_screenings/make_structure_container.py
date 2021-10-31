def get_temperature_dependent_second_order_fc():
    from ase.io import read
    import numpy as np
    from hiphive import ClusterSpace, StructureContainer
    from hiphive.utilities import get_displacements
    from hiphive import ForceConstantPotential
    from hiphive.fitting import Optimizer
    from hiphive.calculators import ForceConstantCalculator
    import os

    if os.path.exists('POSCAR-md'):
        reference_structure = read('POSCAR-md')
    else:
        return None
    if not os.path.exists('./vasprun_md.xml'):
        return None

    cs = ClusterSpace(reference_structure, [3])
    fit_structures = []
    atoms = read("./vasprun_md.xml", index=':')
    for i, a in enumerate(atoms):
        displacements = get_displacements(a, reference_structure)
        atoms_tmp = reference_structure.copy()
        atoms_tmp.new_array('displacements', displacements)
        atoms_tmp.new_array('forces', a.get_forces())
        atoms_tmp.positions = reference_structure.positions
        fit_structures.append(atoms_tmp)

    sc = StructureContainer(cs)  # need a cluster space to instantiate the object!
    sc.delete_all_structures()
    for ii, s in enumerate(fit_structures):
        try:
            sc.add_structure(s)
        except Exception as e:
            logger.info(ii, e)
            pass
    path_to_structure_container_filename='structure_container'
    sc.write(path_to_structure_container_filename)

if __name__=="__main__":
    get_temperature_dependent_second_order_fc()