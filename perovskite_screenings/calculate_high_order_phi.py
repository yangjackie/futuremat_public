def calculate_high_order_phi():
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

    a=reference_structure.get_cell_lengths_and_angles()[0]
    cs = ClusterSpace(reference_structure, [0.45*a,0.30*a,0.30*a])
    sc=None
    #try:
    #    sc = StructureContainer.read("./structure_container")
    #    print("successfully loaded the structure container "+str(sc))

    #except:
    try:
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
                print(ii, e)
                pass
    except:
        pass

    if sc is not None:
       try:
          opt = Optimizer(sc.get_fit_data(),fit_method="ardr",train_size=0.9)
          opt.train()
          fcp = ForceConstantPotential(cs, opt.parameters)
          fcs = fcp.get_force_constants(reference_structure)
       except Exception as e:
          print(e)
          return None

       from core.external.vasp.anharmonic_score import AnharmonicScore
       sigmas_2=None
       sigmas_3=None
       sigmas_4=None
       try:
           scorer = AnharmonicScore(md_frames='./vasprun_md.xml', ref_frame='./POSCAR-md',
                                                                  force_constants=fcs.get_fc_array(2),
                                                                  include_third_order=False,
                                                                  include_fourth_order=False)
           sigmas_2, _ = scorer.structural_sigma(return_trajectory=False)
       except:
           pass

       try:
           scorer = AnharmonicScore(md_frames='./vasprun_md.xml', ref_frame='./POSCAR-md',
                                                                  force_constants=fcs.get_fc_array(2),
                                                                  include_third_order=True,
                                                                  third_order_fc=fcs.get_fc_array(3),
                                                                  include_fourth_order=False)
           sigmas_3, _ = scorer.structural_sigma(return_trajectory=False)
       except:
           pass

       try:
           scorer = AnharmonicScore(md_frames='./vasprun_md.xml', ref_frame='./POSCAR-md',
                                                                  force_constants=fcs.get_fc_array(2),
                                                                  include_third_order=True,
                                                                  third_order_fc=fcs.get_fc_array(3),
                                                                  include_fourth_order=True,
                                                                  fourth_order_fc=fcs.get_fc_array(4))
           sigmas_4, _ = scorer.structural_sigma(return_trajectory=False)
       except:
           pass
       f=open('sigmas_updated.dat','w')
       f.write('sigma_2'+'\t'+str(sigmas_2)+'\n')
       f.write('sigma_3'+'\t'+str(sigmas_3)+'\n')
       f.write('sigma_4'+'\t'+str(sigmas_4)+'\n')
       f.close()

if __name__=="__main__":
    from core.external.vasp.anharmonic_score import *
    sigma = None
    try:
        scorer = AnharmonicScore(md_frames='vasprun_md.xml', ref_frame='./SPOSCAR',
                             force_constants=None, force_sets_filename='FORCE_SETS_222')
        sigma, _ = scorer.structural_sigma(return_trajectory=False)
    except:
        pass
    if (sigma is not None) and (sigma<1.0):
        calculate_high_order_phi()
