from phonopy.interface.calculator import read_crystal_structure
from pymatgen.analysis.magnetism import CollinearMagneticStructureAnalyzer
from pymatgen.transformations.standard_transformations import ConventionalCellTransformation

from core.calculators.vasp import Vasp, VaspReader
from core.internal.builders.crystal import build_supercell
from twodPV.calculators import default_bulk_optimisation_set, setup_logger, update_core_info, load_structure
import argparse, os, tarfile, shutil

def phonopy_workflow(force_rerun=False):
    from phonopy.interface.calculator import read_crystal_structure, write_crystal_structure
    from phonopy.interface.vasp import parse_set_of_forces
    from phonopy.file_IO import write_force_constants_to_hdf5
    from phonopy import Phonopy

    mp_points = [1, 1, 1]
    gamma_centered = True
    force_no_spin = False
    use_default_encut = False
    supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    ialgo = 38
    use_gw = True
    ncore = 32

    if mp_points != [1, 1, 1]:
        gamma_only = False
    else:
        gamma_only = True

    phonopy_set = {'prec': 'Normal', 'ibrion': -1, 'encut': 350, 'ediff': '1e-08', 'ismear': 0, 'ialgo': ialgo,
                   'lreal': False, 'lwave': False, 'lcharg': False, 'sigma': 0.05, 'isym': 0, 'ncore': ncore,
                   'ismear': 0, 'MP_points': mp_points, 'nelm': 400, 'lreal': False, 'use_gw': use_gw,
                   'Gamma_centered': gamma_centered, 'LMAXMIX': 6, 'amin':0.01, 'gpu_run': True,'SYMPREC':1e-4}
    # 'amix': 0.2, 'amix_mag':0.8, 'bmix':0.0001, 'bmix_mag':0.0001}

    logger = setup_logger(output_filename='phonopy.log')
    cwd = os.getcwd()
    vasp = Vasp()
    #vasp.check_convergence(outcar='./OUTCAR_nospin')
    #if not vasp.completed:
    #    logger.exception("Initial structure optimimization failed, will not proceed!")
    #    raise Exception("Initial structure optimimization failed, will not proceed!")

    if os.path.isfile('./force_constants.hdf5'):
        logger.info("previous phonopy calculations completed, will not rerun it again")
        return

    if os.path.isfile('phonopy.log'):
        success = []
        for l in open('phonopy.log', 'r').readlines():
            if 'VASP calculation completed successfully? ' in l:
                if l.split()[-1] == "True":
                    success.append(True)
                elif l.split()[-1] == 'False':
                    success.append(False)
        if len(success) != 0:
            if not all(success):
                if not force_rerun:
                    logger.exception("Encounter convergence problems with VASP before, will not attempt again.")
                    raise Exception("Encounter convergence problems with VASP before, will not attempt again.")
                else:
                    logger.info("try to rerun all VASP calculations")
                    print("try to rerun all VASP calculations with RMM for ialgo")

    try:
        if os.path.isfile('./CONTCAR_nospin'):
            unitcell, _ = read_crystal_structure('./CONTCAR_nospin', interface_mode='vasp')
        elif os.path.isfile('./CONTCAR'):
            unitcell, _ = read_crystal_structure('./CONTCAR', interface_mode='vasp')
    except:
        raise Exception("No CONTCAR!")

    if not force_rerun:
        if not os.path.exists('./phonopy'):
            os.mkdir('./phonopy')
        elif os.path.isfile("./phonopy.tar.gz"):
            tf = tarfile.open("./phonopy.tar.gz")
            tf.extractall()
    else:
        try:
            shutil.rmtree('./phonopy')
        except:
            pass
        try:
            os.rmtree('./phonopy')
        except:
            pass
        os.mkdir('./phonopy')

        try:
            os.remove("./phonopy.tar.gz")
        except:
            pass

    os.chdir('./phonopy')

    phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix)
    if not force_rerun:
        phonon.generate_displacements()
    else:
        phonon.generate_displacements(distance=0.0005)

    supercells = phonon.supercells_with_displacements
    logger.info(
        "PHONOPY - generate (2x2x2) displaced supercells, total number of configurations " + str(len(supercells)))

    completed = []
    force_files = []

    calculate_next = True
    for i, sc in enumerate(supercells):
        proceed = True
        if calculate_next:
            dir = 'ph-POSCAR-' + str(i)
            force_files.append('./' + dir + '/vasprun.xml')
            if not os.path.exists(dir):
                os.mkdir(dir)
            os.chdir(dir)
            proceed = True
            if os.path.isfile('./OUTCAR'):
                logger.info("Configuration " + str(i + 1) + '/' + str(
                    len(supercells)) + " previous calculation exists, check convergence")
                vasp = Vasp()
                vasp.check_convergence()
                if vasp.completed:
                    proceed = False
                    logger.info("Configuration " + str(i + 1) + '/' + str(
                        len(supercells)) + " previous calculation converged.")
                else:
                    calculate_next = True
                    proceed = True
                    logger.info('At least one finite displaced configuration cannot converge, quit the rest')

            if proceed:
                logger.info("Configuration " + str(i + 1) + '/' + str(len(supercells)) + " proceed VASP calculation")
                write_crystal_structure('POSCAR', sc, interface_mode='vasp')
                structure = load_structure(logger)
                structure.gamma_only = gamma_only
                #phonopy_set['magmom'], all_zeros = magmom_string_builder(structure)
                phonopy_set['ispin'] = 1

                if force_rerun:
                    phonopy_set['ispin'] = 1
                    if 'magmom' in phonopy_set.keys():
                        del phonopy_set['magmom']
                    try:
                        vasp = Vasp(**phonopy_set)
                        vasp.set_crystal(structure)
                        vasp.execute()
                    except:
                        pass

                    if vasp.completed is not True:
                        calculate_next = False
                        proceed = False
                        logger.info('At least one finite displaced configuration cannot converge, quit the rest')

                if not force_rerun:
                    try:
                        vasp = Vasp(**phonopy_set)
                        vasp.set_crystal(structure)
                        vasp.execute()
                    except:
                        pass

                    if vasp.completed is not True:
                        calculate_next = False
                        proceed = False
                        logger.info('At least one finite displaced configuration cannot converge, quit the rest')

                logger.info("Configuration " + str(i) + '/' + str(
                    len(supercells)) + "VASP terminated?: " + str(vasp.completed))
                print("Configuration " + str(i) + '/' + str(
                    len(supercells)) + "VASP terminated?: " + str(vasp.completed))
            completed.append(vasp.completed)
            os.chdir("..")

    if all(completed) and len(completed) == len(supercells):
        logger.info("All finite displacement calculations completed, extract force constants")
        set_of_forces = parse_set_of_forces(structure.total_num_atoms(), force_files)
        phonon.set_forces(sets_of_forces=set_of_forces)
        phonon.produce_force_constants()
        write_force_constants_to_hdf5(phonon.force_constants, filename='force_constants.hdf5')

        if os.path.isfile('force_constants.hdf5'):
            shutil.copy('./force_constants.hdf5', '../force_constants.hdf5')

    with tarfile.open('phonopy.tar.gz', mode='w:gz') as archive:
        archive.add('phonopy.tar.gz', recursive=True)

    try:
        shutil.rmtree('./phonopy')
    except:
        pass
    try:
        os.rmtree('./phonopy')
    except:
        pass

    os.chdir(cwd)


def molecular_dynamics_workflow(force_rerun=False):
    logger = setup_logger(output_filename='molecular_dynamics.log')
    cwd = os.getcwd()

    logger.info(
        "Setting up the room temperature molecular dynamics calculations, check if we have previous phonon data")

    if not os.path.isfile('./force_constants.hdf5'):
        logger.info("previous phonopy calculations did not complete properly, will not proceed...")
        raise Exception("No phonopy data! QUIT")
    #else:
    #    spin_polarized = check_phonon_run_settings()
    structure = __load_supercell_structure()
    structure.gamma_only = True

    #logger.info("Will run MD with spin polarization: " + str(spin_polarized))

    equilibrium_set = {'prec': 'normal','algo': 'Normal', 'lreal': 'AUTO', 'ismear': 0, 'isym': 0, 'ibrion': 0, 'maxmix': 40, 'amin':0.01,
                       'lmaxmix': 6, 'ncore': 32, 'nelmin': 4, 'nsw': 300, 'smass': -1, 'isif': 1, 'tebeg': 10,
                       'teend': 300, 'potim': 1, 'nblock': 10, 'nwrite': 0, 'lcharg': False, 'lwave': False,
                       'iwavpr': 11, 'encut': 350, 'Gamma_centered': True, 'MP_points': [1, 1, 1], 'use_gw': True,
                       'write_poscar': True}

    production_set = {'prec': 'normal','algo': 'Normal', 'lreal': 'AUTO', 'ismear': 0, 'isym': 0, 'ibrion': 0, 'maxmix': 40, 'amin':0.01,
                      'lmaxmix': 6, 'ncore': 32, 'nelmin': 4, 'nsw': 2000, 'isif': 1, 'tebeg': 300,
                      'teend': 300, 'potim': 1, 'nblock': 1, 'nwrite': 0, 'lcharg': False, 'lwave': False, 'iwavpr': 11,
                      'encut': 350, 'andersen_prob': 0.5, 'mdalgo': 1, 'Gamma_centered': True, 'MP_points': [1, 1, 1],
                      'use_gw': True, 'write_poscar': False}

    #if spin_polarized:
    #    raise Exception("Skip running spin polarized MD first ...")

    spin_polarized = False
    if spin_polarized:
        equilibrium_set['ispin'] = 2
        equilibrium_set['magmom'], _ = magmom_string_builder(structure)
        production_set['ispin'] = 2
        production_set['magmom'], _ = magmom_string_builder(structure)
    else:
        equilibrium_set['ispin'] = 1
        production_set['ispin'] = 1

    if not os.path.exists('./MD'):
        os.mkdir('./MD')
    os.chdir('./MD')

    try:
        os.remove('./INCAR')
    except:
        pass
    try:
        os.remove('./KPOINTS')
    except:
        pass

    run_equilibration = True
    run_production = True

    if os.path.exists('./CONTCAR_equ') and os.path.exists('./OSZICAR_equ'):
        logger.info("Previous equilibration run output exists, check how many cylces have been run...")
        oszicar = open('./OSZICAR_equ', 'r')
        cycles_ran = 0
        for l in oszicar.readlines():
            if 'T=' in l:
                cycles_ran += 1
        if cycles_ran == equilibrium_set['nsw']:
            logger.info('Previous equilibrium run completed, will skip running equilibration MD')
            shutil.copy('CONTCAR_equ', 'POSCAR')
            run_equilibration = False
        else:
            logger.info('Previous equilibrium run not completed, will rerun equilibration MD')
            run_equilibration = True

    if run_equilibration:
        try:
            logger.info("start equilibrium run ...")
            vasp = Vasp(**equilibrium_set)
            vasp.set_crystal(structure)
            vasp.execute()
        except:
            pass

        dav_error = False
        if not vasp.completed:
            logfile = open('vasp.log', 'r')
            for f in logfile.readlines():
                if 'Error EDDDAV' in f:
                    dav_error = True
            if dav_error:
                equilibrium_set['algo'] = 'VeryFast'
                production_set['algo'] = 'VeryFast'
            try:
                os.remove('./INCAR')
                logger.info("start equilibrium run ...")
                vasp = Vasp(**equilibrium_set)
                vasp.set_crystal(structure)
                vasp.execute()
            except:
                pass

        # error catching for md run needs to be implemented
        shutil.copy('INCAR', 'INCAR_equ')
        shutil.copy('POSCAR', 'POSCAR_equ')
        shutil.copy('CONTCAR', 'CONTCAR_equ')
        shutil.copy('CONTCAR', 'POSCAR')
        shutil.copy('vasprun.xml', 'vasprun_equ.xml')
        shutil.copy('OUTCAR', 'OUTCAR_equ')
        shutil.copy('OSZICAR', 'OSZICAR_equ')
        shutil.copy('vasp.log', 'vasp_equ.log')

    try:
        os.remove('./INCAR')
    except:
        pass

    if run_equilibration:
        run_production = True
    else:
        has_andersen = False
        if os.path.exists('./OUTCAR_prod'):
            logger.info("Check if previous production run has applied andersen thermostat")
            outcar = open('./OUTCAR_prod', 'r')
            for l in outcar.readlines():
                if 'ANDERSEN_PROB =' in l:
                    prob = float(l.split()[-1])
                    if prob == 0.5:
                        has_andersen = True

        if os.path.exists('./CONTCAR_prod') and os.path.exists('./OSZICAR_prod'):
            logger.info("Previous production run output exists, check how many cylces have been run...")
            oszicar = open('./OSZICAR_prod', 'r')
            cycles_ran = 0
            for l in oszicar.readlines():
                if 'T=' in l:
                    cycles_ran += 1
            if cycles_ran >= production_set['nsw']:
                logger.info('Previous production run completed, will skip running production MD')
                run_production = False
            else:
                logger.info('Previous production run not completed, will rerun production MD')
                shutil.copy('CONTCAR_equ', 'POSCAR')
                run_production = True

        if not run_production:
            if not has_andersen:
                run_production = True
                logger.info("Previous run has not applied thermal stats, rerun production MD")

    if run_production:
        try:
            logger.info("start production run")
            vasp = Vasp(**production_set)
            vasp.set_crystal(structure)
            vasp.execute()
        except:
            pass

        if not vasp.completed:
            dav_error = False
            logfile = open('vasp.log', 'r')
            for f in logfile.readlines():
                if 'Error EDDDAV' in f:
                    dav_error = True
            if dav_error:
                production_set['algo'] = 'VeryFast'
            try:
                os.remove('./INCAR')
                logger.info("start equilibrium run ...")
                vasp = Vasp(**production_set)
                vasp.set_crystal(structure)
                vasp.execute()
            except:
                pass

        shutil.copy('POSCAR', 'POSCAR_prod')
        shutil.copy('CONTCAR', 'CONTCAR_prod')
        shutil.copy('vasprun.xml', 'vasprun_prod.xml')
        shutil.copy('OUTCAR', 'OUTCAR_prod')
        shutil.copy('OSZICAR', 'OSZICAR_prod')

    os.chdir(cwd)

def __load_supercell_structure():
    from phonopy.interface.calculator import read_crystal_structure, write_crystal_structure
    from phonopy import Phonopy
    supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    unitcell, _ = read_crystal_structure('./CONTCAR', interface_mode='vasp')
    phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix)
    write_crystal_structure('POSCAR_super', phonon.supercell, interface_mode='vasp')
    supercell = VaspReader(input_location='./POSCAR_super').read_POSCAR()
    os.remove('./POSCAR_super')
    return supercell

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='workflow control for calculating the dynamics of 2D perovskites',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--phonopy", action='store_true', help='run phonopy calculations')
    parser.add_argument("--force_rerun", action='store_true', help='force rerun  calculations')
    parser.add_argument("--MD", action='store_true', help='run MD calculations')

    args = parser.parse_args()

    if args.phonopy:
        phonopy_workflow(force_rerun=args.force_rerun)

    if args.MD:
        molecular_dynamics_workflow(force_rerun=args.force_rerun)