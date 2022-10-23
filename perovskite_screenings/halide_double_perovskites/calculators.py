from phonopy.interface.calculator import read_crystal_structure
from pymatgen.analysis.magnetism import CollinearMagneticStructureAnalyzer
from pymatgen.transformations.standard_transformations import ConventionalCellTransformation

from core.calculators.vasp import Vasp, VaspReader
from core.internal.builders.crystal import build_supercell, map_pymatgen_IStructure_to_crystal
from twodPV.calculators import default_bulk_optimisation_set, setup_logger, update_core_info, load_structure
import argparse, os, tarfile, shutil, glob


def default_symmetry_preserving_optimisation():
    # optimise the unit cell parameters whilst preserving the space and point group symmetry of the starting
    # structure.
    default_bulk_optimisation_set.update(
        {'ISIF': 7, 'Gamma_centered': True, 'NCORE': 28, 'ENCUT': 520, 'PREC': "ACCURATE", 'ispin': 2, 'IALGO': 38,
         'use_gw': True})
    structural_optimization_with_initial_magmom()


def structural_optimization_with_initial_magmom(retried=None, gamma_only=False):
    """
    Perform geometry optimization without spin polarisation. It is always helpful to converge an initial
    structure without spin polarization before further refined with a spin polarization calculations.
    This makes the SCF converge faster and less prone to cause the structural from collapsing due to problematic
    forces from unconverged SCF.
    """
    MAX_RETRY = 3
    if retried is None:
        retried = 0

    logger = setup_logger(output_filename='relax.log')

    update_core_info()
    try:
        os.remove("./WAVECAR")
        logger.info("Previous WAVECAR found, remove before start new optimisation.")
    except:
        pass

    structure = load_structure(logger)
    structure.gamma_only = gamma_only

    default_bulk_optimisation_set['magmom'], is_magnetic = magmom_string_builder(structure)

    if is_magnetic:
        default_bulk_optimisation_set['ispin'] = 2
    else:
        default_bulk_optimisation_set['ispin'] = 1
        del default_bulk_optimisation_set['magmom']

    try:
        del default_bulk_optimisation_set['magmom']
    except:
        pass
    default_bulk_optimisation_set['IALGO'] = 38
    # default_bulk_optimisation_set['ISIF'] = 3
    default_bulk_optimisation_set['use_gw'] = True
    # default_bulk_optimisation_set['ismear'] = -5
    # default_bulk_optimisation_set['sigma'] = 0.05

    logger.info("incar options" + str(default_bulk_optimisation_set))

    try:
        vasp = Vasp(**default_bulk_optimisation_set)
        vasp.set_crystal(structure)
        vasp.execute()
    except:
        vasp.completed = False
        pass

    logger.info("VASP terminated?: " + str(vasp.completed))

    # if (vasp.completed is not True) and (retried<MAX_RETRY):
    #    retried+=1
    #    structural_optimization_with_initial_magmom(retried=retried)

def static_calculation_with_SOC():
    logger = setup_logger(output_filename='soc.log')

    structure = load_structure(logger)
    structure.gamma_only = False

    _default_bulk_optimisation_set = {'ADDGRID': True,
                                      'AMIN': 0.01,
                                      'IALGO': 38,
                                      'ISMEAR': 0,
                                      'ISPIN': 2,
                                      'ISTART': 1,
                                      'ISIF': 0,
                                      'IBRION': -1,
                                      'NSW': -1,
                                      'ISYM': 0,
                                      'LCHARG': False,
                                      'LREAL': 'Auto',
                                      'LVTOT': False,
                                      'LWAVE': False,
                                      # 'NPAR': 48,
                                      'PREC': 'Normal',
                                      'SIGMA': 0.05,
                                      'ENCUT': 500,
                                      'EDIFF': '1e-04',
                                      'executable': 'vasp_ncl',
                                      'LSORBIT': True}

    default_bulk_optimisation_set = {key.lower(): value for key, value in _default_bulk_optimisation_set.items()}
    default_bulk_optimisation_set.update(
        {'Gamma_centered': True, 'NCORE': 28, 'ENCUT': 520, 'PREC': "ACCURATE", 'ispin': 2, 'IALGO': 38,'use_gw': True})

    if not os.path.exists('./SOC'):
        os.mkdir('./SOC')

    os.chdir('./SOC')


    logger.info("start equilibrium run ...")
    vasp = Vasp(**default_bulk_optimisation_set)
    vasp.set_crystal(structure)

    try:
        vasp.execute()
    except:
        pass

    os.chdir('../')

    logger.info("VASP terminated?: " + str(vasp.completed))

def magmom_string_builder(structure):
    from core.internal.builders.crystal import map_to_pymatgen_Structure
    analyzer = CollinearMagneticStructureAnalyzer(structure=map_to_pymatgen_Structure(structure), make_primitive=False,
                                                  overwrite_magmom_mode='replace_all')
    magmom_string = ""
    all_zeros = True
    for i in analyzer.magmoms:
        magmom_string += '1*' + str(i) + ' '
        if i != 0:
            all_zeros = False

    if all_zeros:
        is_magnetic = False
    else:
        is_magnetic = True
    return magmom_string, is_magnetic


def phonopy_workflow(force_rerun=False):
    from phonopy.interface.calculator import read_crystal_structure, write_crystal_structure
    from phonopy.interface.vasp import parse_set_of_forces
    from phonopy.file_IO import write_force_constants_to_hdf5, write_FORCE_SETS, parse_disp_yaml, write_disp_yaml
    from phonopy import Phonopy

    mp_points = [1, 1, 1]
    gamma_centered = True
    force_no_spin = False
    use_default_encut = False
    supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    ialgo = 38
    use_gw = True
    ncore = 32

    if mp_points != [1, 1, 1]:
        gamma_only = False
    else:
        gamma_only = True

    phonopy_set = {'prec': 'Accurate', 'ibrion': -1, 'encut': 520, 'ediff': '1e-08', 'ismear': 0, 'ialgo': ialgo,
                   'lreal': False, 'lwave': False, 'lcharg': False, 'sigma': 0.05, 'isym': 0, 'ncore': ncore,
                   'ismear': 0, 'MP_points': mp_points, 'nelm': 250, 'lreal': False, 'use_gw': use_gw,
                   'Gamma_centered': gamma_centered, 'LMAXMIX': 6, 'EDIFF': 1e-7}
    # 'amix': 0.2, 'amix_mag':0.8, 'bmix':0.0001, 'bmix_mag':0.0001}

    logger = setup_logger(output_filename='phonopy.log')
    cwd = os.getcwd()
    vasp = Vasp()
    vasp.check_convergence()
    if not vasp.completed:
        logger.exception("Initial structure optimimization failed, will not proceed!")
        raise Exception("Initial structure optimimization failed, will not proceed!")

    # check if we need to run it with spin polarizations
    f = open('./vasp.log', 'r')
    for line in f.readlines():
        if 'F=' in line:
            if 'mag' not in line:
                spin_polarized = False
            if 'mag' in line:
                magnetization = abs(float(line.split()[-1]))
                if magnetization >= 0.005:
                    spin_polarized = True
                else:
                    spin_polarized = False

    if os.path.isfile('./FORCE_SETS'):
        if not force_rerun:
            logger.info("previous phonopy calculations completed, will not rerun it again")
            return
        else:
            logger.info('previous phonopy not completed, continue')
    else:
        logger.info('NO FORCE_SETS file, previous phonon calculations crashed, try to :rerun')

    # if os.path.isfile('phonopy.log'):
    #     success = []
    #     for l in open('phonopy.log', 'r').readlines():
    #         if 'VASP calculation completed successfully? ' in l:
    #             if l.split()[-1] == "True":
    #                 success.append(True)
    #             elif l.split()[-1] == 'False':
    #                 success.append(False)
    #     if len(success) != 0:
    #         if not all(success):
    #             if not force_rerun:
    #                 logger.exception("Encounter convergence problems with VASP before, will not attempt again.")
    #                 raise Exception("Encounter convergence problems with VASP before, will not attempt again.")
    #             else:
    #                 logger.info("try to rerun all VASP calculations")
    #                 print("try to rerun all VASP calculations with RMM for ialgo")

    try:
        unitcell, _ = read_crystal_structure('./CONTCAR', interface_mode='vasp')
    except:
        raise Exception("No CONTCAR!")

    if not force_rerun:
        if not os.path.exists('./phonon'):
            os.mkdir('./phonon')
        elif os.path.isfile("./phonon.tar.gz"):
            tf = tarfile.open("./phonon.tar.gz")
            tf.extractall()
    else:
        try:
            shutil.rmtree('./phonon')
        except:
            pass
        try:
            os.rmtree('./phonon')
        except:
            pass
        os.mkdir('./phonon')

        try:
            os.remove("./phonon.tar.gz")
        except:
            pass

    os.chdir('./phonon')

    phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix)
    phonon.generate_displacements()

    supercells = phonon.supercells_with_displacements
    write_crystal_structure('SPOSCAR', phonon.supercell)
    write_disp_yaml(displacements=phonon.displacements, supercell=phonon.supercell, filename='disp.yaml')

    logger.info("Will phonon calculations be run with spin polarisation? " + str(spin_polarized))
    if spin_polarized:
        crystal = VaspReader(input_location='./SPOSCAR').read_POSCAR()
        phonopy_set['magmom'], all_zeros = magmom_string_builder(crystal)
        phonopy_set['ispin'] = 2

    logger.info('Force everything to be spin unpolarised calculation, avoid problem of electronic convergence')
    phonopy_set['ispin'] = 1

    logger.info(
        "PHONOPY - generate (2x2x2) displaced supercells, total number of configurations " + str(len(supercells)))

    completed = []
    force_files = []

    calculate_next = True
    for i, sc in enumerate(supercells):
        phonopy_set['ialgo'] = 38

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
                    # if not force_rerun:
                    # calculate_next = False
                    # proceed = False
                    # logger.info('At least one finite displaced configuration cannot converge, quit the rest')

            if proceed:
                logger.info("Configuration " + str(i + 1) + '/' + str(len(supercells)) + " proceed VASP calculation")
                write_crystal_structure('POSCAR', sc, interface_mode='vasp')
                structure = load_structure(logger)
                structure.gamma_only = gamma_only
                # phonopy_set['magmom'], all_zeros = magmom_string_builder(structure)
                # phonopy_set['ispin'] = 2

                try:
                    vasp = Vasp(**phonopy_set)
                    vasp.set_crystal(structure)
                    vasp.execute()
                except:
                    pass

                if vasp.completed is not True:
                    phonopy_set['ialgo'] = 48
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
        structure = VaspReader(input_location='./SPOSCAR').read_POSCAR()
        set_of_forces = parse_set_of_forces(structure.total_num_atoms(), force_files)

        phonon.set_forces(sets_of_forces=set_of_forces)
        phonon.produce_force_constants()
        write_force_constants_to_hdf5(phonon.force_constants, filename='force_constants.hdf5')

        displacements = parse_disp_yaml(filename='disp.yaml')
        num_atoms = displacements['natom']
        for forces, disp in zip(set_of_forces, displacements['first_atoms']):
            disp['forces'] = forces
        write_FORCE_SETS(displacements, filename='FORCE_SETS')

        if os.path.isfile('force_constants.hdf5'):
            shutil.copy('./force_constants.hdf5', '../force_constants.hdf5')

        if os.path.isfile('FORCE_SETS'):
            shutil.copy('./FORCE_SETS', '../FORCE_SETS')

    os.chdir('..')

    output_filename = 'phonon.tar.gz'
    source_dir = './phonon'
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

    try:
        shutil.rmtree('./phonon')
    except:
        pass
    try:
        os.rmtree('./phonon')
    except:
        pass

    os.chdir(cwd)


def clean_up_phonon():
    from phonopy.interface.calculator import read_crystal_structure, write_crystal_structure
    from phonopy.interface.vasp import parse_set_of_forces
    from phonopy.file_IO import write_force_constants_to_hdf5, write_FORCE_SETS, parse_disp_yaml, write_disp_yaml
    from phonopy import Phonopy
    from glob import glob

    if os.path.isfile('./FORCE_SETS') and os.path.isfile('./force_constants.hdf5') and os.path.isfile('phonon.tar.gz'):
        return

    if not os.path.isfile('./FORCE_SETS') and os.path.isdir('./phonon'):
        print('here')
        unitcell, _ = read_crystal_structure('./CONTCAR', interface_mode='vasp')

        # write out the supercell structures used for phonon calculations
        supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix)
        phonon.generate_displacements()
        write_crystal_structure('SPOSCAR', phonon.supercell)

        structure = VaspReader(input_location='./SPOSCAR').read_POSCAR()

        os.chdir('phonon')
        write_disp_yaml(displacements=phonon.displacements, supercell=phonon.supercell, filename='disp.yaml')

        try:
            all_dir = glob("ph-*/")
            force_files = ['./ph-POSCAR-' + str(i) + '/vasprun.xml' for i in range(len(all_dir))]
            set_of_forces = parse_set_of_forces(structure.total_num_atoms(), force_files)
            phonon.set_forces(sets_of_forces=set_of_forces)
            phonon.produce_force_constants()
            displacements = parse_disp_yaml(filename='disp.yaml')
            for forces, disp in zip(set_of_forces, displacements['first_atoms']):
                disp['forces'] = forces
            write_FORCE_SETS(displacements, filename='FORCE_SETS')
        except:
            pass

        if os.path.isfile('force_constants.hdf5'):
            shutil.copy('./force_constants.hdf5', '../force_constants.hdf5')

        if os.path.isfile('FORCE_SETS'):
            shutil.copy('./FORCE_SETS', '../FORCE_SETS')

        os.chdir('..')

        output_filename = 'phonon.tar.gz'
        source_dir = './phonon'
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))

        try:
            shutil.rmtree('./phonon')
        except:
            pass
        try:
            os.rmtree('./phonon')
        except:
            pass


def molecular_dynamics_workflow(force_rerun=False, continue_MD=True):
    logger = setup_logger(output_filename='molecular_dynamics.log')
    cwd = os.getcwd()

    logger.info(
        "Setting up the room temperature molecular dynamics calculations, check if we have previous phonon data")

    if not os.path.isfile('./force_constants.hdf5'):
        logger.info("previous phonopy calculations did not complete properly, will not proceed...")
        raise Exception("No phonopy data! QUIT")
    else:
        spin_polarized = check_phonon_run_settings()
        structure = load_supercell_structure()
        structure.gamma_only = True
        logger.info("Will run MD with spin polarization: " + str(spin_polarized))

    equilibrium_set = {'prec': 'Accurate', 'algo': 'Normal', 'lreal': 'AUTO', 'ismear': 0, 'isym': 0, 'ibrion': 0,
                       'maxmix': 40,
                       'lmaxmix': 6, 'ncore': 28, 'nelmin': 4, 'nsw': 100, 'smass': -1, 'isif': 1, 'tebeg': 10,
                       'teend': 300, 'potim': 1, 'nblock': 10, 'nwrite': 0, 'lcharg': False, 'lwave': False,
                       'iwavpr': 11, 'encut': 520, 'Gamma_centered': True, 'MP_points': [1, 1, 1], 'use_gw': True,
                       'write_poscar': True}

    production_set = {'prec': 'Accurate', 'algo': 'Normal', 'lreal': 'AUTO', 'ismear': 0, 'isym': 0, 'ibrion': 0,
                      'maxmix': 40,
                      'lmaxmix': 6, 'ncore': 28, 'nelmin': 4, 'nsw': 2000, 'isif': 1, 'tebeg': 300,
                      'teend': 300, 'potim': 1, 'nblock': 1, 'nwrite': 0, 'lcharg': False, 'lwave': False, 'iwavpr': 11,
                      'encut': 520, 'andersen_prob': 0.5, 'mdalgo': 1, 'Gamma_centered': True, 'MP_points': [1, 1, 1],
                      'use_gw': True, 'write_poscar': False}

    del equilibrium_set['ncore']
    del production_set['ncore']

    # if spin_polarized:
    #    raise Exception("Skip running spin polarized MD first ...")

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

        # Check if the previous run is also ran with spin-polarisation settings as for the phonon calculations
        md_spin_polarization = check_MD_run_settings()
        if md_spin_polarization != spin_polarized:
            logger.info(
                "Previous MD was not run with the same spin polarization setting as for Phonopy, will rerun with " + str(
                    spin_polarized))
            run_equilibration = True
            run_production = True

    if spin_polarized:
        equilibrium_set['ispin'] = 2
        equilibrium_set['magmom'], _ = magmom_string_builder(structure)
        production_set['ispin'] = 2
        production_set['magmom'], _ = magmom_string_builder(structure)
    else:
        equilibrium_set['ispin'] = 1
        production_set['ispin'] = 1

    if run_equilibration:
        if continue_MD:
            logger.warning(
                "Only trying to continue a production MD, but there seems to be incomplete equilibration, will not continue")
            os.chdir(cwd)
            return
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
        if not continue_MD:
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
                logger.info("Previous production run output exists, check how many cycles have been run...")
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
        else:
            run_production = True

    previous_productions = len(glob.glob('CONTCAR_prod*'))
    logger.info('Number of previous production MD runs:' + str(previous_productions))

    if previous_productions >= 3:
        run_production = False

    if run_production:

        if continue_MD:
            if previous_productions == 1:
                shutil.copy('CONTCAR_prod', 'POSCAR')
            elif previous_productions > 1:
                shutil.copy('CONTCAR_prod_' + str(previous_productions - 1), 'POSCAR')

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

        if previous_productions == 0:
            shutil.copy('POSCAR', 'POSCAR_prod')
            shutil.copy('CONTCAR', 'CONTCAR_prod')
            shutil.copy('vasprun.xml', 'vasprun_prod.xml')
            shutil.copy('OUTCAR', 'OUTCAR_prod')
            shutil.copy('OSZICAR', 'OSZICAR_prod')
        elif previous_productions > 0:
            shutil.copy('POSCAR', 'POSCAR_prod_' + str(previous_productions))
            shutil.copy('CONTCAR', 'CONTCAR_prod_' + str(previous_productions))
            shutil.copy('vasprun.xml', 'vasprun_prod_' + str(previous_productions) + '.xml')
            shutil.copy('OUTCAR', 'OUTCAR_prod_' + str(previous_productions))
            shutil.copy('OSZICAR', 'OSZICAR_prod_' + str(previous_productions))

    os.chdir(cwd)


def check_phonon_run_settings():
    spin_polarized = False
    if os.path.exists('./phonon'):
        os.chdir('./phonon')
        f = open('./ph-POSCAR-0/vasp.log', 'r')
        for line in f.readlines():
            if 'F=' in line:
                if 'mag' not in line:
                    spin_polarized = False
                if 'mag' in line:
                    magnetization = abs(float(line.split()[-1]))
                    if magnetization >= 0.05:
                        spin_polarized = True
                    else:
                        spin_polarized = False
        os.chdir('..')
    return spin_polarized


def check_MD_run_settings():
    spin_polarized = False
    if os.path.isfile('OUTCAR_prod'):
        output = 'OUTCAR_prod'
    elif os.path.isfile('OUTCAR_equ'):
        output = 'OUTCAR_equ'
    else:
        return spin_polarized
    f = open(output, 'r')
    for l in f.readlines():
        if 'ISPIN' in l:
            ispin = int(l.split()[2])
    if ispin == 1:
        return False
    elif ispin == 2:
        return True


def load_supercell_structure(supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]]):
    from phonopy.interface.calculator import read_crystal_structure, write_crystal_structure
    from phonopy import Phonopy
    supercell_matrix = supercell_matrix
    unitcell, _ = read_crystal_structure('./CONTCAR', interface_mode='vasp')
    phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix)
    write_crystal_structure('POSCAR_super', phonon.supercell, interface_mode='vasp')
    supercell = VaspReader(input_location='./POSCAR_super').read_POSCAR()
    os.remove('./POSCAR_super')
    return supercell

def rapid_quench_from_MD(part=0,batch_size=50):
    from pymatgen.io.vasp.outputs import Vasprun
    from core.utils.zipdir import ZipDir

    vasprun = Vasprun('vasprun_prod.xml')
    trajectory = vasprun.get_trajectory()
    all_structures = [trajectory.get_structure(i) for i in range(len(trajectory.frac_coords))]

    if (part + 1) * batch_size > len(all_structures):
        raise Exception("over the length of the trajectory, quit")

    for i in range(batch_size * part, batch_size * (part + 1)):
        pwd = os.getcwd()
        folder = 'frame_' + str(i)

        if os.path.exists(pwd + '/' + folder + '.zip'): continue

        try:
            os.mkdir(folder)
        except:
            pass
        os.chdir(pwd + '/' + folder)

        this_structure = map_pymatgen_IStructure_to_crystal(all_structures[i])
        this_structure.gamma_only = True  # DO NOT DELETE THIS!!!

        optimisation_set = {'ISPIN': 1, 'PREC': "Normal", 'IALGO': 38, 'NPAR': 28, 'ENCUT': 300,  'ISIF':0, 'ibrion':2,
                         'LCHARG': False, 'LWAVE': False, 'use_gw': True, 'Gamma_centered': True, 'MP_points': [1, 1, 1],
                         'clean_after_success': True, 'LREAL': 'False', 'executable': 'vasp_gam', 'NSW':500}

        vasp = Vasp(**optimisation_set)
        vasp.set_crystal(this_structure)
        vasp.execute()

        files = ['CHG', 'CHGCAR', 'LOCPOT', 'EIGENVAL', 'IBZKPT', 'PCDAT', 'POTCAR', 'WAVECAR', 'DOSCAR',
                 'OUTCAR', 'PROCAR', 'KPOINTS']
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass

        os.chdir(pwd)

        ZipDir(folder, folder + '.zip')
        shutil.rmtree(folder, ignore_errors=True)


def run_electronic_dos_for_md_trajectory(part=0, batch_size=20):
    # load all the production run xml file
    from pymatgen.io.vasp.outputs import Vasprun
    from core.utils.zipdir import ZipDir
    all_vasp_runs = []
    all_structures = []
    for v in ['vasprun_prod.xml', 'vasprun_prod_1.xml', 'vasprun_prod_2.xml']:
        if os.path.exists('./' + v):
            all_vasp_runs.append(v)

    for v in all_vasp_runs:
        vasprun = Vasprun(v)
        trajectory = vasprun.get_trajectory()
        all_structures += [trajectory.get_structure(i) for i in range(len(trajectory.frac_coords))]

    if (part + 1) * batch_size > len(all_structures):
        raise Exception("over the length of the trajectory, quit")

    for i in range(batch_size * part, batch_size * (part + 1)):

        pwd = os.getcwd()
        folder = 'frame_' + str(i)

        if os.path.exists(pwd + '/' + folder + '.zip'): continue

        try:
            os.mkdir(folder)
        except:
            pass
        os.chdir(pwd + '/' + folder)

        this_structure = map_pymatgen_IStructure_to_crystal(all_structures[i])
        this_structure.gamma_only = True #DO NOT DELETE THIS!!!

        single_pt_set = {'ISPIN': 1, 'PREC': "Normal", 'IALGO': 38, 'NPAR': 28, 'ENCUT': 350, 'PRECFOCK':'Fast',
                         'LCHARG': True, 'LWAVE': True, 'use_gw': True, 'Gamma_centered': True, 'MP_points': [1, 1, 1],
                         'clean_after_success': False, 'LREAL': 'False', 'executable':'vasp_gam'}
        vasp = Vasp(**single_pt_set)
        vasp.set_crystal(this_structure)
        vasp.execute()

        gga_gap = None
        hybrid_gap = None

        if vasp.completed:
            shutil.copy('OUTCAR','OUTCAR_GGA')
            gga_gap = get_dos_gap()
            shutil.copy('vasprun.xml', 'vasprun_gga.xml')
            del single_pt_set['IALGO']
            single_pt_set.update(
                {'LCHARG': False, 'LVTOT': False, 'LWAVE': False, 'NELM': 200, 'ISPIN': 1, 'ENCUT': 350,
                 'LHFCALC': True, 'HFSCREEN': 0.2, 'PRECFOCK': 'FAST', 'Gamma_centered': True, 'MP_points': [1, 1, 1],
                 'use_gw': True, 'clean_after_success': True, 'write_poscar': False, 'PREC': "Normal"})
            vasp = Vasp(**single_pt_set)
            vasp.set_crystal(this_structure)
            vasp.execute()
            if vasp.completed:
                shutil.copy('OUTCAR', 'OUTCAR_hybrid')
                hybrid_gap = get_dos_gap()
                shutil.copy('vasprun.xml', 'vasprun_hybrid.xml')

        files = ['CHG', 'CHGCAR', 'LOCPOT', 'EIGENVAL', 'IBZKPT', 'PCDAT', 'POTCAR', 'WAVECAR', "vasprun.xml", 'DOSCAR', 'OUTCAR',
                 'PROCAR', 'KPOINTS']
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass

        os.chdir(pwd)

        ZipDir(folder, folder + '.zip')
        shutil.rmtree(folder, ignore_errors=True)

        output = open(pwd + '/gap_dynamics_300K_' + str(part) + '.dat', 'a+')
        output.write(str(i) + '\t' + "{:.4f}".format(gga_gap) + '\t' + "{:.4f}".format(hybrid_gap) + '\n')
        output.close()

def get_dos_gap():
    f=open('OUTCAR','r')
    start_collect = False
    cbm = None
    for line in f.readlines():
        if not start_collect:
            if ' band No.  band energies     occupation' in line:
                start_collect = True
                continue
        if start_collect:
            s=line.split()
            if len(s) == 3:
                if float(s[-1])>0.0: vbm = float(s[1])
                if (float(s[-1])==0.0) and (cbm is None): cbm = float(s[1])
            else:
                break
    gap = cbm-vbm
    print('band gap is :',gap)
    return gap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='workflow control for double perovskite ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--opt", action='store_true', help='perform initial structural optimization')
    parser.add_argument("--phonopy", action='store_true', help='run phonopy calculations')
    parser.add_argument("--force_rerun", action='store_true', help='force rerun  calculations')
    parser.add_argument("--continue_MD", action='store_true', help='continue running MD')
    parser.add_argument("--MD", action='store_true', help='run MD calculations')
    parser.add_argument("--clean_phonon", action='store_true', help='clean up the phonon calculation')
    parser.add_argument("--electdyn", action='store_true')
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--get_gap", action='store_true', default=0)
    parser.add_argument("--quench", action='store_true')
    parser.add_argument("--soc", action='store_true')
    args = parser.parse_args()

    if args.opt:
        default_symmetry_preserving_optimisation()

    if args.phonopy:
        phonopy_workflow(force_rerun=args.force_rerun)

    if args.MD:
        molecular_dynamics_workflow(force_rerun=args.force_rerun, continue_MD=args.continue_MD)

    if args.clean_phonon:
        clean_up_phonon()

    if args.electdyn:
        run_electronic_dos_for_md_trajectory(part=args.part)

    if args.quench:
        rapid_quench_from_MD(part=args.part)

    if args.get_gap:
        get_dos_gap()

    if args.soc:
        static_calculation_with_SOC()