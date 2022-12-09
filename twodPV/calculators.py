import os
import shutil

from pymatgen.analysis.magnetism import CollinearMagneticStructureAnalyzer

from core.calculators.vasp import Vasp
from core.dao.vasp import *
from core.utils.loggings import setup_logger
from pymatgen.io.vasp.outputs import Vasprun

# we set the default calculation to be spin-polarized.
_default_bulk_optimisation_set = {'ADDGRID': True,
                                  'AMIN': 0.01,
                                  'IALGO': 38,
                                  'ISMEAR': 0,
                                  'ISPIN': 2,
                                  'ISTART': 1,
                                  'ISIF': 3,
                                  'IBRION': 2,
                                  'NSW': 500,
                                  'ISYM': 0,
                                  'LCHARG': False,
                                  'LREAL': 'Auto',
                                  'LVTOT': False,
                                  'LWAVE': False,
                                  # 'NPAR': 48,
                                  'PREC': 'Normal',
                                  'SIGMA': 0.05,
                                  'SIGMA': 0.05,
                                  'ENCUT': 500,
                                  'EDIFF': '1e-04',
                                  # 'NPAR': 7,
                                  'executable': 'vasp_std'}

default_bulk_optimisation_set = {key.lower(): value for key, value in _default_bulk_optimisation_set.items()}


def update_core_info():
    try:
        ncpus = None
        f = open('node_info', 'r')
        for l in f.readlines():
            if 'normalbw' in l:
                ncpus = 28
            elif 'normalsl' in l:
                ncpus = 32
            else:
                ncpus = 12
        default_bulk_optimisation_set.update({'NPAR': ncpus, 'NCORE': 3})
    except:
        pass


def default_run_with_existing_vasp_setup():
    """
    Run vasp calculation in folders where all POSCAR, INCAR, POTCAR and KPOINTs are already in place
    """
    logger = setup_logger(output_filename='relax.log')
    update_core_info()
    logger.info("==========Run Vasp with pre-existing setup==========")
    try:
        os.remove("./WAVECAR")
        logger.info("Previous WAVECAR found, remove before start new optimisation.")
    except:
        pass

    if os.path.isfile('./CONTCAR') and (os.path.getsize('./CONTCAR') > 0):
        import shutil
        shutil.copy('./CONTCAR', './POSCAR')
        logger.info("Restart optimisation from previous CONTCAR.")
    else:
        logger.info("Start new optimisation from POSCAR")

    vasp = Vasp(**default_bulk_optimisation_set)
    vasp.run()
    vasp.check_convergence()

    if vasp.completed:
        vasp.tear_down()


def default_structural_optimisation():
    """
    Perform a full geometry optimization on the structure in POSCAR stored in the current folder (supposed named as opt/).
    To submit this optimisation job using the command line argument from myqueue package, do

    mq submit twodPV.calculators@default_structural_optimisation -R <resources> opt/

    Note that this method is implemented in such a way that an existing CONTCAR will be read first, if it can be found,
    otherwise, it will read in the POSCAR file. So restart mechansim is already built in. For example, if a VASP calculation
    is timeout in the opt/ folder with ID, running the following command

    mq resubmit -i ID

    should resubmit a job continuing the previous unfinished structural optimisation.
    """

    logger = setup_logger(output_filename='relax.log')
    update_core_info()
    logger.info("==========Full Structure Optimisation with VASP==========")
    structure = load_structure(logger)

    # default_bulk_optimisation_set.update({"MAGMOM": "5*0 11*0 16*4 48*0"})
    __default_spin_polarised_vasp_optimisation_procedure(logger, structure)


def __default_spin_polarised_vasp_optimisation_procedure(logger, structure):
    try:
        os.remove("./WAVECAR")
        logger.info("Previous WAVECAR found, remove before start new optimisation.")
    except:
        pass

    # vasp = Vasp(**default_bulk_optimisation_set)
    # vasp.set_crystal(structure)
    # vasp.execute()
    if True:
        # if vasp.self_consistency_error:
        # Spin polarisation calculations might be very difficult to converge.
        # For this case, we converge a non-spin polarisation calculation first and
        # then use the converged wavefunction to carry out spin-polarised optimisation
        logger.info(
            "Spin-polarised SCF convergence failed, try generate a non-spin-polarised wavefunction as starting guess")

        default_bulk_optimisation_set.update(
            {'Gamma_centered': True, 'MP_points': [1, 1, 1], 'executable': 'vasp_gam', 'gpu_run': False})
        structure.gamma_only = True

        default_bulk_optimisation_set.update({'ISPIN': 1, 'NSW': 500, 'LWAVE': True, 'clean_after_success': False})

        vasp = Vasp(**default_bulk_optimisation_set)
        vasp.set_crystal(structure)
        vasp.execute()

        if os.path.isfile('./WAVECAR') and (os.path.getsize('./WAVECAR') > 0):
            logger.info("WAVECAR found")
            logger.info("Restart spin-polarised structure relaxation...")
            structure = VaspReader(input_location='./CONTCAR').read_POSCAR()
            default_bulk_optimisation_set.update(
                {'ISPIN': 2, 'NSW': 500, 'LWAVE': False, 'clean_after_success': True, 'write_poscar': True})
            vasp = Vasp(**default_bulk_optimisation_set)
            vasp.set_crystal(structure)
            vasp.execute()

            logger.info("VASP terminated properly: " + str(vasp.completed))
            if not vasp.completed:
                raise Exception("VASP did not completed properly, you might want to check it by hand.")


def ionic_optimisation():
    logger = setup_logger(output_filename='relax.log')
    update_core_info()
    logger.info("==========Full Structure Optimisation with VASP==========")

    default_bulk_optimisation_set.update(
        {'ISPIN': 2, 'NSW': 500, 'LWAVE': False, 'clean_after_success': True, 'Gamma_centered': True,
         'MP_points': [4, 4, 1], 'executable': 'vasp_std', 'gpu_run': False, 'IBRION': 2, 'ISIF': 0})
    structure = load_structure(logger)
    vasp = Vasp(**default_bulk_optimisation_set)
    vasp.set_crystal(structure)
    vasp.execute()
    logger.info("VASP terminated properly: " + str(vasp.completed))
    if not vasp.completed:
        raise Exception("VASP did not completed properly, you might want to check it by hand.")


def single_point_calculation():
    logger = setup_logger(output_filename='single_point.log')
    update_core_info()

    logger.info("==========Full Structure Optimisation with VASP==========")
    try:
        os.remove('./INCAR')
    except:
        pass

    default_bulk_optimisation_set.update(
        {'ISPIN': 2, 'NSW': 500, 'LWAVE': False, 'clean_after_success': True, 'Gamma_centered': True,
         'MP_points': [4, 4, 1], 'executable': 'vasp_std', 'gpu_run': False, 'IBRION': -1, 'ISIF': 0, 'NSW':0, 'IALGO':38,'NELM':150})
    structure = load_structure(logger)
    vasp = Vasp(**default_bulk_optimisation_set)
    vasp.set_crystal(structure)
    vasp.execute()
    logger.info("VASP terminated properly: " + str(vasp.completed))
    if not vasp.completed:
        raise Exception("VASP did not completed properly, you might want to check it by hand.")

def load_structure(logger):
    if os.path.isfile('./CONTCAR') and (os.path.getsize('./CONTCAR') > 0):
        structure = VaspReader(input_location='./CONTCAR').read_POSCAR()
        logger.info("Restart optimisation from previous CONTCAR.")
    else:
        structure = VaspReader(input_location='./POSCAR').read_POSCAR()
        logger.info("Start new optimisation from POSCAR")
    return structure


def spin_unpolarised_optimization():
    """
    Perform geometry optimization without spin polarisation. It is always helpful to converge an initial
    structure without spin polarization before further refined with a spin polarization calculations.
    This makes the SCF converge faster and less prone to cause the structural from collapsing due to problematic
    forces from unconverged SCF.
    """

    logger = setup_logger(output_filename='relax.log')

    update_core_info()
    try:
        os.remove("./WAVECAR")
        logger.info("Previous WAVECAR found, remove before start new optimisation.")
    except:
        pass

    logger.info("==========Full Structure Optimisation with VASP==========")

    structure = load_structure(logger)

    logger.info("Perform an"
                " initial spin-non-polarised calculations to help convergence")
    # default_bulk_optimisation_set.update({'ispin': 1, 'nsw': 500, 'ENCUT': 300, 'EDIFF': '1e-04','MP_points':[4,4,1],'Gamma_centered': True, 'NCORE':28, 'KPAR':28, })#'executable':'vasp_gam'})

    logger.info("incar options" + str(default_bulk_optimisation_set))

    try:
        vasp = Vasp(**default_bulk_optimisation_set)
        vasp.set_crystal(structure)
        vasp.execute()
    except:
        vasp.completed = False
        pass

    logger.info("VASP terminated?: " + str(vasp.completed))


def default_two_d_optimisation():
    # Method to be called for optimising a single 2D slab, where the lattice parameters in the
    # xy-plane (parallel to the 2D material will be optimised) while keeping z-direction fixed.
    # this can be achieved by using a specific vasp executable.
    default_bulk_optimisation_set.update(
        {'executable': 'vasp_std-xy', 'MP_points': [4, 4, 1], 'idipol': 3})
    default_structural_optimisation()


def spin_unploarised_two_d_optimisation():
    default_bulk_optimisation_set.update(
        {'executable': 'vasp_std-xy', 'MP_points': [4, 4, 1], 'nsw': 500, 'ENCUT': 300, 'EDIFF': '1e-04', 'idipol': 3,
         'ispin': 1, 'NCORE': 28, 'KPAR': 28})
    spin_unpolarised_optimization()


def default_symmetry_preserving_optimisation():
    # optimise the unit cell parameters whilst preserving the space and point group symmetry of the starting
    # structure.
    default_bulk_optimisation_set.update({'ISIF': 7, 'MP_points': [6, 6, 6]})
    default_structural_optimisation()


def default_bulk_phonon_G_calculation():
    return __default_G_phonon(two_d=False)


def default_twod_phonon_G_calculation():
    return __default_G_phonon(two_d=True)


def __default_G_phonon(two_d=False):
    logger = setup_logger(output_filename='phonon.log')

    try:
        os.remove("./WAVECAR")
        logger.info("Previous WAVECAR found, remove before start new optimisation.")
    except:
        pass

    if two_d:
        kpoints = [4, 4, 1]
    else:
        kpoints = [6, 6, 6]
    """
    default_bulk_optimisation_set.update(
        {'PREC': 'Accurate',
         'ISPIN': 1,
         'NSW': 0,
         'LWAVE': True,
         'ISYM': 0,
         'MP_points': kpoints,
         'NELM':120,
         'clean_after_success': False})

    if two_d:
        default_bulk_optimisation_set.update({'idipol': 3})

    __G_phonon()
    """
    default_bulk_optimisation_set.update(
        {'ISPIN': 1,
         'LWAVE': False,
         'NSW': 1,
         'PREC': 'Accurate',
         'EDIFF': 1e-05,
         'IBRION': 8,
         'ISIF': 0,
         'ISYM': 0,
         'LREAL': 'Auto',
         'POTIM': 0.01,
         "NELM": 200,
         'clean_after_success': True})
    __G_phonon()


def __G_phonon():
    update_core_info()
    logger.info("==========Gamma point phonon calculation with VASP==========")
    structure = VaspReader(input_location='./POSCAR').read_POSCAR()
    logger.info("Start from supercell defined in POSCAR")
    vasp = Vasp(**default_bulk_optimisation_set)
    vasp.set_crystal(structure)
    vasp.execute()
    logger.info("VASP terminated properly: " + str(vasp.completed))


# =================================================================
# Procedures for calculating frequency dependent dielectric
# constants
# =================================================================
single_point_pbe = {'PREC': 'HIGH',
                    'ISMEAR': 0,
                    'SIGMA': 0.01,
                    'EDIFF': 1e-05,
                    'IALGO': 38,
                    'ISPIN': 1,
                    'NELM': 500,
                    'AMIN': 0.01,
                    'ISYM': 0,
                    'PREC': 'HIGH',
                    'ENCUT': 300,
                    'NSW': 0,
                    'LWAVE': True,
                    'LVTOT': True,
                    'clean_after_success': False,
                    'use_gw': True,
                    'MP_points': [4, 4, 1],
                    'Gamma_centered': True}

pbe_omega = {'PREC': 'HIGH',
             'ISMEAR': 0,
             'SIGMA': 0.01,
             'EDIFF': 1e-05,
             'AMIN': 0.01,
             'ALGO': 'EXACT',
             'LOPTICS': True,
             'NELM': 1,
             'OMEGAMAX': 40,
             'ISPIN': 1,
             'ISYM': 0,
             'LPEAD': True,
             'ENCUT': 500,
             'NEDOS': 1000,
             'clean_after_success': False,
             'use_gw': True,
             'MP_points': [4, 4, 1],
             'Gamma_centered': True}

pbe_rpa_omega = {'PREC': 'HIGH',
                 'ISMEAR': 0,
                 'SIGMA': 0.01,
                 'EDIFF': 1e-05,
                 'AMIN': 0.01,
                 'ALGO': 'CHI',
                 'LRPA': True,  # this option turns off the exchange-correlation kernel
                 'LOPTICS': True,
                 'NELM': 1,
                 'OMEGAMAX': 10,
                 'ISPIN': 1,
                 'ISYM': 0,
                 'ENCUT': 500,
                 'ENCUTGW': 100,
                 'NEDOS': 1000,
                 'clean_after_success': True,
                 'use_gw': True,
                 'MP_points': [4, 4, 1],
                 'Gamma_centered': True}

hse06_set = {'LHFCALC': True,
             'HFSCREEN': 0.2,
             'PRECFOCK': 'Normal',
             'ICHARG': 12}  # Non self-consistent HSE06


def rpa_dielectric_constants_pbe():
    return rpa_dielectric_constants(hybrid_GGA=False)


def rpa_dielectric_constants_hse06():
    return rpa_dielectric_constants(hybrid_GGA=True)


def rpa_dielectric_constants(hybrid_GGA=False):
    directory = 'electronic_hybrid'
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.chdir(directory)

    shutil.copy('../CONTCAR', 'POSCAR')

    logger = setup_logger(output_filename="dielectrics.log")
    single_pt_set = {'NCORE': 28, 'NPAR': 4, 'ENCUT': 350, 'ISPIN': 1, 'PREC': "Normal", 'IALGO': 38}
    structure = VaspReader(input_location='./POSCAR').read_POSCAR()
    logger.info("Starting from structure in POSCAR in " + os.getcwd())

    try:
        os.remove("./WAVECAR")
        logger.info("Previous WAVECAR found, remove before new calculation.")
        os.remove("./WAVEDER")
        logger.info("Previous WAVEDER found, remove before new calculation.")
    except:
        pass

    # =======================================
    # stage I - ground state PBE calculation
    # =======================================
    logger.info("PBE self-consistent run")
    logger.info("INCAR settings: ")
    for k in single_pt_set.keys():
        logger.info("      " + str(k) + "=" + str(single_pt_set[k]))

    single_pt_set.update(single_point_pbe)
    # single_pt_set['IALGO']=48
    vasp = Vasp(**single_pt_set)
    vasp.set_crystal(structure)
    vasp.execute()

    if vasp.completed:
        logger.info("PBE self-consistent run completed properly.")
    else:
        single_pt_set = {'NCORE': 28, 'NPAR': 4, "ENCUT": 350, 'ISPIN': 1, 'PREC': "Normal", 'IALGO': 38}
        single_pt_set.update(single_point_pbe)
        single_pt_set['IALGO'] = 38
        structure = VaspReader(input_location='./POSCAR').read_POSCAR()
        logger.info("try again with a different SCF optimisation routine")
        vasp = Vasp(**single_pt_set)
        vasp.set_crystal(structure)
        vasp.execute()
        if vasp.completed:
            logger.info("PBE self-consistent run completed properly.")
        else:
            files = ['CHG', 'CHGCAR', 'EIGENVAL', 'IBZKPT', 'PCDAT', 'POTCAR', 'WAVECAR', 'LOCPOT', 'node_info',
                     "WAVECAR", "WAVEDER", 'DOSCAR', 'PROCAR']
            for f in files:
                try:
                    os.remove(f)
                except OSError:
                    pass

            os.chdir('../')
            from core.utils.zipdir import ZipDir
            ZipDir('electronic_hybrid', 'electronic_hybrid.zip')
            shutil.rmtree('./electronic_hybrid')
            shutil.copy('vasprun.xml', 'vasprun.PBE.xml')
            raise Exception("PBE self-consistent run failed to converge, will stop proceeding")

    # check if this is a semiconductor, if not quit
    semiconductor = False
    dos_run = Vasprun("./vasprun.xml")
    dos = dos_run.complete_dos
    for tol in range(2000):
        t = 0.001 + tol * 0.01
        gap = dos.get_gap(tol=tol)
        if gap > 0.2:
            semiconductor = True
            break

    if not semiconductor:
        logger.info("Band gap too small, most likely a metal, will not process further")
        files = ['CHG', 'CHGCAR', 'EIGENVAL', 'IBZKPT', 'PCDAT', 'POTCAR', 'WAVECAR', 'node_info', "WAVECAR",
                 "WAVEDER", 'DOSCAR', 'PROCAR']
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass
        os.chdir('../')
        from core.utils.zipdir import ZipDir
        ZipDir('electronic_hybrid', 'electronic_hybrid.zip')
        shutil.rmtree('./electronic_hybrid')
        return

    if hybrid_GGA:
        # ==================================================
        # Ground state hybrid-GGA calculation
        # ==================================================
        logger.info('Hybrid GGA (HSE06) self-consistent run')
        single_pt_set.update(hse06_set)
        single_pt_set.update(
            {'ALGO': 'ALL', 'LVHAR': True, 'ISPIN': 1, 'ICHRG': 1, 'NELM': 80, 'LCHARG': True, "ENCUT": 350})
        vasp = Vasp(**single_pt_set)
        vasp.set_crystal(structure)
        vasp.execute()

        for k in single_pt_set.keys():
            logger.info("      " + str(k) + "=" + str(single_pt_set[k]))

        if vasp.completed:
            logger.info("HSE06 self-consistent run completed properly.")
            # copy the files across for debugging or data grabbing
            shutil.copy('OSZICAR', 'OSZICAR.HYBRID')
            shutil.copy('OUTCAR', 'OUTCAR.HYBRID')
            shutil.copy('vasprun.xml', 'vasprun.HYBRID.xml')
            shutil.copy('CHGCAR', 'CHGCAR.HYBRID')
            shutil.copy('LOCPOT', 'LOCPOT.HYBRID')
        else:
            logger.info("HSE06 self-consistent run failed to converge, will not store results")
            os.remove('OSZICAR')
            os.remove('OUTCAR')
            os.remove('vasprun.xml')

        logger.info("Clean Up calculations")
        files = ['CHG', 'CHGCAR', 'LOCPOT', 'EIGENVAL', 'IBZKPT', 'PCDAT', 'POTCAR', 'WAVECAR', 'node_info', "WAVECAR",
                 "WAVEDER", 'DOSCAR', 'PROCAR']
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass
        os.chdir('../')
        from core.utils.zipdir import ZipDir
        ZipDir('electronic_hybrid', 'electronic_hybrid.zip')
        shutil.rmtree('./electronic_hybrid', ignore_errors=True)
        return

    # # ====================================================================================
    # # stage II - frequency dependent dielectric constant with independent-particle picture
    # # ====================================================================================
    # logger.info("Frequency dependent dielectric constant with PBE (independent particle approximation) run")
    # pbe_omega.update({'NBANDS': nbands})
    # # parallisation wont work properly
    # try:
    #     del pbe_omega['NPAR']
    # except KeyError:
    #     pass
    #
    # if hybrid_GGA:
    #     logger.info("Update INCAR configuration for hybrid GGA")
    #     pbe_omega.update(hse06_set)
    #
    # logger.info("INCAR settings: ")
    # for k in pbe_omega.keys():
    #     logger.info("       " + str(k) + '=' + str(pbe_omega[k]))
    #
    # vasp = Vasp(**pbe_omega)
    # vasp.set_crystal(structure)
    # vasp.execute()
    #
    # if vasp.completed:
    #     logger.info("PBE frequency-dependent dielectric constant run completed properly")
    # else:
    #     raise Exception("PBE frequency-dependent dielectric constant run failed, will stop proceeding")
    #
    # # copy the files across for debugging or data grabbing
    # shutil.copy('OSZICAR', 'OSZICAR.PBE.DIAG')
    # shutil.copy('OUTCAR', 'OUTCAR.PBE.DIAG')
    # shutil.copy('vasprun.xml', 'vasprun.PBE.DIAG.xml')
    #
    # # ====================================================================================
    # # stage III - frequency dependent dielectric constant with RPA
    # # ====================================================================================
    # logger.info("Frequency dependent dielectric constant under PBE-RPA")
    # pbe_rpa_omega.update({'NBANDS': nbands})
    # try:
    #     del pbe_rpa_omega['NPAR']
    # except KeyError:
    #     pass
    #
    # if hybrid_GGA:
    #     logger.info("Update INCAR configuration for hybrid GGA")
    #     pbe_rpa_omega.update(hse06_set)
    #
    # logger.info("INCAR settings: ")
    # for k in pbe_rpa_omega.keys():
    #     logger.info("       " + str(k) + '=' + str(pbe_rpa_omega[k]))
    #
    # vasp = Vasp(**pbe_rpa_omega)
    # vasp.set_crystal(structure)
    # vasp.execute()
    #
    # if vasp.completed:
    #     logger.info("PBE-RPA calculation terminated")
    # else:
    #     raise Exception("PBE-RPA run failed, please check what's going on...")
    #
    # # copy the files across for debugging or data grabbing
    # shutil.copy('OSZICAR', 'OSZICAR.RPA.DIAG')
    # shutil.copy('OUTCAR', 'OUTCAR.RPA.DIAG')
    # shutil.copy('vasprun.xml', 'vasprun.RPA.DIAG.xml')


# =================================================================
# Procedures for calculating spin-polarized charge densities and
# electronic band structures
# =================================================================

KPOINTS_string_dict = {'100': """KPOINTS 
5
Line-mode
rec
 0.5 0.0 0.0 ! X
 0.0 0.0 0.0 ! G

 0.0 0.0 0.0 ! G
 0.5 0.5 0.0 ! M""",  ##cubic
                       '110': """KPOINTS 
5
Line-mode
rec
 0.0 0.0 0.0 ! G
 0.5 0.0 0.0 ! X
 
 0.5 0.0 0.0 ! X
 0.5 0.5 0.0 ! S
 
 0.5 0.5 0.0 ! S
 0.0 0.5 0.0 ! Y
 
 0.0 0.5 0.0 ! Y
 0.0 0.0 0.0 ! G
 
 0.0 0.0 0.0 ! G
 0.5 0.5 0.0 ! S""",  ##rectangular
                       "111": """KPOINTS 
5
Line-mode
rec
  0.5 0.5 0.0 ! M
  0.0 0.0 0.0 ! G
  
  0.0 0.0 0.0 ! G
  0.5 0.0 0.0 ! K
  
  0.5 0.0 0.0 ! K
  0.5 0.5 0.0 ! M"""}  ##hexagonal?


def electronic_structure_calculator():
    logger = setup_logger(output_filename='electronic_structures.log')
    update_core_info()
    logger.info("==========GGA electronic (spin-densities/band) structure calculations with VASP==========")
    structure = load_structure(logger)

    logger.info("STAGE 1 Converging spin-unploarized charge densities")
    single_point_pbe.update(
        {'IALGO': 48, 'ISPIN': 1, 'MP_points': [4, 4, 1], 'ENCUT': 500, 'clean_after_success': False, 'LCHARG': True,
         'LWAVE': False})
    vasp = Vasp(**single_point_pbe)
    vasp.set_crystal(structure)
    vasp.execute()

    logger.info("STAGE 2 Converging spin-ploarized charge densities")
    single_point_pbe.update(
        {'IALGO': 48, 'ISPIN': 2, 'MP_points': [4, 4, 1], 'ENCUT': 500, 'clean_after_success': False, 'LCHARG': True,
         'LWAVE': False, 'ICHARG': 1})
    vasp = Vasp(**single_point_pbe)
    vasp.set_crystal(structure)
    vasp.execute()

    # save vasprun.xml for charge-density analysis
    shutil.copy('./vasprun.xml', './vasprun_SPIN_CHG.xml')
    shutil.copy('./CHGCAR', './CHGCAR_SPIN')

    if '100' in os.getcwd():
        orient = '100'
    elif '110' in os.getcwd():
        orient = '110'
    elif '111' in os.getcwd():
        orient = '111'

    single_point_pbe.update(
        {'IALGO': 48, 'KPOINT_string': KPOINTS_string_dict[orient], 'clean_after_success': True, 'LWAVE': False,
         'ICHARG': 1, "LCHARG": False})
    logger.info("STAGE 3 Spin-polarized band structure calculations")
    logger.info("Customized K-Point Path ")
    logger.info(KPOINTS_string_dict[orient])

    vasp = Vasp(**single_point_pbe)
    vasp.set_crystal(structure)
    vasp.execute()

    shutil.move('./vasprun.xml', './vasprun_spin_BAND.xml')


# =================================================================
# Used by the bulk LCMO project
# =================================================================


def GGA_U_structure_optimisation():
    logger = setup_logger(output_filename='relax.log')
    update_core_info()
    logger.info("==========Full GGA+U Structure Optimisation with VASP==========")
    structure = load_structure(logger)
    gga_u_options = __set_U_correction_dictionary(structure)
    default_bulk_optimisation_set.update(gga_u_options)
    default_bulk_optimisation_set.update(
        {'ENCUT': 400, 'ISPIN': 2, 'IVDW': 12, 'MP_points': [4, 4, 1], 'Gamma_centered': True})  # ,'NELECT':0})

    if 'ISIF' in default_bulk_optimisation_set:
        del default_bulk_optimisation_set['ISIF']
    if 'isif' in default_bulk_optimisation_set:
        del default_bulk_optimisation_set['isif']

    vasp = Vasp(**default_bulk_optimisation_set)
    vasp.set_crystal(structure)
    vasp.execute()


def GGA_U_high_spin_structure_optimisation():
    logger = setup_logger(output_filename='relax.log')
    update_core_info()
    logger.info("==========Full GGA+U high spin Structure Optimisation with VASP==========")
    structure = load_structure(logger)
    gga_u_options = __set_U_correction_dictionary(structure)
    default_bulk_optimisation_set.update(gga_u_options)

    magmom_options = __set_high_spin_magmom_dictionary(structure)
    default_bulk_optimisation_set.update(magmom_options)

    vasp = Vasp(**default_bulk_optimisation_set)
    vasp.set_crystal(structure)
    vasp.execute()


def high_spin_structure_optimisation():
    logger = setup_logger(output_filename='relax.log')
    update_core_info()
    logger.info("==========Full  high spin Structure Optimisation with VASP==========")
    structure = load_structure(logger)

    magmom_options = __set_high_spin_magmom_dictionary(structure)
    default_bulk_optimisation_set.update(magmom_options)

    vasp = Vasp(**default_bulk_optimisation_set)
    vasp.set_crystal(structure)
    vasp.execute()


def __set_U_correction_dictionary(structure):
    from core.models.element import U_corrections, orbital_index
    LDAUL = ''
    LDAUU = ''
    labels = [x.label for x in structure.all_atoms(unique=True, sort=True)]
    unique_labels = []
    for l in labels:
        if l not in unique_labels:
            unique_labels.append(l)
    for label in unique_labels:
        if label in U_corrections.keys():
            orbital = list(U_corrections[label].keys())[-1]
            LDAUL += ' ' + str(orbital_index[orbital])
            LDAUU += ' ' + str(U_corrections[label][orbital])
        else:
            LDAUL += ' -1'
            LDAUU += ' 0'
    GGA_U_options = {'LDAU': '.TRUE.', 'LDAUTYPE': 2, 'LDAUJ': '0 ' * len(unique_labels), 'LDAUL': LDAUL,
                     'LDAUU': LDAUU}
    return GGA_U_options


def __set_high_spin_magmom_dictionary(structure):
    # this sets transition metal ions into its highest spin state for performing calculations in an initial high-spin FM states
    MAGMOM = ''
    labels = [x.label for x in structure.all_atoms(unique=True, sort=True)]
    unique_labels = []
    for l in labels:
        if l not in unique_labels:
            unique_labels.append(l)
    from core.models.element import high_spin_states
    for l in unique_labels:
        if l in high_spin_states.keys():
            MAGMOM += str(structure.all_atoms_count_dictionaries()[l]) + '*' + str(high_spin_states[l]) + ' '
        else:
            MAGMOM += str(structure.all_atoms_count_dictionaries()[l]) + '*0 '
    return {"MAGMOM": MAGMOM}


def default_xy_strained_optimisation():
    default_bulk_optimisation_set.update({'executable': 'vasp_std-z'})
    default_structural_optimisation()


def default_spin_unpolarised_xy_strained_optimisation():
    default_bulk_optimisation_set.update({'executable': 'vasp_std-z'})
    spin_unpolarised_optimization()


def default_highspin_xy_strained_optimisation():
    default_bulk_optimisation_set.update({'executable': 'vasp_std-z'})
    high_spin_structure_optimisation()


def default_GGA_U_highspin_xy_strained_optimisation():
    default_bulk_optimisation_set.update({'executable': 'vasp_std-z'})
    GGA_U_high_spin_structure_optimisation()


def default_xy_strained_optimisation_with_existing_vasp_setup():
    default_bulk_optimisation_set.update({'executable': 'vasp_std-z'})
    default_run_with_existing_vasp_setup()


if __name__ == "__main__":
    # default_symmetry_preserving_optimisation()
    # rpa_dielectric_constants(hybrid_GGA=True)
    default_structural_optimisation()
