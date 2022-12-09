from twodPV.calculators import *

def energy_cutoff_scan():
    logger = setup_logger(output_filename='electronic_structures.log')
    update_core_info()
    structure = VaspReader(input_location='./POSCAR').read_POSCAR()

    #for e_c in [100,150,200,250,300,350,400,450,500]:
    for e_c in [100, 120,140,150, 160,180,200,220,240,250,260,280,300,320,340,350,400,450,500]:
        try:
            os.remove('./CHGCAR')
            os.remove('./KPOINTS')
        except:
            pass

        logger.info("Testing energy cutoff of "+str(e_c)+' eV')
        single_point_pbe.update(
            {'IALGO': 38, 'ISPIN': 1, 'MP_points': [4, 4, 1], 'ENCUT': e_c, 'clean_after_success': False,
             'LCHARG': True,
             'LWAVE': False})
        vasp = Vasp(**single_point_pbe)
        vasp.set_crystal(structure)
        vasp.execute()

        logger.info("Check crystal orientations and set up K-point path accordingly")
        if '100' in os.getcwd():
            orient = '100'
        elif '110' in os.getcwd():
            orient = '110'
        elif '111' in os.getcwd():
            orient = '111'

        if vasp.completed:
            single_point_pbe.update(
                {'IALGO': 38, 'KPOINT_string': KPOINTS_string_dict[orient], 'clean_after_success': True, 'LWAVE': False,
                 'ICHARG': 1, "LCHARG": False, 'LORBIT': 11, 'ENCUT': e_c})
            logger.info("STAGE 3 Spin-polarized band structure calculations")
            logger.info("Customized K-Point Path ")
            logger.info(KPOINTS_string_dict[orient])

            vasp = Vasp(**single_point_pbe)
            vasp.set_crystal(structure)
            vasp.execute()

            from pymatgen.io.vasp.outputs import Vasprun
            run = Vasprun("./vasprun.xml")
            bands = run.get_band_structure("./KPOINTS")
            band_gap_data = bands.get_band_gap()

            logger.info("Cutoff "+str(e_c)+" eV,  PBE band gap energy is " + str(band_gap_data['energy']) + ' eV')
            shutil.move('./vasprun.xml', './vasprun_BAND_'+str(e_c)+'.xml')

def execute():
    try:
        os.remove('./CHGCAR')
        os.remove('./WAVECAR')
        os.remove("./WAVEDER")
        os.remove('./KPOINTS')
    except:
        pass

    logger = setup_logger(output_filename='electronic_structures.log')
    update_core_info()
    structure = VaspReader(input_location='./POSCAR').read_POSCAR()

    # ==================================================================
    logger.info("STAGE 1 Converging spin-unploarized charge densities")
    single_point_pbe.update(
        {'IALGO': 38, 'ISPIN': 1, 'MP_points': [4, 4, 1], 'ENCUT': 250, 'clean_after_success': False, 'LCHARG': True,
         'LWAVE': True})
    vasp = Vasp(**single_point_pbe)
    vasp.set_crystal(structure)
    vasp.execute()
    shutil.copy('./vasprun.xml', './vasprun_NOSPIN_CHG.xml')
    shutil.copy('./CHGCAR', './CHGCAR_NOSPIN')
    shutil.copy('./OUTCAR', './OUTCAR_NOSPIN')
    shutil.copy('./OSZICAR', './OSZICAR_NOSPIN')
    shutil.copy('./vasp.log', './vasp_NOSPIN.log')

    # ==================================================================

    if not vasp.completed:
        logger.info("Unsuccessful termination, try to rerun with a different IALGO")
        single_point_pbe.update(
            {'IALGO': 48, 'ISPIN': 1, 'MP_points': [4, 4, 1], 'ENCUT': 250, 'clean_after_success': False,
             'LCHARG': True, 'LWAVE': True})
        vasp = Vasp(**single_point_pbe)
        vasp.set_crystal(structure)
        vasp.execute()
        shutil.copy('./vasprun.xml', './vasprun_NOSPIN_CHG.xml')
        shutil.copy('./CHGCAR', './CHGCAR_NOSPIN')
        shutil.copy('./OUTCAR', './OUTCAR_NOSPIN')
        shutil.copy('./OSZICAR', './OSZICAR_NOSPIN')
        shutil.copy('./vasp.log', './vasp_NOSPIN.log')

        if not vasp.completed:
            raise Exception("Failed to converge SCF with two trials")
        else:
            f = open('./OUTCAR', 'r')
            for l in f.readlines():
                if 'NBANDS=' in l:
                    nbands = int(l.split()[-1]) * 3
                    logger.info("NBANDS from ground state spin-unpolarized calculations: " + str(nbands))
    else:
        f = open('./OUTCAR', 'r')
        for l in f.readlines():
            if 'NBANDS=' in l:
                nbands = int(l.split()[-1]) * 3
                logger.info("NBANDS from ground state spin-unpolarized calculations: " + str(nbands))

    # ====================================================================================
    # Frequency dependent dielectric constant with independent-particle picture
    # ====================================================================================
    os.remove("./KPOINTS")
    logger.info("Frequency dependent dielectric constant with PBE (independent particle approximation) run")
    logger.info("Copy the converged spin-polarization charge densities across")
    shutil.copy('./CHGCAR_NOSPIN', 'CHGCAR')
    pbe_omega.update({'NBANDS': nbands, 'ISPIN': 1, 'ICHARG': 1, 'clean_after_success': False, 'ENCUT': 250})
    # parallisation wont work properly
    try:
        del pbe_omega['NPAR']
    except KeyError:
        pass

    logger.info("INCAR settings: ")
    for k in pbe_omega.keys():
        logger.info("       " + str(k) + '=' + str(pbe_omega[k]))

    vasp = Vasp(**pbe_omega)
    vasp.set_crystal(structure)
    vasp.execute()

    if vasp.completed:
        logger.info("PBE frequency-dependent dielectric constant run completed properly")
    else:
        raise Exception("PBE frequency-dependent dielectric constant run failed, will stop proceeding")

    # copy the files across for debugging or data grabbing
    shutil.copy('OSZICAR', 'OSZICAR.PBE.DIAG')
    shutil.copy('OUTCAR', 'OUTCAR.PBE.DIAG')
    shutil.copy('vasprun.xml', 'vasprun.PBE.DIAG.xml')
    shutil.copy('vasp.log', 'vasprun.PBE.DIAG.log')

    # ====================================================================================
    # Frequency dependent dielectric constant with RPA
    # ====================================================================================
    logger.info("Frequency dependent dielectric constant under PBE-RPA")
    pbe_rpa_omega.update({'NBANDS': nbands, 'ISPIN': 1, 'ICHARG': 1, 'clean_after_success': False, 'ENCUT': 250})
    try:
        del pbe_rpa_omega['NPAR']
    except KeyError:
        pass

    logger.info("INCAR settings: ")
    for k in pbe_rpa_omega.keys():
        logger.info("       " + str(k) + '=' + str(pbe_rpa_omega[k]))

    vasp = Vasp(**pbe_rpa_omega)
    vasp.set_crystal(structure)
    vasp.execute()

    if vasp.completed:
        logger.info("PBE-RPA calculation terminated")
    else:
        raise Exception("PBE-RPA run failed, please check what's going on...")

    # copy the files across for debugging or data grabbing
    shutil.copy('OSZICAR', 'OSZICAR.RPA.DIAG')
    shutil.copy('OUTCAR', 'OUTCAR.RPA.DIAG')
    shutil.copy('vasprun.xml', 'vasprun.RPA.DIAG.xml')

    # ==================================================================
    logger.info("Converging spin-ploarized charge densities")
    shutil.copy('./CHGCAR_NOSPIN', 'CHGCAR')
    try:
        os.remove("./WAVECAR")
    except:
        pass
    single_point_pbe.update(
        {'IALGO': 38, 'ISPIN': 2, 'MP_points': [4, 4, 1], 'ENCUT': 250, 'clean_after_success': False, 'LCHARG': True,
         'LWAVE': False, 'ICHARG': 1, 'LORBIT': 11})
    vasp = Vasp(**single_point_pbe)
    vasp.set_crystal(structure)
    vasp.execute()
    shutil.copy('./vasprun.xml', './vasprun_SPIN_CHG.xml')
    shutil.copy('./CHGCAR', './CHGCAR_SPIN')
    shutil.copy('./OUTCAR', './OUTCAR_SPIN')
    shutil.copy('./OSZICAR', './OSZICAR_SPIN')
    shutil.copy('./vasp.log', './vasp_SPIN.log')
    # ==================================================================

    if not vasp.completed:
        logger.info("Unsuccessful termination, try to rerun with a different IALGO")
        single_point_pbe.update(
            {'IALGO': 48, 'ISPIN': 2, 'MP_points': [4, 4, 1], 'ENCUT': 250, 'clean_after_success': False,
             'LCHARG': True, 'LWAVE': False})
        vasp = Vasp(**single_point_pbe)
        vasp.set_crystal(structure)
        vasp.execute()
        shutil.copy('./vasprun.xml', './vasprun_SPIN_CHG.xml')
        shutil.copy('./CHGCAR', './CHGCAR_SPIN')
        shutil.copy('./OUTCAR', './OUTCAR_SPIN')
        shutil.copy('./OSZICAR', './OSZICAR_SPIN')
        shutil.copy('./vasp.log', './vasp_SPIN.log')

        if not vasp.completed:
            raise Exception("Failed to converge SCF with two trials")

    # ==================================================================
    logger.info("Check crystal orientations and set up K-point path accordingly")
    if '100' in os.getcwd():
        orient = '100'
    elif '110' in os.getcwd():
        orient = '110'
    elif '111' in os.getcwd():
        orient = '111'
    # ==================================================================

    # ==================================================================
    single_point_pbe.update(
        {'IALGO': 38, 'KPOINT_string': KPOINTS_string_dict[orient], 'clean_after_success': True, 'LWAVE': False,
         'ICHARG': 1, "LCHARG": False, 'LORBIT': 11, 'ISPIN': 2})
    logger.info("STAGE 3 Spin-polarized band structure calculations")
    logger.info("Customized K-Point Path ")
    logger.info(KPOINTS_string_dict[orient])

    vasp = Vasp(**single_point_pbe)
    vasp.set_crystal(structure)
    vasp.execute()

    shutil.move('./vasprun.xml', './vasprun_spin_BAND.xml')
    shutil.copy('./OUTCAR', './OUTCAR_BAND')
    shutil.copy('./OSZICAR', './OSZICAR_BAND')
    shutil.copy('./vasp.log', './vasp_BAND.log')

    # ==================================================================

    if not vasp.completed:
        single_point_pbe.update(
            {'IALGO': 48, 'ISPIN': 2, 'KPOINT_string': KPOINTS_string_dict[orient], 'ENCUT': 250,
             'clean_after_success': True, 'LWAVE': False, 'ICHARG': 1, "LCHARG": False, 'LORBIT': 11})
        vasp = Vasp(**single_point_pbe)
        vasp.set_crystal(structure)
        vasp.execute()
        shutil.move('./vasprun.xml', './vasprun_spin_BAND.xml')
        shutil.copy('./OUTCAR', './OUTCAR_BAND')
        shutil.copy('./OSZICAR', './OSZICAR_BAND')
        shutil.copy('./vasp.log', './vasp_BAND.log')

        if not vasp.completed:
            raise Exception("Failed to converge SCF with two trials")

    # ==================================================================
    # Extracting the band gap energies to see if it is a semiconductor
    # ==================================================================

    from pymatgen.io.vasp.outputs import Vasprun
    run = Vasprun("./vasprun_spin_BAND.xml")
    bands = run.get_band_structure("./KPOINTS")
    band_gap_data = bands.get_band_gap()

    logger.info("PBE band gap energy is " + str(band_gap_data['energy']) + ' eV')


    #os.chdir('../')
    #from core.utils.zipdir import ZipDir
    #ZipDir('electronic_workflow','electronic_workflow.zip')
    #shutil.rmtree('./electronic_workflow')

