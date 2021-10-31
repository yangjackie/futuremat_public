import os, glob, tarfile, math, time, h5py
from core.calculators.vasp import Vasp
from core.dao.vasp import VaspReader, VaspWriter
from twodPV.collect_data import populate_db, _atom_dict
from pymatgen.io.vasp.outputs import Vasprun, BSVasprun
from pymatgen.electronic_structure.core import Spin
try:
    from sumo.electronic_structure.bandstructure import get_reconstructed_band_structure
except:
    pass
from ase.db import connect
import shutil
from ase.io.vasp import *
import itertools
from core.external.vasp.anharmonic_score import *
from core.utils.loggings import setup_logger
import phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

logger = setup_logger(output_filename='data_collector.log')
# logger = setup_logger(output_filename='formation_energies_data.log')
# logger = setup_logger(output_filename='phonon_data.log')


halide_C = ['F', 'Cl', 'Br', 'I']
halide_A = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Cu', 'Ag', 'Au', 'Hg', 'Ga', 'In', 'Tl']
halide_B = ['Mg', 'Ca', 'Sr', 'Ba', 'Se', 'Te', 'As', 'Si', 'Ge', 'Sn', 'Pb', 'Ga', 'In', 'Sc', 'Y', 'Ti', 'Zr', 'Hf',
            'V', 'Nb', 'Ta', 'Cr', 'Mo', 'W', 'Mn', 'Tc', 'Re', 'Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir', 'Ni', 'Pd', 'Pt',
            'Cu', 'Ag', 'Au', 'Zn', 'Cd', 'Hg']

chalco_C = ['O', 'S', 'Se']
chalco_A = ['Ba', 'Mg', 'Ca', 'Sr', 'Be', 'Ra', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Pd', 'Pt', 'Cu', 'Ag', 'Zn',
            'Cd', 'Hg', 'Ge', 'Sn', 'Pb']
chalco_B = ['Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'Cr', 'Mo', 'W', 'Mn', 'Tc', 'Re', 'Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir',
            'Ni', 'Pd', 'Pt', 'Sn', 'Ge', 'Pb', 'Si', 'Te', 'Po']

# chalco_B=['Po']

A_site_list = [chalco_A]#, halide_A]
B_site_list = [chalco_B]#, halide_B]
C_site_list = [chalco_C]#, halide_C]

all_elements_list = list(itertools.chain(*[A_site_list, B_site_list, C_site_list]))
all_elements_list = list(itertools.chain(*all_elements_list))
all_elements_list = list(set(all_elements_list))

reference_atomic_energies = {}


def element_energy(db):
    logger.info("========== Collecting reference energies for constituting elements ===========")
    cwd = os.getcwd()
    os.chdir('./elements')

    for element in all_elements_list:
        kvp = {}
        data = {}
        uid = 'element_' + str(element)
        logger.info(uid)
        tf = tarfile.open(element + '.tar.gz')
        tf.extractall()
        os.chdir(element)

        calculator = Vasp()
        calculator.check_convergence()
        if not calculator.completed:
            logger.info(uid, 'failed')
        atoms = [i for i in read_vasp_xml(index=-1)][-1]
        e = list(_atom_dict(atoms).keys())[-1]
        reference_atomic_energies[element] = atoms.get_calculator().get_potential_energy() / _atom_dict(atoms)[e]
        kvp['uid'] = uid
        kvp['total_energy'] = atoms.get_calculator().get_potential_energy()
        logger.info(uid + ' ' + str(reference_atomic_energies[element]) + ' eV/atom')
        populate_db(db, atoms, kvp, data)
        os.chdir("..")
        shutil.rmtree(element)
        try:
            os.rmtree(element)
        except:
            pass
    os.chdir('..')


def formation_energy(atoms):
    fe = atoms.get_calculator().get_potential_energy()
    # print(fe,_atom_dict(atoms),reference_atomic_energies)
    for k in _atom_dict(atoms).keys():
        fe = fe - _atom_dict(atoms)[k] * reference_atomic_energies[k]
    #   print(k,reference_atomic_energies[k],fe)
    return fe / atoms.get_number_of_atoms()


def full_relax_data(db):
    cwd = os.getcwd()
    system_counter = 0
    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:

                for c in C_site_list[i]:
                    kvp={}
                    data={}
                    if (a!='Ra'):
                        if (b!='Po'):
                            continue

                    system_counter += 1
                    logger.info("Working on system number: " + str(system_counter))
                    system_name = a + b + c
                    uid = system_name + '_Pm3m'

                    # open up the tar ball
                    cwd = os.getcwd()
                    try:
                        tf = tarfile.open(system_name + '_Pm3m.tar.gz')
                        tf.extractall()
                        os.chdir(system_name + '_Pm3m')
                    except:
                        logger.info(system_name + '_Pm3m' + ' tar ball not working')
                        continue


                    try:
                        calculator = Vasp()
                        calculator.check_convergence()
                        if calculator.completed:
                            atoms = [a for a in read_vasp_xml(index=-1)][-1]
                            kvp['uid'] = uid
                            kvp['total_energy'] = atoms.get_calculator().get_potential_energy()
                            kvp['formation_energy'] = formation_energy(atoms)
                            populate_db(db, atoms, kvp, data)
                            logger.info(system_name + '_Pm3m' + ' formation energy: ' + str(
                                kvp['formation_energy']) + ' eV/atom')
                        else:
                            pass
                    except:
                        logger.info(system_name + '_Pm3m' + ' formation energy: ' + str('NaN'))

                    try:
                        tf = tarfile.open('full_relax.tar.gz')
                        tf.extractall()

                    except:
                        logger.info(system_name + '_Pm3m' + ' full_relax tar ball not working')

                    if os.path.isdir('./full_relax'):

                        os.chdir('./full_relax')
                        try:
                            calculator = Vasp()
                            calculator.check_convergence()
                            atoms = None
                            if calculator.completed:
                                atoms = [a for a in read_vasp_xml(index=-1)][-1]
                                kvp['uid'] = uid + '_fullrelax'
                                kvp['total_energy'] = atoms.get_calculator().get_potential_energy()

                                kvp['formation_energy'] = formation_energy(atoms)
                                populate_db(db, atoms, kvp, data)
                                logger.info(system_name + '_Pm3m' + ' formation energy (fully relaxed): ' + str(
                                    kvp['formation_energy']) + ' eV/atom')
                        except:
                            logger.info(system_name + '_Pm3m' + ' formation energy (fully relaxed): NaN ')

                        os.chdir('..')

                    # get the formation energies for the randomised structures
                    try:
                        tf = tarfile.open('randomised.tar.gz')
                        tf.extractall()
                    except:
                        pass
                    #  print(os.getcwd())
                    if os.path.isdir('./randomised'):
                        os.chdir('randomised')
                        # print(os.getcwd()+'\n')

                        for counter in range(10):
                            rkvp = {}
                            if os.path.isdir('./str_' + str(counter)):
                                os.chdir('./str_' + str(counter))
                                try:
                                   calculator = Vasp()
                                   calculator.check_convergence()
                                   atoms = None
                                   if calculator.completed:
                                        atoms = [a for a in read_vasp_xml(index=-1)][-1]
                                        rkvp['uid'] = uid + '_rand_str_' + str(counter)
                                        rkvp['total_energy'] = atoms.get_calculator().get_potential_energy()

                                        rkvp['formation_energy'] = formation_energy(atoms)
                                        populate_db(db, atoms, rkvp, data)
                                        logger.info(system_name + '_Pm3m' + ' formation energy (rand ' + str(
                                            counter) + '): ' + str(rkvp['formation_energy']) + ' eV/atom')
                                except:
                                    logger.info(
                                        system_name + '_Pm3m' + ' formation energy (rand ' + str(counter) + '): ' + str(
                                            'NaN'))
                                os.chdir('..')
                        os.chdir('..')
                        try:
                            shutil.rmtree('randomised')
                        except:
                            pass
                        try:
                            os.rmtree('randomised')
                        except:
                            pass

                    os.chdir("..")
                    try:
                        shutil.rmtree(system_name + '_Pm3m')
                    except:
                        pass
                    try:
                        os.rmtree(system_name + '_Pm3m')
                    except:
                        pass


def all_data(db):
    cwd = os.getcwd()
    system_counter = 0
    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:

                    system_counter += 1
                    logger.info("Working on system number: " + str(system_counter))
                    system_name = a + b + c
                    uid = system_name + '_Pm3m'

                    # open up the tar ball
                    cwd = os.getcwd()
                    try:
                        tf = tarfile.open(system_name + '_Pm3m.tar.gz')
                        tf.extractall()
                        os.chdir(system_name + '_Pm3m')
                    except:
                        logger.info(system_name + '_Pm3m' + ' tar ball not working')
                        continue

                    # get the formation energies for the randomised structures
                    try:
                        tf = tarfile.open('randomised.tar.gz')
                        tf.extractall()
                    except:
                        pass
                    # print(os.getcwd())
                    # if os.path.isdir('./randomised'):
                    #     os.chdir('randomised')
                    #     # print(os.getcwd()+'\n')
                    #
                    #     for counter in range(10):
                    #         rkvp = {}
                    #         if os.path.isdir('./str_' + str(counter)):
                    #             os.chdir('./str_' + str(counter))
                    #             try:
                    #                 calculator = Vasp()
                    #                 calculator.check_convergence()
                    #                 atoms = None
                    #                 if calculator.completed:
                    #                     atoms = [a for a in read_vasp_xml(index=-1)][-1]
                    #                     rkvp['uid'] = uid + '_rand_str_' + str(counter)
                    #                     rkvp['total_energy'] = atoms.get_calculator().get_potential_energy()
                    #
                    #                     rkvp['formation_energy'] = formation_energy(atoms)
                    #                     populate_db(db, atoms, rkvp, data)
                    #                     logger.info(system_name + '_Pm3m' + ' formation energy (rand ' + str(
                    #                         counter) + '): ' + str(rkvp['formation_energy']) + ' eV/atom')
                    #                 # else:
                    #                 #    continue
                    #             except:
                    #                 logger.info(
                    #                     system_name + '_Pm3m' + ' formation energy (rand ' + str(counter) + '): ' + str(
                    #                         'NaN'))
                    #             os.chdir('..')
                    #     os.chdir('..')
                    #     try:
                    #         shutil.rmtree('randomised')
                    #     except:
                    #         pass
                    #     try:
                    #         os.rmtree('randomised')
                    #     except:
                    #         pass

                    kvp = {}
                    data = {}

                    # get the formation energies for the cubic Pm3m phase
                    get_properties = True
                    try:
                        calculator = Vasp()
                        calculator.check_convergence()
                        if calculator.completed:
                            atoms = [a for a in read_vasp_xml(index=-1)][-1]
                            kvp['uid'] = uid
                            kvp['total_energy'] = atoms.get_calculator().get_potential_energy()
                            kvp['formation_energy'] = formation_energy(atoms)
                            populate_db(db, atoms, kvp, data)
                            logger.info(system_name + '_Pm3m' + ' formation energy: ' + str(
                                kvp['formation_energy']) + ' eV/atom')
                        else:
                            logger.info(
                                system_name + '_Pm3m' + ' not converged in structure optimisation, not continuing ')
                            os.chdir("..")
                            try:
                                shutil.rmtree(system_name + '_Pm3m')
                            except:
                                pass
                            try:
                                os.rmtree(system_name + '_Pm3m')
                            except:
                                pass
                            get_properties = False
                    except:
                        logger.info(system_name + '_Pm3m' + ' formation energy: ' + str('NaN'))

                    #                    os.chdir("..")
                    #                    try:
                    #                        shutil.rmtree(system_name + '_Pm3m')
                    #                    except:
                    #                        pass
                    #                    try:
                    #                        os.rmtree(system_name + '_Pm3m')
                    #                    except:
                    #                        pass


                    print(system_name + '_Pm3m' + ' get_properties? ' + str(get_properties))
                    if not get_properties: continue

                    # collect the electronic structure data (band gap)
                    # try:
                    #     dos_tf = tarfile.open('dos.tar.gz')
                    #     dos_tf.extractall()
                    # except:
                    #     pass
                    #
                    # if os.path.isdir('./dos'):
                    #    logger.info(system_name + '_Pm3m' + ' collect band gap')
                    #    os.chdir('./dos')
                    #    if os.path.exists('vasprun_BAND.xml'):
                    #        vr = BSVasprun('./vasprun_BAND.xml')
                    #        bs = vr.get_band_structure(line_mode=True, kpoints_filename='KPOINTS')
                    #        bs = get_reconstructed_band_structure([bs])
                    #
                    #        # =====================================
                    #        # get band gap data
                    #        # =====================================
                    #        bg_data = bs.get_band_gap()
                    #        kvp['direct_band_gap']=bg_data['direct']
                    #        kvp['band_gap']=bg_data['energy']
                    #        logger.info(system_name + '_Pm3m' + ' direct band gap '+str(bg_data['energy'])+'  band gap energy:'+str(kvp['band_gap'])+' eV')
                    #        populate_db(db, atoms, kvp, data)
                    #
                    #    os.chdir('..')

                    # Check the phonon calculations are converged
                    force_constant_exists = os.path.isfile('force_constants.hdf5')
                    md_calculations_exists = os.path.isfile('vasprun_md.xml')
                    if force_constant_exists:
                        try:
                            phonon_tf = tarfile.open('phonon.tar.gz')
                            phonon_tf.extractall()
                        except:
                            pass

                    if os.path.isdir('./phonon'):
                        # Check that individual finite displacement calculation is well converged
                        phonon_converged = True
                        os.chdir('phonon')
                        for sub_f in ['ph-POSCAR-001', 'ph-POSCAR-002', 'ph-POSCAR-003']:
                            os.chdir(sub_f)
                            f = open('./OUTCAR', 'r')
                            for l in f.readlines():
                                if 'NELM' in l:
                                    nelm = l.split()[2].replace(';', '')
                                    nelm = int(nelm)
                            f.close()

                            f = open('./vasp.log', 'r')
                            lines = []
                            for l in f.readlines():
                                if ('DAV:' in l) or ("RMM:" in l):
                                    lines.append(l)
                            if len(lines) >= nelm:
                                phonon_converged = False
                            os.chdir('..')  # step out from str_# directory
                        os.chdir('..')  # step out from phonon directory

                    if phonon_converged:
                        # get the phonon eigen frequencies at high symmetry Q points
                        at_phonon = False
                        try:
                            os.chdir('phonon')
                            at_phonon = True
                        except:
                            pass
                        try:
                            try:
                                os.rename('force_constants.hdf5', 'f.hdf5')
                            except:
                                pass

                            phonon = phonopy.load(supercell_matrix=[2, 2, 2], primitive_matrix='auto',
                                                  unitcell_filename='POSCAR')
                            path = [[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0]]]
                            labels = ["G", "R", "M", "X"]
                            qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=10)
                            phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)
                            phonon_dict = phonon.get_band_structure_dict()

                            for _pp, p in enumerate(path[0]):
                                for _i, qset in enumerate(phonon_dict['qpoints']):
                                    for _j, _q in enumerate(qset):
                                        if (_q[0] == p[0]) and (_q[1] == p[1]) and (_q[2] == p[2]):
                                            kvp[labels[_pp] + "_min_ph_freq"] = min(phonon_dict['frequencies'][_i][_j])

                            for _pp, p in enumerate(path[0]):
                                logger.info(system_name + ' ' + labels[_pp] + "_min_ph_freq:" + str(
                                    kvp[labels[_pp] + "_min_ph_freq"]))
                            populate_db(db, atoms, kvp, data)
                        except:
                            pass
                        if at_phonon:
                            os.chdir("..")  # step out of phonon directory

                    md_done = False
                    if md_calculations_exists:
                        try:
                            md_tf = tarfile.open('MD.tar.gz')
                            md_tf.extractall()
                        except:
                            pass
                    if os.path.isdir('./MD'):
                        os.chdir('MD')
                        t = 0
                        try:
                            f = open('./OSZICAR', 'r')
                            for l in f.readlines():
                                if 'T=' in l:
                                    t += 1
                            f.close()
                        except:
                            pass
                        if t == 800:
                            md_done = True
                        os.chdir('..')  # step out from MD directory

                    if force_constant_exists and phonon_converged and md_done:
                        logger.info(system_name + '_Pm3m' + ' Valid Phonon and MD Results')
                        uid = system_name + '_Pm3m'
                        kvp['uid'] = uid
                        # calculate the anharmonic score
                        os.chdir('./phonon')

                        try:
                            try:
                                os.rename('force_constants.hdf5', 'f.hdf5')
                            except:
                                pass
                            # scorer = AnharmonicScore(md_frames='../vasprun_md.xml',ref_frame='./SPOSCAR',force_constants=None)
                            # __sigmas, _ = scorer.structural_sigma(return_trajectory=False)
                            scorer = AnharmonicScore(md_frames='../vasprun_md.xml', ref_frame='./SPOSCAR',
                                                     force_constants=None, atoms=[a])
                            __sigmas, _ = scorer.structural_sigma(return_trajectory=False)
                            kvp['sigma_300K_single_A'] = __sigmas
                            logger.info(system_name + '_Pm3m' + ' anharmonic score done A ' + str(__sigmas))

                            scorer = AnharmonicScore(md_frames='../vasprun_md.xml', ref_frame='./SPOSCAR',
                                                     force_constants=None, atoms=[b])
                            __sigmas, _ = scorer.structural_sigma(return_trajectory=False)
                            kvp['sigma_300K_single_B'] = __sigmas
                            logger.info(system_name + '_Pm3m' + ' anharmonic score done B ' + str(__sigmas))

                            scorer = AnharmonicScore(md_frames='../vasprun_md.xml', ref_frame='./SPOSCAR',
                                                     force_constants=None, atoms=[c])
                            __sigmas, _ = scorer.structural_sigma(return_trajectory=False)
                            kvp['sigma_300K_single_C'] = __sigmas
                            logger.info(system_name + '_Pm3m' + ' anharmonic score done C ' + str(__sigmas))

                            os.chdir('..')
                            # kvp['sigma_300K']=True
                            # kvp['sigma_300K_single']=__sigmas
                            # data['sigma_300K']=__sigmas
                            # logger.info(system_name + '_Pm3m' + ' anharmonic score done '+str(__sigmas))
                            populate_db(db, atoms, kvp, data)
                        except Exception as e:
                            print(e)
                            logger.info(system_name + '_Pm3m' + ' anharmonic score failed')
                            os.chdir('..')
                            pass

                    else:
                        logger.info(system_name + '_Pm3m' + ' Invalid')

                    os.chdir("..")
                    try:
                        shutil.rmtree(system_name + '_Pm3m')
                    except:
                        pass
                    try:
                        os.rmtree(system_name + '_Pm3m')
                    except:
                        pass


def lattice_thermal_conductivities(db):
    cwd = os.getcwd()
    system_counter = 0
    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:

                    system_counter += 1
                    logger.info("Working on system number: " + str(system_counter))
                    system_name = a + b + c
                    uid = system_name + '_Pm3m'

                    # open up the tar ball
                    cwd = os.getcwd()
                    try:
                        tf = tarfile.open(system_name + '_Pm3m.tar.gz')
                        tf.extractall()
                        os.chdir(system_name + '_Pm3m')
                    except:
                        logger.info(system_name + '_Pm3m' + ' tar ball not working')
                        continue

                    kvp = {}
                    data = {}
                    try:
                        calculator = Vasp()
                        calculator.check_convergence()
                        if calculator.completed:
                            atoms = [a for a in read_vasp_xml(index=-1)][-1]
                            kvp['uid'] = uid
                            kvp['total_energy'] = atoms.get_calculator().get_potential_energy()
                            kvp['formation_energy'] = formation_energy(atoms)
                            populate_db(db, atoms, kvp, data)
                            logger.info(system_name + '_Pm3m' + ' formation energy: ' + str(
                                kvp['formation_energy']) + ' eV/atom')
                        else:
                            logger.info(
                                system_name + '_Pm3m' + ' not converged in structure optimisation, not continuing ')
                            os.chdir("..")
                            try:
                                shutil.rmtree(system_name + '_Pm3m')
                            except:
                                pass
                            try:
                                os.rmtree(system_name + '_Pm3m')
                            except:
                                pass
                            continue
                    except:
                        logger.info(system_name + '_Pm3m' + ' formation energy: ' + str('NaN'))

                    if not os.path.isfile('./kappa-m111111.hdf5'):
                        logger.info(system_name + '_Pm3m' + ' no phono3py results available')
                    else:
                        # check whether all vasp calculations converged for this set
                        all_vasp_converged = True
                        phono3py_folder = None
                        if os.path.isfile("phono3py_2.tar.gz"):
                            phono3py_folder = "phono3py_2"
                        elif os.path.isfile("phono3py.tar.gz"):
                            phono3py_folder = 'phono3py'
                        else:
                            # no trace of phono3py results, just skip this one in case it is not a good one
                            continue

                        tf = tarfile.open(phono3py_folder + '.tar.gz')
                        tf.extractall()
                        os.chdir(phono3py_folder)  # step in the phono3py folder

                        all_directories = glob.glob("./ph-POSCAR-*")
                        for _dir in all_directories:
                            os.chdir(_dir)  # step into the folder for vasp force calculation
                            calculator = Vasp()
                            calculator.check_convergence()
                            if not calculator.completed:
                                all_vasp_converged = False
                            os.chdir("..")  # step out the folder for vasp force calculation
                        os.chdir("..")  # step out the phono3py folder

                        if all_vasp_converged:
                            logger.info(
                                system_name + '_Pm3m' + ' all VASP calculation for phono3py converged! Extracting results')
                            f = h5py.File("kappa-m111111.hdf5")
                            temperatures = f['temperature'][:]
                            for t_id, temp in enumerate(temperatures):
                                kappa = (f['kappa'][t_id][0] + f['kappa'][t_id][1] + f['kappa'][t_id][2]) / 3.0
                                kvp['kappa_' + str(int(temp))] = kappa
                                logger.info(system_name + '_Pm3m kappa ' + ' (' + str(int(temp)) + ' K): ' + str(
                                    kvp['kappa_' + str(int(temp))]))
                            populate_db(db, atoms, kvp, data)
                    os.chdir("..")  # step out this Pm3m folder
                    try:
                        shutil.rmtree(system_name + '_Pm3m')
                    except:
                        pass
                    try:
                        os.rmtree(system_name + '_Pm3m')
                    except:
                        pass

def high_order_anharmonicity(db):
    cwd = os.getcwd()
    system_counter = 0
    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:
                    system_counter += 1
                    logger.info("Working on system number: " + str(system_counter))
                    system_name = a + b + c
                    uid = system_name + '_Pm3m'
                    logger.info(system_name + '_Pm3m')

                    # open up the tar ball
                    cwd = os.getcwd()
                    try:
                        tf = tarfile.open(system_name + '_Pm3m.tar.gz')
                        tf.extractall()
                        os.chdir(system_name + '_Pm3m')
                    except:
                        logger.info(system_name + '_Pm3m' + ' tar ball not working')
                        continue

                    kvp = {}
                    data = {}

                    row = None

                    try:
                        row = db.get(selection=[('uid', '=', uid)])
                        atoms = row.toatoms()
                        kvp['uid'] = uid
                    except:
                        logger.info(
                            system_name + '_Pm3m' + ' no data, not continuing ')
                        os.chdir("..")
                        try:
                            shutil.rmtree(system_name + '_Pm3m')
                        except:
                            pass
                        try:
                            os.rmtree(system_name + '_Pm3m')
                        except:
                            pass
                        continue

                    if os.path.isfile('./sigmas.dat'):
                        f=open('./sigmas.dat','r')
                        for h,l in enumerate(f.readlines()):
                            sp = l.split()
                            if sp[-1]!='None':
                                if h==0:
                                    kvp['sigma_300K_4th_tdep_2'] = float(sp[-1])
                                if h==1:
                                    kvp['sigma_300K_4th_tdep_3'] = float(sp[-1])
                                if h==2:
                                    kvp['sigma_300K_4th_tdep_4'] = float(sp[-1])
                        try:
                            logger.info(system_name + '_Pm3m SIGMA ' + str(kvp['sigma_300K_4th_tdep_2'])+' '+str(kvp['sigma_300K_4th_tdep_3'])+' '+str(kvp['sigma_300K_4th_tdep_4']))
                        except:
                            pass
                        populate_db(db, atoms, kvp, data)

                    os.chdir("..")  # step out this Pm3m folder
                    try:
                        shutil.rmtree(system_name + '_Pm3m')
                    except:
                        pass
                    try:
                        os.rmtree(system_name + '_Pm3m')
                    except:
                        pass


def high_order_anharmonicity_old(db):
    cwd = os.getcwd()
    system_counter = 0
    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:
                    system_counter += 1
                    logger.info("Working on system number: " + str(system_counter))
                    system_name = a + b + c
                    uid = system_name + '_Pm3m'
                    logger.info(system_name + '_Pm3m')

                    row=None
                    try:
                        row = db.get(selection=[('uid', '=', uid)])
                    except:
                        pass
                    if row is not None:
                        if ('sigma_300K_tdep' in row.key_value_pairs.keys()) or ('sigma_300K_third_order' in row.key_value_pairs.keys()):
                            logger.info(system_name + '_Pm3m' + ' data already collected ')
                            continue

                    # open up the tar ball
                    cwd = os.getcwd()
                    try:
                        tf = tarfile.open(system_name + '_Pm3m.tar.gz')
                        tf.extractall()
                        os.chdir(system_name + '_Pm3m')
                    except:
                        logger.info(system_name + '_Pm3m' + ' tar ball not working')
                        continue

                    kvp = {}
                    data = {}

                    row = None

                    try:
                        row = db.get(selection=[('uid', '=', uid)])
                        atoms = row.toatoms()
                        kvp['uid'] = uid
                    except:
                        logger.info(
                            system_name + '_Pm3m' + ' no data, not continuing ')
                        os.chdir("..")
                        try:
                            shutil.rmtree(system_name + '_Pm3m')
                        except:
                            pass
                        try:
                            os.rmtree(system_name + '_Pm3m')
                        except:
                            pass
                        continue


                    if (not os.path.isfile('./kappa-m111111.hdf5')) or (not os.path.isfile('./vasprun_md.xml')) or (not os.path.isfile('./phonon.tar.gz')):
                        logger.info(system_name + '_Pm3m' + ' no available higher order phonon info ')
                        os.chdir("..")
                        try:
                            shutil.rmtree(system_name + '_Pm3m')
                        except:
                            pass
                        try:
                            os.rmtree(system_name + '_Pm3m')
                        except:
                            pass
                        continue
                    else:
                        # get the information about third-order force constants
                        if os.path.isfile("phono3py_2.tar.gz"):
                            phono3py_folder = "phono3py_2"
                        elif os.path.isfile("phono3py.tar.gz"):
                            phono3py_folder = 'phono3py'
                        else:
                            # no trace of phono3py results, just skip this one in case it is not a good one
                            continue

                        tf = tarfile.open(phono3py_folder + '.tar.gz')
                        #tf.extractall()
                        for member in tf.getmembers():
                            if "fc3.hdf5" in member.name:
                                tf.extract(member, os.getcwd())

                        try:
                            tf = tarfile.open('phonon.tar.gz')
                            tf.extractall()
                            os.chdir('./phonon')
                        except:
                            pass

                        try:
                            os.rename('force_constants.hdf5', 'f.hdf5')
                        except:
                            pass

                        logger.info(system_name + '_Pm3m' + ' 3rd order fc? ' + str(
                            os.path.exists('../' + phono3py_folder + '/fc3.hdf5')))
                        if os.path.exists('../' + phono3py_folder + '/fc3.hdf5'):
                            try:
                                scorer = AnharmonicScore(md_frames='../vasprun_md.xml', ref_frame='./SPOSCAR',
                                                        force_constants=None,
                                                        include_third_order=True,
                                                        path_to_third_order_fc='../'+phono3py_folder+'/fc3.hdf5')
                                __sigmas, _ = scorer.structural_sigma(return_trajectory=False)
                                kvp['sigma_300K_third_order'] = __sigmas
                                logger.info(system_name + '_Pm3m' + ' third order anharmonic score ' + str(__sigmas))
                            except:
                                pass
                        os.chdir('..')
                        # finish getting the information about third-order force constants

                        tdep_fc2 = None
                        try:
                            tdep_fc2 = get_temperature_dependent_second_order_fc()
                            logger.info(system_name + '_Pm3m' + ' got temperature-dependent effective force constants')
                        except:
                            pass

                        if tdep_fc2 is not None:
                            __sigmas=None
                            try:
                                scorer = AnharmonicScore(md_frames='./vasprun_md.xml', ref_frame='./POSCAR-md',
                                                         force_constants=tdep_fc2,
                                                         include_third_order=False)
                                __sigmas, _ = scorer.structural_sigma(return_trajectory=False)
                            except:
                                pass

                            if __sigmas is not None:
                                kvp['sigma_300K_tdep'] = __sigmas
                                logger.info(system_name + '_Pm3m' + ' harmonic score from TDEP ' + str(__sigmas))

                        populate_db(db, atoms, kvp, data)

                    os.chdir("..")  # step out this Pm3m folder
                    try:
                        shutil.rmtree(system_name + '_Pm3m')
                    except:
                        pass
                    try:
                        os.rmtree(system_name + '_Pm3m')
                    except:
                        pass


def get_temperature_dependent_second_order_fc():
    from ase.io import read
    import numpy as np
    from hiphive import ClusterSpace, StructureContainer
    from hiphive.utilities import get_displacements
    from hiphive import ForceConstantPotential
    from hiphive.fitting import Optimizer
    from hiphive.calculators import ForceConstantCalculator

    if os.path.exists('POSCAR-md'):
        reference_structure = read('POSCAR-md')
    else:
        return None
    if not os.path.exists('./vasprun_md.xml'):
        return None

    cs = ClusterSpace(reference_structure, [3])
    fit_structures = []
    atoms = read("./vasprun_md.xml", index=':')
    for i, a in enumerate(atoms[:800]):
        displacements = get_displacements(a, reference_structure)
        atoms_tmp = reference_structure.copy()
        atoms_tmp.new_array('displacements', displacements)
        atoms_tmp.new_array('forces', a.get_forces())
        atoms_tmp.positions = reference_structure.positions
        fit_structures.append(atoms_tmp)

    try:
        sc = StructureContainer.read("./structure_container")
        logger.info("successfully loaded the structure container")
    except:
        sc = StructureContainer(cs)  # need a cluster space to instantiate the object!
        sc.delete_all_structures()
        for ii, s in enumerate(fit_structures):
            try:
                sc.add_structure(s)
            except Exception as e:
                logger.info(ii, e)
                pass

    opt = Optimizer(sc.get_fit_data(), fit_method="ardr", train_size=0.9)
    opt.train()
    fcp = ForceConstantPotential(cs, opt.parameters)
    fcs = fcp.get_force_constants(reference_structure)
    return fcs.get_fc_array(2)


def collect(db):
    errors = []
    steps = [element_energy, full_relax_data]  # all_data]
    #steps = [high_order_anharmonicity]

    for step in steps:
        try:
            step(db)
        except Exception as x:
            print(x)
            error = '{}: {}'.format(x.__class__.__name__, x)
            errors.append(error)
    return errors


if __name__ == "__main__":
    # We use absolute path because of chdir below!
    dbname = os.path.join(os.getcwd(), 'perovskites.db')
    db = connect(dbname)

    logger.info('Established a sqlite3 database object ' + str(db))
    collect(db)
