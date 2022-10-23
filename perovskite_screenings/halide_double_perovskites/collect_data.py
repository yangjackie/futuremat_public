import copy
import os, glob, tarfile, shutil
from ase.db import connect
from ase.io.vasp import *
from core.utils.loggings import setup_logger
from core.calculators.vasp import Vasp
from perovskite_screenings.collect_data import _atom_dict, populate_db
import phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

from core.external.vasp.anharmonic_score import *
import glob

from phonopy.interface.calculator import read_crystal_structure, write_crystal_structure
from phonopy.interface.vasp import parse_set_of_forces
from phonopy.file_IO import write_force_constants_to_hdf5, write_FORCE_SETS, parse_disp_yaml, write_disp_yaml
from phonopy import Phonopy

reference_atomic_energies = {}


def formation_energy(atoms):
    fe = atoms.get_calculator().get_potential_energy()
    # print(fe,_atom_dict(atoms),reference_atomic_energies)
    for k in _atom_dict(atoms).keys():
        fe = fe - _atom_dict(atoms)[k] * reference_atomic_energies[k]
        #print(k,reference_atomic_energies[k],fe)
    return fe / atoms.get_number_of_atoms()


def element_energy(db):
    logger.info("========== Collecting reference energies for constituting elements ===========")
    cwd = os.getcwd()
    os.chdir('./elements')

    all_element_zips = glob.glob('*.tar.gz')

    for zip in all_element_zips:
        kvp = {}
        data = {}
        element = zip.replace('.tar.gz', '')
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


class System(object):

    def __init__(self, name, kvp={}, data={}):
        self.name = name
        self.kvp = kvp
        self.data = data
        self.atoms = None

    def __str__(self):
        return self.name

    @property
    def populate_energy(self):
        if ('total_energy' in self.kvp.keys()) and ('formation_energy' in self.kvp.keys()):
            return True
        else:
            return False

    @property
    def populate_frequencies(self):
        if ("G_min_ph_freq" in self.kvp.keys()):
            return True
        else:
            return False

    @property
    def populate_anharmonic_scores(self):
        if 'sigma_300K_single' in self.kvp.keys():
            return True
        else:
            return False


def get_formation_energy(system):
    try:
        calculator = Vasp()
        calculator.check_convergence()
        if calculator.completed:
            system.atoms = [a for a in read_vasp_xml(index=-1)][-1]
            system.kvp['uid'] = system.name
            system.kvp['total_energy'] = system.atoms.get_calculator().get_potential_energy()
            system.kvp['formation_energy'] = formation_energy(system.atoms)
            logger.info(system.name + ' total energy: ' + str(system.kvp['total_energy']) + ' eV/atom')
            logger.info(system.name + ' formation energy: ' + str(system.kvp['formation_energy']) + ' eV/atom')
        else:
            logger.error(system.name + ' NOT CONVERGED!')
            return system
    except Exception as e:
        logger.info(e)
        logger.info(system.name + ' formation energy: ' + str('NaN'))
    return system


def get_softest_mode_frequencies(system):
    try:
        phonon = phonopy.load(supercell_matrix=[2, 2, 2], primitive_matrix='auto',
                              unitcell_filename='POSCAR',
                              force_constants_filename='force_constants.hdf5')
        path = [[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.25, 0.75], [0.5, 0.0, 0.5]]]
        labels = ["G", "L", "W", "X"]
        qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=10)
        phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)
        phonon_dict = phonon.get_band_structure_dict()

        for _pp, p in enumerate(path[0]):
            for _i, qset in enumerate(phonon_dict['qpoints']):
                for _j, _q in enumerate(qset):
                    if (_q[0] == p[0]) and (_q[1] == p[1]) and (_q[2] == p[2]):
                        system.kvp[labels[_pp] + "_min_ph_freq"] = min(phonon_dict['frequencies'][_i][_j])

        for _pp, p in enumerate(path[0]):
            logger.info(system.name + ' ' + labels[_pp] + "_min_ph_freq:" + str(
                system.kvp[labels[_pp] + "_min_ph_freq"]))
    except:
        pass
    return system

def get_frequency_averaged_weighted_by_sigma(system):
    if not os.path.exists("./SPOSCAR"):
        try:
            unitcell, _ = read_crystal_structure('./CONTCAR', interface_mode='vasp')
            supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
            phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix)
            phonon.generate_displacements()
            write_crystal_structure('SPOSCAR', phonon.supercell)
        except:
            return system

    try:
        import numpy as np
        scorer = AnharmonicScore(md_frames=glob.glob('./MD/vasprun_prod*.xml'), ref_frame='./SPOSCAR',
                                force_constants='force_constants.hdf5',   unit_cell_frame='./SPOSCAR',
                                mode_resolved=True)
        eigen_vals, sigmas = scorer.mode_resolved_sigma()
        eigen_vals=np.array(eigen_vals)
        sigmas=np.array(sigmas)
        system.kvp['sigma_mode_averaged_300K']=np.dot(eigen_vals,sigmas)/np.sum(sigmas)
        logger.info(system.name + ' weighted frequency is: '+str(system.kvp['sigma_mode_averaged_300K']))
    except:
        pass
    return system

def get_anharmonic_score(system):
    if not os.path.exists("./SPOSCAR"):
        try:
            unitcell, _ = read_crystal_structure('./CONTCAR', interface_mode='vasp')
            supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
            phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix)
            phonon.generate_displacements()
            write_crystal_structure('SPOSCAR', phonon.supercell)
        except:
            return system

    try:
        #NOTE - the force set file may have some problems, use force_constants.hdf5 is safer
        scorer = AnharmonicScore(md_frames=glob.glob('./MD/vasprun_prod*.xml'), ref_frame='./SPOSCAR',
                                 force_constants='force_constants.hdf5',unit_cell_frame='./SPOSCAR',primitive_matrix='auto')
                                 #force_sets_filename='FORCE_SETS')
        sigma, _ = scorer.structural_sigma(return_trajectory=False)
        system.kvp['sigma_300K_single'] = sigma
        logger.info(system.name + ' anharmonic score: ' + str(sigma))
    except Exception as e:
        logger.info(e)
        pass

    return system

def collect_this(system):
    cwd = os.getcwd()
    try:
        __open_system_tar(system.name)
    except:
        logger.error(system.name + ' - tar ball not working, skip')
        os.chdir(cwd)
        return system

    logger.info("Working on system " + str(system))

    system = get_formation_energy(system)
    if system.populate_energy:
        #system = get_softest_mode_frequencies(system)
        #system = get_anharmonic_score(system)
        system = get_frequency_averaged_weighted_by_sigma(system)

    os.chdir(cwd)
    clear_system(system.name)
    return system

def all_data_parallel(db):
    import multiprocessing
    s = SystemIterator()
    systemIterator = iter(s)
    counter = 1
    this_batch = []
    for s in systemIterator:
        system = System(s)
        this_batch.append(system)
        counter += 1

        if counter==28:
            pool = multiprocessing.Pool(28)
            p = pool.map_async(collect_this, this_batch)
            data = p.get(9999999)
            pool.terminate()
            for returned_system in data:
                if returned_system.populate_energy:
                    try:
                        populate_db(db,returned_system.atoms, returned_system.kvp, returned_system.data)
                    except:
                        pass
            this_batch=[]
            counter=1

    if this_batch!=[]:
        pool = multiprocessing.Pool(len(this_batch))
        p = pool.map_async(collect_this, this_batch)
        data = p.get(9999999)
        pool.terminate()
        for returned_system in data:
            if returned_system.populate_energy:
                try:
                    populate_db(db, returned_system.atoms, returned_system.kvp, returned_system.data)
                except:
                    pass

def soc_energies(db):
    s = SystemIterator()
    systemIterator = iter(s)
    for s in systemIterator:
        cwd = os.getcwd()
        row = None
        try:
            row = db.get(selection=[('uid', '=', s)])
        except:
            pass

        try:
            system = System(s)
            __open_system_tar(system.name)
        except:
            logger.error(system.name + ' - tar ball not working, skip')
            os.chdir(cwd)
            continue

        if row is not None:
            system.kvp = copy.deepcopy(row.key_value_pairs)
            system.data = copy.deepcopy(row.data)

            if os.path.exists('./SOC'):
                os.chdir('./SOC')
                try:
                    calculator = Vasp()
                    calculator.check_convergence()
                    if calculator.completed:
                        system.atoms = [a for a in read_vasp_xml(index=-1)][-1]
                        system.kvp['total_energy_soc'] = system.atoms.get_calculator().get_potential_energy()
                        system.kvp['formation_energy_soc'] = formation_energy(system.atoms)
                        logger.info(system.name + ' total energy: ' + str(system.kvp['total_energy_soc']) + ' eV/atom')
                        logger.info(
                            system.name + ' formation energy: ' + str(system.kvp['formation_energy_soc']) + ' eV/atom')
                    else:
                        logger.error(system.name + ' NOT CONVERGED!')
                except Exception as e:
                    logger.info(e)
                    logger.info(system.name + ' formation energy: ' + str('NaN'))
                os.chdir('../')

                try:
                    populate_db(db, system.atoms, system.kvp, system.data)
                except:
                    pass

        os.chdir(cwd)
        clear_system(system.name)


def all_data(db):
    s = SystemIterator()
    systemIterator = iter(s)
    for s in systemIterator:
        sigma = None
        try:
            row = db.get(selection=[('uid', '=', s)])
            sigma = row.key_value_pairs['sigma_300K_single']
        except:
            pass

        if sigma is not None:
            logger.info(s + ' anharmonic score: ' + str(sigma))

            system = System(s)
            cwd = os.getcwd()
            try:
                __open_system_tar(system.name)
            except:
                logger.error(system.name + ' - tar ball not working, skip')
                os.chdir(cwd)
                continue

            logger.info("Working on system " + str(system))

            system.kvp = copy.deepcopy(row.key_value_pairs)
            system.data = copy.deepcopy(row.data)

            system = get_formation_energy(system)

            if system.populate_energy:
                try:
                    populate_db(db, system.atoms, system.kvp, system.data)
                except:
                    pass

            row = db.get(selection=[('uid', '=', s)])
            sigma = row.key_value_pairs['sigma_300K_single']
            logger.info("after data update, sigma " + str(sigma))

            os.chdir(cwd)
            clear_system(system.name)

        else:
            system = System(s)
            cwd = os.getcwd()
            try:
                __open_system_tar(system.name)
            except:
                logger.error(system.name + ' - tar ball not working, skip')
                os.chdir(cwd)
                continue

            logger.info("Working on system " + str(system))

            system = get_formation_energy(system)
            if system.populate_energy:
                try:
                    populate_db(db, system.atoms, system.kvp, system.data)
                except:
                    pass

                system = get_softest_mode_frequencies(system)
                if system.populate_frequencies:
                    try:
                        populate_db(db, system.atoms, system.kvp, system.data)
                    except:
                        pass

                system = get_anharmonic_score(system)
                if system.populate_anharmonic_scores:
                    try:
                        populate_db(db, system.atoms, system.kvp, system.data)
                    except:
                        pass

            os.chdir(cwd)
            clear_system(system.name)


def __open_system_tar(system):
    system_tar = system + '.tar.gz'
    tf = tarfile.open(system_tar)
    tf.extractall()
    os.chdir(system)


def clear_system(system):
    try:
        shutil.rmtree(system)
    except:
        pass
    try:
        os.rmtree(system)
    except:
        pass


class SystemIterator():

    def __init__(self):
        self.all_systems = glob.glob('dpv_*.tar.gz')
        self.all_systems = list(sorted(self.all_systems))
        self.all_systems = [x.replace('.tar.gz', '') for x in self.all_systems]
        print('all initialised')

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter < len(self.all_systems):
            x = self.all_systems[self.counter]
            self.counter += 1
            logger.info('Counting systems: '+str(self.counter)+'/'+str(len(self.all_systems)))
            return x
        else:
            raise StopIteration


class DataCollector():

    def __init__(self, db=None):
        self.db = db


def collect(db):
    errors = []
    steps = [element_energy,soc_energies]

    for step in steps:
        try:
            step(db)
        except Exception as x:
            error = '{}: {}'.format(x.__class__.__name__, x)
            logger.info(error)
            errors.append(x)
            continue
    return errors


if __name__ == "__main__":
    dbname = os.path.join(os.getcwd(), 'double_halide_pv.db')
    logger = setup_logger(output_filename='data_collector.log')
    db = connect(dbname)
    collect(db)
