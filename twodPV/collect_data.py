"""
This module contains methods to collect results from the calculation folder
and stored them in a proper object-relational-database. The io functionaities
from the ase packages will be used because it already has a good interface to handle
all the database operations. In this case, the data structures from ASE will
also be used rather than our internal ones. This also allows our data to be
more easily accessible by other users using the existing ase functionalities.
"""

import os
import glob
import numpy as np
import math
import zipfile
import time

from core.calculators.vasp import Vasp
from core.dao.vasp import VaspReader, VaspWriter
from core.utils.loggings import setup_logger

logger = setup_logger(output_filename='data_collector.log')
# logger = setup_logger()

from twodPV.analysis.electronic_permitivity import get_geometry_corrected_electronic_polarizability
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.core import Spin
from scipy.interpolate import interp1d

from ase.io.vasp import *
from ase.db import connect

from twodPV.bulk_library import A_site_list, B_site_list, C_site_list

reference_atomic_energies = {}


def _atom_dict(atoms):
    """
    Get a dictionary of number of each element in the chemical structure.
    """
    unique = list(set(atoms.get_chemical_symbols()))
    return {u: atoms.get_chemical_symbols().count(u) for u in unique}


def updater(existing_dict, new_dict):
    for ek in existing_dict.keys():
        if ek not in new_dict.keys():
            new_dict[ek] = existing_dict[ek]


def populate_db(db, atoms, kvp, data):
    logger.info("populating database, uid=" + str(kvp['uid']))
    row = None
    try:
        row = db.get(selection=[('uid', '=', kvp['uid'])])
        logger.info("Updating an existing row.")

        ##############################################################################################################
        # Always update an existing row with the data extracted from the latest calculations
        # unless there is a piece of data that already exists in the database but does not exist
        # in the new data, then leave the old data untouched. (this is done by revsely updating new data
        # with these missing info that pre-existed in the database, and update the database with the newly
        # packaged dictionaries).
        ##############################################################################################################
        if row.key_value_pairs is not None:
            updater(row.key_value_pairs, kvp)
        if row.data is not None:
            updater(row.data, data)
        if atoms is None:
            atoms = row.toatoms()
        db.write(atoms, data=data, id=row.id, **kvp)
    except KeyError:
        try:
            db.write(atoms, data=data, **kvp)
        except Exception as e:
            logger.warning(e)


def formation_energy(atoms):
    fe = atoms.get_calculator().get_potential_energy()
    for k in _atom_dict(atoms).keys():
        fe = fe - _atom_dict(atoms)[k] * reference_atomic_energies[k]
    return fe / atoms.get_number_of_atoms()


def element_energy(db):
    print("========== Collecting reference energies for constituting elements ===========")
    cwd = os.getcwd()
    element_directory = cwd + '/elements/'
    for dir in [o for o in os.listdir(element_directory) if os.path.isdir(os.path.join(element_directory, o))]:
        kvp = {}
        data = {}

        dir = os.path.join(element_directory, dir)
        uid = 'element_' + str(dir.split('/')[-1])

        os.chdir(dir)
        calculator = Vasp()
        calculator.check_convergence()
        if calculator.completed:
            atoms = [i for i in read_vasp_xml(index=-1)][-1]  # just to be explicit that we want the very last one
            e = list(_atom_dict(atoms).keys())[-1]
            reference_atomic_energies[e] = atoms.get_calculator().get_potential_energy() / _atom_dict(atoms)[e]
            kvp['uid'] = uid
            kvp['total_energy'] = atoms.get_calculator().get_potential_energy()
            populate_db(db, atoms, kvp, data)
        else:
            raise Exception("Vasp calculation incomplete in " + dir + ". Please check!")
        os.chdir(cwd)


def pm3m_formation_energy(db):
    print("========== Collecting formation energies for bulk perovskites in Pm3m symmetry ===========")
    cwd = os.getcwd()
    base_dir = cwd + '/relax_Pm3m/'
    kvp = {}
    data = {}
    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:
                    system_name = a + b + c
                    uid = system_name + '3_pm3m'

                    dir = os.path.join(base_dir, system_name + "_Pm3m")
                    os.chdir(dir)
                    calculator = Vasp()
                    calculator.check_convergence()
                    if calculator.completed:
                        atoms = [k for k in read_vasp_xml(index=-1)][-1]
                        kvp['uid'] = uid
                        kvp['total_energy'] = atoms.get_calculator().get_potential_energy()
                        kvp['formation_energy'] = formation_energy(atoms)
                        print("System " + uid + " formation energy :" + str(kvp['formation_energy']) + ' eV')
                        populate_db(db, atoms, kvp, data)
                    os.chdir(cwd)


def __pm3m_phonon_frequencies(db):
    print("========== Collecting phonon frequencies for bulk perovskites in Pm3m symmetry ===========")
    property_populator(system='pm3m', property='phonon', db=db)


def __two_d_100_AO_phonon_frequencies(db):
    for thickness in [3, 5, 7, 9]:
        system = '100_AO_' + str(thickness)
        property_populator(system=system, property='phonon', db=db)


def __two_d_100_BO2_phonon_frequencies(db):
    for thickness in [3, 5, 7, 9]:
        system = '100_BO2_' + str(thickness)
        property_populator(system=system, property='phonon', db=db)


def __two_d_110_ABO_phonon_frequencies(db):
    for thickness in [3, 5, 7, 9]:
        system = '110_ABO_' + str(thickness)
        property_populator(system=system, property='phonon', db=db)


def __two_d_110_O2_phonon_frequencies(db):
    for thickness in [3, 5, 7, 9]:
        system = '110_O2_' + str(thickness)
        property_populator(system=system, property='phonon', db=db)


def __two_d_111_AO3_phonon_frequencies(db):
    for thickness in [3, 5, 7, 9]:
        system = '111_AO3_' + str(thickness)
        property_populator(system=system, property='phonon', db=db)


def __two_d_111_B_phonon_frequencies(db):
    for thickness in [3, 5, 7, 9]:
        system = '111_B_' + str(thickness)
        property_populator(system=system, property='phonon', db=db)


def property_populator(property='phonon', db=None, system=None):
    cwd = os.getcwd()

    if system is 'pm3m':
        base_dir = cwd + '/relax_Pm3m/'
    else:
        base_dir = cwd + '/slab_' + system.replace(system.split("_")[-1], 'small')
        if 'AO3_3' in system:
            base_dir = cwd + '/slab_' + system.replace("_3", "_small")

    kvp = {}
    data = {}
    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:
                    chemistry = a + b + c

                    if system is 'pm3m':
                        uid = chemistry + '3_pm3m'
                        system_folder = chemistry + '_Pm3m'
                        kvp['uid'] = uid
                    else:
                        uid = chemistry + '3_' + str(system)
                        system_folder = chemistry + "_" + str(system.split("_")[-1])
                        kvp['uid'] = uid

                    if property is 'phonon':
                        dir = os.path.join(base_dir, system_folder + "/phonon_G")
                        print(str(dir))
                        try:
                            reader = VaspReader(input_location=dir + '/OUTCAR')
                            freqs = reader.get_vibrational_eigenfrequencies_from_outcar()
                            print(freqs)
                            data['gamma_phonon_freq'] = np.array(freqs)
                            populate_db(db, None, kvp, data)
                        except:
                            pass


def randomised_structure_formation_energy(db):
    print("========== Collecting formation energies for distorted perovskites  ===========")
    cwd = os.getcwd()
    base_dir = cwd + '/relax_randomized/'
    kvp = {}
    data = {}
    counter = 0
    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:
                    system_name = a + b + c
                    all_rand_for_this = glob.glob(base_dir + '/' + system_name + '*rand*')
                    for r in all_rand_for_this:
                        uid = system_name + '3_random_str_' + str(r.split("_")[-1])
                        os.chdir(r)
                        try:
                            calculator = Vasp()
                            calculator.check_convergence()
                            if calculator.completed:
                                atoms = [k for k in read_vasp_xml(index=-1)][-1]
                                kvp['uid'] = uid
                                kvp['total_energy'] = atoms.get_calculator().get_potential_energy()
                                kvp['formation_energy'] = formation_energy(atoms)
                                counter += 1
                                print(str(counter) + '\t' + "System " + uid + " formation energy :" + str(
                                    kvp['formation_energy']) + ' eV')
                                populate_db(db, atoms, kvp, data)
                        except:
                            continue  # if job failed we dont worry too much about it
                        os.chdir(cwd)


def two_d_formation_energies(db, orientation='100', termination='AO2', thicknesess=[3, 5, 7, 9], large_cell=False):
    cwd = os.getcwd()

    if not large_cell:
        base_dir = cwd + '/slab_' + str(orientation) + '_' + str(termination) + '_small/'
    else:
        base_dir = cwd + '/slab_' + str(orientation) + '_' + str(termination) + '_full_B_octa/'

    kvp = {}
    data = {}
    counter = 0
    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:
                    system_name = a + b + c
                    for thick in thicknesess:
                        work_dir = base_dir + system_name + "_" + str(thick)
                        os.chdir(work_dir)
                        if not large_cell:
                            uid = system_name + '3_' + str(orientation) + "_" + str(termination) + "_" + str(thick)
                        else:
                            uid = system_name + '3_' + str(orientation) + "_" + str(termination) + "_" + str(
                                thick) + "_large_cell_full_B_octa"
                        try:
                            calculator = Vasp()
                            calculator.check_convergence()
                            if calculator.completed:
                                atoms = [k for k in read_vasp_xml(index=-1)][-1]
                                kvp['uid'] = uid
                                kvp['total_energy'] = atoms.get_calculator().get_potential_energy()
                                kvp['formation_energy'] = formation_energy(atoms)
                                kvp['orientation'] = '[' + str(orientation) + ']'
                                kvp['termination'] = termination
                                kvp['nlayer'] = thick
                                counter += 1
                                print(str(counter) + '\t' + "System " + uid + " formation energy :" + str(
                                    kvp['formation_energy']) + ' eV')
                                populate_db(db, atoms, kvp, data)
                        except:
                            print("-------------> failed in " + str(os.getcwd()))
                            continue
                        os.chdir(cwd)


def __two_d_100_AO_energies(db):
    two_d_formation_energies(db, orientation='100', termination='AO')


def __two_d_100_BO2_energies(db):
    two_d_formation_energies(db, orientation='100', termination='BO2')


def __two_d_100_AO_energies_large_cell(db):
    two_d_formation_energies(db, orientation='100', termination='AO', large_cell=True, thicknesess=[3])


def __two_d_100_BO2_energies_large_cell(db):
    two_d_formation_energies(db, orientation='100', termination='BO2', large_cell=True, thicknesess=[3])


def __two_d_111_B_energies(db):
    two_d_formation_energies(db, orientation='111', termination='B')


def __two_d_111_AO3_energies(db):
    two_d_formation_energies(db, orientation='111', termination='AO3')


def __two_d_110_O2_energies(db):
    two_d_formation_energies(db, orientation='110', termination='O2')


def __two_d_110_ABO_energies(db):
    two_d_formation_energies(db, orientation='110', termination='ABO')


# ==========================================================
# Data collections for electronic structure calculations
# ==========================================================
def two_d_electronic_structures(db, orientation='100', termination='AO'):
    cwd = os.getcwd()
    base_dir = cwd + '/slab_' + str(orientation) + '_' + str(termination) + '_small/'
    thicknesess = [3, 5, 7, 9]

    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:
                    system_name = a + b + c
                    for thick in thicknesess:
                        kvp = {}
                        data = {}
                        uid = system_name + '3_' + str(orientation) + "_" + str(termination) + "_" + str(thick)
                        kvp['uid'] = uid
                        logger.info("Getting data for system  " + str(
                            uid) + ' from directory ' + base_dir + '/' + system_name + '_' + str(thick))
                        os.chdir(base_dir + '/' + system_name + '_' + str(thick))

                        #                        # get the two-dimensional electronic permitivity
                        #                        try:
                        #                            data['e_polarizability_freq'] = get_geometry_corrected_electronic_polarizability()
                        #                        except:
                        #                            logger.info("Cannot get frequency-dependent electronic polarizability")

                        logger.info("\n" + "============= out_of_plane_charge_polarisations ==============")
                        kvp = get_out_of_plane_charge_polarisations(kvp)

                        logger.info("\n" + "============= dos_related_properties ==============")
                        kvp = get_dos_related_properties(kvp)

                        logger.info("\n" + "============= band_structures_properties =============")
                        data = get_band_structures_properties(data)

                        logger.info("\n" + "====Summary of band structure information====")

                        for k in data['band_structure'].keys():
                            if isinstance(data['band_structure'][k], list):
                                for kk in range(len(data['band_structure'][k])):
                                    for kkk in data['band_structure'][k][kk].keys():
                                        logger.info(
                                            str(k) + ' ' + str(kkk) + ' ' + str(data['band_structure'][k][kk][kkk]))
                            else:
                                logger.info(str(k) + ' ' + str(data['band_structure'][k]))

                        logger.info("\n")
                        populate_db(db, None, kvp, data)
                        os.chdir(cwd)


def get_dos_related_properties(kvp):
    dos_run = None
    try:
        with zipfile.ZipFile('./electronic_workflow.zip') as z:
            with open("./vasprun_SPIN_CHG_temp.xml", 'wb') as f:
                f.write(z.read("electronic_workflow/vasprun_SPIN_CHG.xml"))
        f.close()
        z.close()
    except:
        try:
            with zipfile.ZipFile('./electronic_workflow.zip') as z:
                with open("./vasprun_SPIN_CHG_temp.xml", 'wb') as f:
                    f.write(z.read("vasprun_SPIN_CHG.xml"))
            f.close()
            z.close()
        except:
            logger.info("Loading results from spin-polarised charge density runs unsuccessful ")
            try:
                os.remove('./vasprun_SPIN_CHG_temp.xml')
            except:
                pass
            return kvp

    # need to write out the POTCAR so reading vasprun.xml will work
    try:
        __xtal = VaspReader(input_location='./POSCAR').read_POSCAR()
        VaspWriter().write_potcar(__xtal)
    except:
        logger.warning("Loading POSCAR failed ")

    try:
        dos_run = Vasprun("./vasprun_SPIN_CHG_temp.xml")
    except:
        logger.warning("Loading vasprun.xml failed ")
        dos_run = None

    if dos_run is not None:
        try:
            kvp['ef_dos'] = dos_run.efermi
            dos = dos_run.complete_dos
            sdos = dos.get_smeared_densities(sigma=0.125)
            en = dos.as_dict()['energies']

            spin_up_dos = sdos[Spin.up]
            spin_up_dos = interp1d(en, spin_up_dos, kind='cubic')
            kvp['spin_up_dos_at_ef'] = float(spin_up_dos(dos_run.efermi))

            spin_down_dos = sdos[Spin.down]
            spin_down_dos = interp1d(en, spin_down_dos, kind='cubic')
            kvp['spin_down_dos_at_ef'] = float(spin_down_dos(dos_run.efermi))
            logger.info("Spin up at Ef " + str(kvp['spin_up_dos_at_ef']) + " Spin down at Ef " + str(
                kvp['spin_down_dos_at_ef']))
        except:
            pass
    try:
        os.remove("./vasprun_SPIN_CHG_temp.xml")
        os.remove("./POTCAR")
    except:
        pass
    del dos_run
    return kvp


def get_out_of_plane_charge_polarisations(kvp):
    from core.models.vector3d import cVector3D
    from core.models.element import atomic_numbers

    efolder_loaded = False
    try:
        with zipfile.ZipFile('./electronic_workflow.zip') as z:
            with open("./CHGCAR_temp", 'wb') as f:
                f.write(z.read("electronic_workflow/CHGCAR_SPIN"))
        logger.info("Spin--polarized charge density loaded")
        efolder_loaded = True
        f.close()
        z.close()
    except:
        logger.info(
            "Error in parsing CHGCAR from spin-polarized calculation in ./electronic_workflow, try with spin non polarised charge densities")
        try:
            with zipfile.ZipFile('./electronic_workflow.zip') as z:
                with open("./CHGCAR_temp", 'wb') as f:
                    f.write(z.read("electronic_workflow/CHGCAR_NOSPIN"))
            logger.info("Spin--non-polarized charge density loaded")
            f.close()
            z.close()
            efolder_loaded = True
        except:
            logger.info(
                "Error in parsing CHGCAR from spin-non-polarized calculation in ./electronic_workflow, try another folder")

    if not efolder_loaded:
        try:
            with zipfile.ZipFile('./electronic_workflow.zip') as z:
                with open("./CHGCAR_temp", 'wb') as f:
                    f.write(z.read("CHGCAR_SPIN"))
            f.close()
            z.close()
            logger.info("Spin--polarized charge density loaded")
        except:
            logger.info(
                "Error in parsing CHGCAR from spin-polarized calculation, try with spin non polarised charge densities")
            try:
                with zipfile.ZipFile('./electronic_workflow.zip') as z:
                    with open("./CHGCAR_temp", 'wb') as f:
                        f.write(z.read("CHGCAR_NOSPIN"))
                f.close()
                z.close()
                logger.info("Spin--non-polarized charge density loaded")
            except:
                logger.warning("Error in parsing CHGCAR from spin-non-polarized calculation, skip")
                try:
                    os.remove('./CHGCAR_temp')
                except:
                    pass
                return kvp

    logger.info("")

    try:
        vasp_reader = VaspReader(input_location='./CHGCAR_temp')
        #######################################
        # TODO - THIS ROUNTINE IS VERY SLOW!!!!!
        # TODO - THIS ROUNTINE IS VERY SLOW!!!!!
        #######################################
        charge_grid, crystal = vasp_reader.read_CHGCAR()

        a = cVector3D(crystal.lattice.lattice_vectors[0][0], crystal.lattice.lattice_vectors[0][1],
                      crystal.lattice.lattice_vectors[0][2])
        b = cVector3D(crystal.lattice.lattice_vectors[1][0], crystal.lattice.lattice_vectors[1][1],
                      crystal.lattice.lattice_vectors[1][2])
        area = a.cross(b).l2_norm()

        NGX = charge_grid.shape[0]
        NGY = charge_grid.shape[1]
        NGZ = charge_grid.shape[2]
        resolution_x = crystal.lattice.a / NGX
        resolution_y = crystal.lattice.b / NGY
        resolution_z = crystal.lattice.c / NGZ

        electron_densities = 0
        for z_value in range(NGZ):
            z_plane = charge_grid[:, :, z_value] / crystal.lattice.volume
            z_plane_total = np.sum(z_plane) * (resolution_x * resolution_y)
            electron_densities -= z_plane_total * (0 + resolution_z * z_value) * resolution_z

        nuclear_densities = 0
        for a in crystal.asymmetric_unit[0].atoms:
            nuclear_densities += get_zval_from_potcar(a.label) * a.position.z

        kvp['e_pol'] = electron_densities / area
        kvp['nu_pol'] = nuclear_densities / area
        logger.info('electronic polarisation: ' + str(kvp['e_pol']) + ' nuclear polarisation: ' + str(kvp['nu_pol']))
    except:
       pass

    for i in ['CHGCAR_temp', 'CHGCAR_SPIN_temp', 'CHGCAR_NOSPIN_temp']:
        try:
            os.remove('./' + str(i))
        except:
            pass
    return kvp


def get_zval_from_potcar(label, use_GW=True):
    import settings
    from core.models.element import pbe_pp_choices
    import os
    z = 0
    if not use_GW:
        potcar = settings.vasp_pp_directory + '/' + pbe_pp_choices[label] + '/POTCAR'
    else:
        if os.path.exists(settings.vasp_pp_directory + '/' + pbe_pp_choices[label] + '_GW/'):
            potcar = settings.vasp_pp_directory + '/' + pbe_pp_choices[label] + '_GW/POTCAR'
        else:
            potcar = settings.vasp_pp_directory + '/' + pbe_pp_choices[label] + '/POTCAR'

    _potcar_file = open(potcar, 'r')
    for line in _potcar_file.readlines():
        if 'ZVAL' in line:
            z = float(line.split()[5])
    _potcar_file.close()
    return z


def get_band_structures_properties(data):
    """
    Retrieve the information about the band structure from vasp calculations. This basically resembles the sumo-bandstats
    code with snipplets borrowed from Pymatgen. The major difference here is we get the band extrema and effective masses
    along different reciprocal space directions for each spin channel. So it provides additional flexibilities for users to
    extract quantitites such as spin-up/spin-down electronic gaps separately.
    """
    # TODO - A bit tedious, contains repetitive codes that can be further tidied up! (26/03/2020)

    from pymatgen.io.vasp.outputs import BSVasprun
    from pymatgen.electronic_structure.core import Spin
    from sumo.electronic_structure.bandstructure import get_reconstructed_band_structure
    from sumo.electronic_structure.effective_mass import (get_fitting_data, fit_effective_mass)
    from twodPV.calculators import KPOINTS_string_dict
    from twodPV.calculators import KPOINTS_string_dict
    import numpy as np

    if '100' in os.getcwd():
        orient = '100'
    elif '110' in os.getcwd():
        orient = '110'
    elif '111' in os.getcwd():
        orient = '111'

    data['band_structure'] = {}

    try:
        with zipfile.ZipFile('./electronic_workflow.zip') as z:
            with open("./vasprun_spin_BAND_temp.xml", 'wb') as f:
                f.write(z.read("electronic_workflow/vasprun_spin_BAND.xml"))
        f.close()
        z.close()
        __xtal = VaspReader(input_location='./POSCAR').read_POSCAR()
        VaspWriter().write_potcar(__xtal)
        VaspWriter().write_KPOINTS(crystal=__xtal, filename='KPOINTS', kpoint_str=KPOINTS_string_dict[orient])
        vr = BSVasprun('./vasprun_spin_BAND_temp.xml')
    except:
        try:
            with zipfile.ZipFile('./electronic_workflow.zip') as z:
                with open("./vasprun_spin_BAND_temp.xml", 'wb') as f:
                    f.write(z.read("vasprun_spin_BAND.xml"))
            f.close()
            z.close()
            __xtal = VaspReader(input_location='./POSCAR').read_POSCAR()
            VaspWriter().write_potcar(__xtal)
            VaspWriter().write_KPOINTS(crystal=__xtal, filename='KPOINTS', kpoint_str=KPOINTS_string_dict[orient])
            vr = BSVasprun('./vasprun_spin_BAND_temp.xml')
        except:
            logger.info("No spin polarised band structure calculation results found! RETURN")
            for file in ['./vasprun_spin_BAND_temp.xml', "./POTCAR", "./KPOINTS", './KPOINTS_temp']:
                try:
                    os.remove(file)
                except:
                    pass
            return data
    try:
        bs = vr.get_band_structure(line_mode=True, kpoints_filename='KPOINTS')
        bs = get_reconstructed_band_structure([bs])
        for file in ['./vasprun_spin_BAND_temp.xml', "./POTCAR", "./KPOINTS", './KPOINTS_temp']:
            try:
                os.remove(file)
            except:
                pass
    except:
        logger.warning("Failed to retrieve the band structure from the vasprun.xml, RETURN")
        return data
    if not bs.is_spin_polarized:
        logger.warning("Not a spin polarised band structure. Don't want this now! RETURN")
        return data

    data['band_structure']['is_metal'] = bs.is_metal()
    if bs.is_metal():
        logger.info("=========================> METAL <=========================")
        for file in ['./vasprun_spin_BAND_temp.xml', "./POTCAR", "./KPOINTS", './KPOINTS_temp']:
            try:
                os.remove(file)
            except:
                pass
        return data

    # ======================================
    # get direct and indirect  band gap data
    # ======================================
    kpt_str = '[{k[0]:.2f}, {k[1]:.2f}, {k[2]:.2f}]'
    bg_data = bs.get_band_gap()

    data['band_structure']['band_spin_polarized'] = bs.is_spin_polarized

    if not bg_data['direct']:
        data['band_structure']['direct_band_gap'] = False
        data['band_structure']['indirect_band_gap_energy'] = bg_data['energy']
    else:
        data['band_structure']['direct_band_gap'] = True

    direct_data = bs.get_direct_band_gap_dict()
    if bs.is_spin_polarized:
        data['band_structure']['direct_band_gap_energy'] = min((spin_data['value'] for spin_data in
                                                                direct_data.values()))  # band gap defined as the smaller value between the spin-up and spin-down gaps
        data['band_structure']['direct_kindex'] = direct_data[Spin.up]['kpoint_index']
        data['band_structure']['direct_kpoint'] = kpt_str.format(
            k=bs.kpoints[data['band_structure']['direct_kindex']].frac_coords)
    else:
        data['band_structure']['direct_band_gap_energy'] = direct_data[Spin.up]['value']
        data['band_structure']['direct_kindex'] = direct_data[Spin.up]['kpoint_index']
        data['band_structure']['direct_kpoint'] = kpt_str.format(
            k=bs.kpoints[data['band_structure']['direct_kindex']].frac_coords)

    spin_label = {'1': 'up', '-1': 'down'}
    # =====valence band=====#
    list_index_band = {}
    list_index_kpoints = {}
    for spin, v in bs.bands.items():
        max_tmp = -float("inf")
        for i, j in zip(*np.where(v < bs.efermi)):
            if v[i, j] > max_tmp:
                max_tmp = float(v[i, j])
                index = j
                kpoint_vbm = bs.kpoints[j]
        data['band_structure'][spin_label[str(spin)] + "_vbm_energy"] = max_tmp
        data['band_structure'][spin_label[str(spin)] + "_vbm_kpoint"] = kpoint_vbm.frac_coords
        data['band_structure'][spin_label[str(spin)] + "_vbm_kindex"] = index

        # for this vbm at this spin, figure out equivalent kpoints
        list_index_kpoints[spin] = []
        if kpoint_vbm.label is not None:
            for i in range(len(bs.kpoints)):
                if bs.kpoints[i].label == kpoint_vbm.label:
                    list_index_kpoints[spin].append(i)
        else:
            list_index_kpoints[spin].append(index)

        # for this vbm at this spin, figure out other bands that will also cross this energy
        list_index_band[spin] = []
        for i in range(bs.nb_bands):
            if math.fabs(bs.bands[spin][i][index] - max_tmp) < 0.001:
                list_index_band[spin].append(i)

    hole_extrema = []
    data['band_structure']['hole_eff_mass'] = []
    for spin, bands in list_index_band.items():
        hole_extrema.extend([(spin, band, kpoint) for band in bands for kpoint in list_index_kpoints[spin]])
    hole_data = []
    for extrema in hole_extrema:
        hole_data.extend(get_fitting_data(bs, *extrema, num_sample_points=3))
    for d in hole_data:
        eff_mass = fit_effective_mass(d['distances'], d['energies'], parabolic=True)
        data['band_structure']['hole_eff_mass'].append({'spin': str(d['spin']),
                                                        'eff_mass': eff_mass,
                                                        'kpt_start': kpt_str.format(k=d['start_kpoint'].frac_coords),
                                                        'kpt_end': kpt_str.format(k=d['end_kpoint'].frac_coords)})

    # ===conduction band=======#
    list_index_band = {}
    list_index_kpoints = {}
    for spin, v in bs.bands.items():
        max_tmp = float("inf")
        for i, j in zip(*np.where(v >= bs.efermi)):
            if v[i, j] < max_tmp:
                max_tmp = float(v[i, j])
                index = j
                kpoint_cbm = bs.kpoints[j]
        data['band_structure'][spin_label[str(spin)] + "_cbm_energy"] = max_tmp
        data['band_structure'][spin_label[str(spin)] + "_cbm_kpoint"] = kpoint_cbm.frac_coords
        data['band_structure'][spin_label[str(spin)] + "_cbm_kindex"] = index

        # for this cbm at this spin, figure out equivalent kpoints
        list_index_kpoints[spin] = []
        if kpoint_cbm.label is not None:
            for i in range(len(bs.kpoints)):
                if bs.kpoints[i].label == kpoint_cbm.label:
                    list_index_kpoints[spin].append(i)
        else:
            list_index_kpoints[spin].append(index)

        # for this vbm at this spin, figure out other bands that will also cross this energy
        list_index_band[spin] = []
        for i in range(bs.nb_bands):
            if math.fabs(bs.bands[spin][i][index] - max_tmp) < 0.001:
                list_index_band[spin].append(i)

    electron_extrema = []
    data['band_structure']['electron_eff_mass'] = []
    for spin, bands in list_index_band.items():
        electron_extrema.extend([(spin, band, kpoint) for band in bands for kpoint in list_index_kpoints[spin]])
    electron_data = []
    for extrema in electron_extrema:
        electron_data.extend(get_fitting_data(bs, *extrema, num_sample_points=3))
    for d in electron_data:
        eff_mass = fit_effective_mass(d['distances'], d['energies'], parabolic=True)
        data['band_structure']['electron_eff_mass'].append({'spin': str(d['spin']),
                                                            'eff_mass': eff_mass,
                                                            'kpt_start': kpt_str.format(
                                                                k=d['start_kpoint'].frac_coords),
                                                            'kpt_end': kpt_str.format(k=d['end_kpoint'].frac_coords)})

    for file in ['./vasprun_spin_BAND_temp.xml', "./POTCAR", "./KPOINTS", './KPOINTS_temp']:
        try:
            os.remove(file)
        except:
            pass
    if bs is not None:
        del bs
    if vr is not None:
        del vr
    return data


def __two_d_100_AO_electronic_structures(db):
    two_d_electronic_structures(db, orientation='100', termination='AO')


def __two_d_100_BO2_electronic_structures(db):
    two_d_electronic_structures(db, orientation='100', termination='BO2')


def __two_d_111_B_electronic_structures(db):
    two_d_electronic_structures(db, orientation='111', termination='B')


def __two_d_111_AO3_electronic_structures(db):
    two_d_electronic_structures(db, orientation='111', termination='AO3')


def __two_d_110_O2_electronic_structures(db):
    two_d_electronic_structures(db, orientation='110', termination='O2')


def __two_d_110_ABO_electronic_structures(db):
    two_d_electronic_structures(db, orientation='110', termination='ABO')


def collect(db):
    errors = []
    steps = [__two_d_100_AO_electronic_structures,
             __two_d_100_BO2_electronic_structures]
    # __two_d_111_AO3_electronic_structures,
    # __two_d_111_B_electronic_structures,
    # __two_d_110_O2_electronic_structures,
    # __two_d_110_ABO_electronic_structures]
    # element_energy,  # do not skip this step, always need this to calculate formation energy on-the-fly
    # pm3m_formation_energy,
    # randomised_structure_formation_energy,
    # __two_d_100_BO2_energies_large_cell,
    # __two_d_100_AO_energies_large_cell
    # __two_d_111_B_energies,
    # __two_d_111_AO3_energies,
    # __two_d_110_O2_energies,
    # __two_d_110_ABO_energies

    # __pm3m_phonon_frequencies]
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
    dbname = os.path.join(os.getcwd(), '2dpv.db')
    db = connect(dbname)

    logger.info('Established a sqlite3 database object ' + str(db))
    collect(db)
