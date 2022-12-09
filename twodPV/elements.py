"""
The elements module, holding lists of elements that will goes into
the A, B and C sites of an inorganic perovskite. This module contains
code that will set up the calculation folder for each element involved.

"""
import itertools
from pymatgen.ext.matproj import MPRester
from pymatgen import Structure
import os

from core.dao.vasp import VaspWriter
from settings import MPRest_key

#A_site_list = [['Li', 'Na', 'K', 'Rb', 'Cs'], ['Li', 'Na', 'K', 'Rb', 'Cs'], ['Mg', 'Ca', 'Sr', 'Ba'],
#               ['Li', 'Na', 'K', 'Rb', 'Cs']]
#B_site_list = [['Pb', 'Sn', 'Ge'], ['V', 'Ta', 'Nb'], ['Ti', 'Zr'], ['V', 'Ta', 'Nb']]
#C_site_list = [['F', 'Cl', 'Br', 'I'], ['F', 'Cl', 'Br', 'I'], ['O', 'S', 'Se', 'Te'], ['O', 'S', 'Se', 'Te']]

halide_C=['F', 'Cl', 'Br', 'I']
halide_A=['Li', 'Na', 'K', 'Rb', 'Cs','Cu','Ag','Au','Hg','Ga','In','Tl']
halide_B=['Mg', 'Ca', 'Sr', 'Ba','Se','Te','As','Si','Ge','Sn','Pb','Ga','In','Sc','Y','Ti','Zr','Hf','V','Nb','Ta','Cr','Mo','W','Mn','Tc','Re','Fe','Ru','Os','Co','Rh','Ir','Ni','Pd','Pt','Cu','Ag','Au','Zn','Cd','Hg']

chalco_C=['O','S','Se']
chalco_A=['Be','Mg','Ca','Sr','Ba','Ra','Ti','V','Cr','Mn','Fe','Co','Ni','Pd','Pt','Cu','Ag','Zn','Cd','Hg','Ge','Sn','Pb']
chalco_B=['Ti','Zr','Hf','V','Nb','Ta','Cr','Mo','W','Mn','Tc','Re','Fe','Ru','Os','Co','Rh','Ir','Ni','Pd','Pt','Sn','Ge','Pb','Si','Te','Po']

A_site_list=[halide_A,chalco_A]
B_site_list=[halide_B,chalco_B]
C_site_list=[halide_C,chalco_C]


all_elements_list = list(itertools.chain(*[A_site_list, B_site_list, C_site_list]))
all_elements_list = list(itertools.chain(*all_elements_list))
all_elements_list = list(set(all_elements_list))


if MPRest_key=="":
    raise Exception("Rest service key to Materials Project needed to connect to the database! Please set it in setting.py")

# Connect to the Materials Project to find out the lowest energy structure for the elemental phase of the solid
# and set up a set of folders for the geometry optimisations accordingly.
mpr = MPRester(MPRest_key)
for element in all_elements_list:
    qs = mpr.query(criteria={"elements": {"$all": [element]}, "nelements": 1},
                   properties=["material_id", "pretty_formula", "formation_energy_per_atom", "cif", "input.kpoints"])

    max_energy = 10000000
    lowest_energy_id = None
    for e in qs:
        if e['formation_energy_per_atom'] < max_energy:
            max_energy = e['formation_energy_per_atom']
            lowest_energy_id = e['material_id']
            lowest_cif = e['cif']
            lowest_k = e["input.kpoints"]

    cwd = os.getcwd()
    wd = cwd + '/elements/' + str(element)

    if not os.path.exists(wd):
        os.makedirs(wd)
    os.chdir(wd)

    # POSCAR
    structure = Structure.from_str(lowest_cif, fmt="cif")
    poscar = structure.to(fmt="poscar")
    f = open('POSCAR', 'w')
    for l in poscar:
        f.write(l)
    f.close()

    ## KPOINTS
    kpoints = lowest_k
    kpoints.write_file('KPOINTS')

    ## INCAR and POSCAR
    #VaspWriter().write_INCAR(default_options=default_bulk_optimisation_set)
    #VaspWriter().write_potcar(structure)

    os.chdir(cwd)