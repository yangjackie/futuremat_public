import argparse
import os
import sqlite3
import json
from ase.db import connect
from core.internal.builders.crystal import map_ase_atoms_to_crystal
from core.dao.vasp import VaspWriter

import ase 

def composition_dependent_demixing_energies(a, b, c,  all_keys, db):
    end_members = [_a + _b + _c for _a in a for _b in b for _c in c]
    print(end_members)
    assert (len(end_members) == 2)
    mixed_site = [site for site in [a, b, c] if len(site) == 2][-1]
    mixed_site = [''.join([_i for _i in k if not _i.isdigit()]) for k in mixed_site]
    end_member_total_energies = {k: 0 for k in mixed_site}
    # get the total energies of the two end members
    for m in mixed_site:
        for em in end_members:
            if m in em:
                matched_key = [k for k in all_keys if em in k][-1]
                row = db.get(selection=[('uid', '=', matched_key)])
                total_energy = row.key_value_pairs['total_energy']
                end_member_total_energies[m] = total_energy
                print(m, total_energy)
    # figure out which site has been mixed with two chemical elements, then we can decide
    #   the chemical compositions should be measured against which element

    mixing_energies = {}
  
    system_content = a + b + c
    for k in all_keys:
        k_contains_all_elements = all([(content in k) for content in system_content])
        if k_contains_all_elements:
            row = db.get(selection=[('uid', '=', k)])
            total_energy = row.key_value_pairs['total_energy']

            structure = map_ase_atoms_to_crystal(row.toatoms())
            element_1 = ''.join([_i for _i in mixed_site[0] if not _i.isdigit()])
            element_2 = ''.join([_i for _i in mixed_site[1] if not _i.isdigit()])
            composition = structure.all_atoms_count_dictionaries()[element_1] / (
                    structure.all_atoms_count_dictionaries()[element_1] + structure.all_atoms_count_dictionaries()[
                element_2])
            #mixing_energy = - total_energy + composition * end_member_total_energies[element_1] + (1.0 - composition) * \
            #                end_member_total_energies[element_2]
            mixing_energy = total_energy
            mixing_energy = mixing_energy / structure.total_num_atoms()

            if mixing_energy < 0: print(k)

            if composition not in mixing_energies.keys():
                mixing_energies[composition] = {}

            mixing_energies[composition][k]=mixing_energy
    return mixing_energies

#===============================================
db = connect('./doping.db')
all_uids = []
_db = sqlite3.connect('./doping.db')
cur = _db.cursor()
cur.execute("SELECT * FROM systems")
rows = cur.fetchall()

for row in rows:
    for i in row:
        if 'uid' in str(i):
            this_dict = json.loads(str(i))
            all_uids.append(this_dict['uid'])
#===============================================

cl_mixing_energies = composition_dependent_demixing_energies(['Cs'],['Pb','Sn'],['Cl3'],all_uids,db)
br_mixing_energies = composition_dependent_demixing_energies(['Cs'],['Pb','Sn'],['Br3'],all_uids,db)
i_mixing_energies = composition_dependent_demixing_energies(['Cs'],['Pb','Sn'],['I3'],all_uids,db)

print("\n")
print("=======> SUMMARY <=======")

for mixing_energies in [cl_mixing_energies,br_mixing_energies,i_mixing_energies]:
    for comp in list(sorted(mixing_energies.keys())):
        this_min = 1000000
        this_min_name = None
        for name in mixing_energies[comp].keys():
            if (mixing_energies[comp][name] < this_min):
               this_min = mixing_energies[comp][name]
               this_min_name = name
        pwd = os.getcwd()
        os.chdir('leading_edge')
        try:
            os.mkdir(this_min_name)
        except:
            pass
        os.chdir(this_min_name)
        row = db.get(selection=[('uid', '=', this_min_name)])
        structure = map_ase_atoms_to_crystal(row.toatoms())
        VaspWriter().write_structure(structure,direct=True,sort=True)
        
        #ase.io.write('POSCAR',row.toatoms(),format='vasp')
        os.chdir(pwd)
        print(comp,this_min_name,this_min)
