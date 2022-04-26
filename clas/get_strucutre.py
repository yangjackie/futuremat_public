"""
Retrieve the formation energies from Materials Project
"""

from mp_api import MPRester
from pathlib import Path
from pandas import DataFrame
import json
import os


api_key = "j04mrdwGpyW6e2O1GqVvhxDGIDd9DJhV" # Please enter your Materials Project API here. More details see: https://next-gen.materialsproject.org/api

def prepare_entries(anion_group, cation_number):
    """
    Get the metal binaries (AX) and ternaries (ABX)
    """
    mpr = MPRester(api_key)
    excluded_atoms_in_cation=["H", "C", "N", "O",
                              "F", "Cl", "Br", "I",
                              "He", "Ne", "Ar", "Kr", "Xe",
                              "Pm", "Ac","Th","Pa","U","Np","Pu"] # elements not considered in cation site
    anions = anion_group.split("-")
    for anion in anions:
        excluded_atoms_in_cation.remove(anion)
    docs = mpr.summary.search(chemsys=anion_group+"-*", nelements=cation_number+len(anions),
                              fields=["material_id",
                                      "formula_pretty",
                                      "formation_energy_per_atom",
                                      "composition_reduced",
                                      "composition",
                                      "energy_above_hull",
                                      "volume"
                                      ])
    new_docs = [d for d in docs if
                not any(x in d.composition_reduced.get_el_amt_dict().keys() for x in excluded_atoms_in_cation)]
    print("Anion: {} | Total number of structures: {}".format(anions, len(new_docs)))
    return new_docs


def filter_polymorph(entries):
    # Select the materials with lowest formation energies. **It does not mean e_above_hull = 0.
    df = DataFrame(data={"formula_pretty":[e.formula_pretty for e in entries],
                         "formation_energy_per_atom":[e.formation_energy_per_atom for e in entries]
                         },
                   index=[e.material_id for e in entries])
    min_id_list = df.groupby("formula_pretty")["formation_energy_per_atom"].idxmin().tolist()
    new_entries = [e for e in entries if e.material_id in min_id_list]
    return new_entries


def filter_atom_number(entries, atom_number):
    return [e for e in entries if sum(e.composition_reduced.get_el_amt_dict().values()) <= atom_number]


def get_solid_compounds(anion, cation_number, stable=None, number=None):
    compound_name_dict = {"H": "hydrides", "N": "nitrides", "O": "oxides", 'N-H': "nit_hydrides"}
    system_dict = {1: 'binary', 2: 'ternary', 3:'quaternary', 4:'pentanary'}
    properties_dict = {}
    dataset = {}
    compound_list = []
    entries = prepare_entries(anion, cation_number)
    if stable:
        print("Filtering polymorph...")
        entries = filter_polymorph(entries)
    if number:
        print("Filtering number...")
        entries = filter_atom_number(entries, number)
    for i in entries:
        properties_dict['material_id'] = i.material_id
        properties_dict['formula'] = i.formula_pretty
        properties_dict['formation_energy'] = i.formation_energy_per_atom
        properties_dict['volume_per_atom'] = i.volume / sum(i.composition.get_el_amt_dict().values())
        properties_dict['e_above_hull'] = i.energy_above_hull
        compound_list.append(properties_dict.copy())
    dataset[compound_name_dict[anion]] = sorted(compound_list, key=lambda d:d['material_id'])
    print("Save {} materials after the filtering.".format(len(compound_list)))
    cwd = os.getcwd()
    wd = cwd + '/data/' + system_dict[cation_number] + '/'
    if not os.path.exists(wd):
        os.makedirs(wd)
    os.chdir(wd)
    with open(compound_name_dict[anion] + '_' + system_dict[cation_number] + '_0k.json', 'w') as f:
        json.dump(dataset, f, indent=2)

def get_ele(entry, family):
    """
    entry = True:
    Get ComputedStructureEntry for all elements and save to data/ele_computed_structure_entries.json.
    When creating GibbsComputedStructureEntry, constituent element entries have to be included in the
    same list with solid compounds.
    :return: list of ComputedStructureEntry objects and store in "data/ele_computed_structure_entries.json".

    family:
    Get the ids of all the elements with the lowest energies.
    This file has to be included when balancing chemical reactions with pure elements involved (either in
    reactants or products)
    """
    with MPRester(api_key) as mpr:
        ele_docs = mpr.thermo.search(nelements=1, is_stable=True,fields=["material_id","formula_pretty","entries"])
    if entry:
        with open("data/ele_computed_structure_entries.json", "w") as f:
            json.dump([d.entries["GGA"].as_dict() for d in ele_docs], f)
    else:
        properties_dict = {}
        ele_list = []
        dataset = {}
        for i in ele_docs:
            properties_dict['material_id'] = i.material_id
            properties_dict['formula'] = i.formula_pretty
            ele_list.append(properties_dict.copy())
        dataset["elements"] = ele_list
        filename = "ele_" + family + "_0k.json"
        with open(Path("data")/family/filename, "w") as f:
            json.dump(dataset, f, indent=2)


def get_solid_compounds_parser(args):
    get_solid_compounds(args.anion, args.cation, stable=args.stable, number=args.number)

def get_ele_parser(args):
    get_ele(args.entry, args.family)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='get the structures for solar thermal ammonia synthesis calculations')
    subparsers = parser.add_subparsers(help='Get compounds or elements')

    # Get compounds data
    compounds_parser = subparsers.add_parser('compounds', help="get solid compounds from MP.")
    compounds_parser.add_argument('anion', action='store', choices=['H', 'N', 'O', 'N-H'], type=str, help='define the anion in the compound')
    compounds_parser.add_argument('cation', action='store', choices=[1, 2, 3, 4], type=int, help='define the number of cations')
    compounds_parser.add_argument('--number', action='store', type=int,
                        help='the total number of atoms in the chemical composition')
    compounds_parser.add_argument('--stable', action='store_true', help='get the most structure with the lowest E above hull')
    compounds_parser.set_defaults(func=get_solid_compounds_parser)

    # Get elements data
    elements_parser = subparsers.add_parser('elements', help="get elements from MP.")
    elements_parser.add_argument('-e', '--entry', action='store_true',
                                 help='get element entries for constructing GibbsComputedStructureEntry')
    elements_parser.add_argument('-f', '--family', choices=["binary", "ternary"],
                                 help="get elements in binary/ternary reactions")
    elements_parser.set_defaults(func=get_ele_parser)

    args = parser.parse_args()
    args.func(args)
