from pymatgen import MPRester
from pymatgen import Structure
from pymatgen.transformations.standard_transformations import ConventionalCellTransformation


from default_settings import MPRest_key
import os,tarfile,shutil

cwd = os.getcwd()

mpr = MPRester(MPRest_key)

"""
response = mpr.session.get("https://materialsproject.org/materials/10.17188/1476059")
text = response.text
all_elements=[]

counter=0

for t in text.split():
    if ('mp-' in t):
        try:
            mp_id = t.split('"')[1].split("/")[-1]
            structure = mpr.get_structure_by_material_id(mp_id)
            elements = list(set([s.symbol for s in structure.species]))
            all_elements += elements
            all_elements = list(set(all_elements))
            all_elements = list(sorted(all_elements))
            counter+=1
            print(str(counter)+'\t'+str(all_elements))
        except:
            pass
"""
all_elements=['Ag', 'Al', 'As', 'Au', 'Bi', 'Br', 'Ce', 'Cl', 'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Er', 'Eu', 'F', 'Fe', 'Ga', 'Gd', 'Hg', 'I', 'In', 'Ir', 'K', 'La', 'Li', 'Lu', 'Mn', 'Mo', 'Na', 'Nb', 'Nd', 'Ni', 'Pd', 'Pr', 'Rb', 'Rh', 'Ru', 'Sb', 'Sc', 'Sm', 'Ta', 'Tb', 'Ti', 'Tl', 'Tm', 'V', 'Y']

all_elements=['V']

for element in all_elements:
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
    structure = mpr.get_structure_by_material_id(lowest_energy_id)
    transformer = ConventionalCellTransformation()
    structure = transformer.apply_transformation(structure)

    #structure = Structure.from_str(lowest_cif, fmt="cif")
    poscar = structure.to(fmt="poscar")
    f = open('POSCAR', 'w')
    for l in poscar:
        f.write(l)
    f.close()

    ## KPOINTS
    kpoints = lowest_k
    kpoints.write_file('KPOINTS')

    os.chdir('..')
    with tarfile.open(str(element) + '.tar.gz', mode='w:gz') as archive:
        archive.add('./' + str(element), recursive=True)

    try:
        shutil.rmtree('./' + str(element))
    except:
        pass
    try:
        os.rmtree('./' + str(element))
    except:
        pass

    os.chdir(cwd)
