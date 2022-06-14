from pymatgen import MPRester
from pymatgen.io.vasp import Incar
from pymatgen.transformations.standard_transformations import ConventionalCellTransformation

from default_settings import MPRest_key
import os,tarfile,shutil

cwd = os.getcwd()

mpr = MPRester(MPRest_key)
response = mpr.session.get("https://materialsproject.org/materials/10.17188/1476059")
text = response.text
for t in text.split():
    if ('mp-' in t) and ('mp-1111031' in t):
        mp_id=t.split('"')[1].split("/")[-1]
        data=mpr.get_data(mp_id)

        system_name='dpv_'+data[0]['pretty_formula']
        print(system_name,mp_id)

        structure=mpr.get_structure_by_material_id(mp_id)
        wd = cwd + '/' + system_name
        if not os.path.exists(wd):
            os.makedirs(wd)

        os.chdir(wd)
        
        structure.to(fmt='poscar',filename='POSCAR')
        os.chdir(cwd)

        with tarfile.open(system_name+'.tar.gz', mode='w:gz') as archive:
            archive.add('./'+system_name, recursive=True)

        try:
            shutil.rmtree('./'+system_name)
        except:
            pass
        try:
            os.rmtree('./'+system_name)
        except:
            pass
