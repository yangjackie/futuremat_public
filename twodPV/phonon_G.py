import zipfile
import os
import shutil
from twodPV.calculators import *

def execute():
    update_core_info()
    directory = 'phonon_G'
    if not os.path.exists(directory):
       os.makedirs(directory)

    os.chdir(directory)
    try:
        os.remove('./INCAR')
    except:
        pass
    try:
        os.remove('./KPOINTS')
    except:
        pass

    logger = setup_logger(output_filename='phonon.log')
    shutil.copy('../CONTCAR', 'POSCAR')

    default_bulk_optimisation_set.update(
        {'ISPIN': 1,
         'LWAVE': False,
         'NSW': 1,
         'PREC': 'Accurate',
         'EDIFF': 1e-05,
         'IBRION': 8,
         'ISIF': 0,
         'ISYM': 0,
         'LREAL': '.FALSE.',
         'POTIM': 0.01,
         "NELM": 200,
         "LEPSILON": True,
         'MP_points': [4,4,1],
         'Gamma_centered': True,
         'clean_after_success': True,
         'NCORE':28})

    logger.info("==========Gamma point phonon calculation with VASP==========")
    structure = VaspReader(input_location='./POSCAR').read_POSCAR()
    logger.info("Start from supercell defined in POSCAR")
    try:
       vasp = Vasp(**default_bulk_optimisation_set)
       vasp.set_crystal(structure)
       vasp.execute()
       logger.info("VASP terminated properly: " + str(vasp.completed))
    except:
       logger.infor("VASP crashed out, just clean up")
    vasp.tear_down()
    os.chdir('..')
    shutil.make_archive('phonon_G', 'zip', 'phonon_G')

    import subprocess
    subprocess.Popen(['rm', '-rf', './phonon_G'])
    #shutil.rmtree('./phonon_G',ignore_errors=True)

if __name__ == '__main__':
   execute()