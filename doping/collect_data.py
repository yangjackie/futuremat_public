import os, glob, zipfile

import ase
from ase.io.vasp import *
from ase.db import connect

from core.dao.vasp import VaspReader
from twodPV.collect_data import populate_db


def get_total_energies(db, dir=None):
    all_zips = glob.glob(dir + "/*.zip")
    for zip in all_zips:
        kvp = {}
        data = {}
        kvp['uid'] = zip.replace(".zip", '').replace('/', '_')
        archive = zipfile.ZipFile(zip)

        atoms = None
        total_energy = None

        has_contcar = False
        for name in archive.namelist():
            if 'OSZICAR' in name:
                oszicar = archive.read(name)
                oszicar_reader = VaspReader(file_content=str(oszicar).split('\\n'))
                total_energy = oszicar_reader.get_free_energies_from_oszicar()[-1]
                kvp['total_energy'] = total_energy

            if 'CONTCAR' in name:
                with open('CONTCAR_temp', 'w') as f:
                    for l in str(archive.read(name)).split('\\n'):
                        f.write(l+'\n')
                has_contcar=True

        if not has_contcar:
            for name in archive.namelist():
                if 'POSCAR' in name:
                    with open('CONTCAR_temp', 'w') as f:
                        for l in str(archive.read(name)).split('\\n'):
                            f.write(l + '\n')

        crystal = ase.io.read('CONTCAR_temp',format='vasp')
        f.close()
        os.remove('CONTCAR_temp')

        if (crystal is not None) and (total_energy is not None):
            print(kvp['uid'], total_energy)
            populate_db(db, crystal, kvp, data)



def pure_total_energies(db):
    get_total_energies(db, dir='pure_O')
    get_total_energies(db, dir='pure')


def CsPbSnCl3_energies(db):
    get_total_energies(db, dir='mixed_CsPbSnCl3')


def CsPbSnBr3_energies(db):
    get_total_energies(db, dir='mixed_CsPbSnBr3')


def CsPbSnI3_energies(db):
    get_total_energies(db, dir='mixed_CsPbSnI3')


def Cs2PbSnCl6_energies(db):
    get_total_energies(db, dir='mixed_Cs2PbSnCl6')


def Cs2PbSnBr6_energies(db):
    get_total_energies(db, dir='mixed_Cs2PbSnBr6')


def Cs2PbSnI6_energies(db):
    get_total_energies(db, dir='mixed_Cs2PbSnI6')

def binaries(db):
    get_total_energies(db, dir='binaries')

def collect(db):
    errors = []
    steps =  [binaries,
              pure_total_energies,
              CsPbSnCl3_energies,
              CsPbSnBr3_energies,
              CsPbSnI3_energies,
              Cs2PbSnCl6_energies,
              Cs2PbSnBr6_energies,
              Cs2PbSnI6_energies]

    for step in steps:
        try:
            step(db)
        except Exception as x:
            print(x)
            error = '{}: {}'.format(x.__class__.__name__, x)
            errors.append(error)
    return errors


if __name__ == "__main__":
    dbname = os.path.join(os.getcwd(), 'doping.db')
    db = connect(dbname)
    print('Established a sqlite3 database object ' + str(db))
    collect(db)
