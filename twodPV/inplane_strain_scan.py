from ase.db import connect
from ase.build import cut, add_vacuum
from ase.io.vasp import *

import numpy as np
import os,tarfile,shutil,argparse,glob,zipfile,sqlite3
from pymatgen.io.vasp.outputs import *


from ase.io.vasp import write_vasp

from core.calculators.vasp import Vasp
from core.internal.builders.crystal import *
from core.dao.vasp import VaspWriter
from ase.spacegroup import crystal as ase_crystal

from twodPV.collect_data import populate_db

from matplotlib import rc, patches
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

rc('text', usetex=True)
params = {'legend.fontsize': '12',
          'figure.figsize': (6, 5),
          'axes.labelsize': 20,
          'axes.titlesize': 16,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)


orientation_dict = {'100': {'a': (1, 0, 0), 'b': (0, 1, 0),
                            'origio': {'AO': (0, 0, 0), 'BO2': (0, 0, 0.25)}},
                    '111': {'a': (1, 1, 0), 'b': (-1, 0, 1),
                            'origio': {'AO3': (0, 0, 0), 'B': (0, 0, 0.25)}},
                    '110': {'a': (1, 1, 0), 'b': (0, 0, 1),
                            'origio': {'O2': (0.05, 0, 0), 'ABO': (0, 0, 0)}}}

def setup_models(a='Sr', b='Ti', c='O', orientation='100', termination='AO', thick=3, db=None):
    for strain in np.arange(-2.0, 3.0, 0.1):
        lattice_constant = 3.9429 * (1.0+strain/100.0)
        atoms = ase_crystal((a, b, c),
                            basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
                            spacegroup=221,
                            cellpar=[lattice_constant, lattice_constant, lattice_constant, 90, 90, 90],
                            size=(1, 1, 1))

        slab = cut(atoms,
                   a=orientation_dict[orientation]['a'],
                   b=orientation_dict[orientation]['b'],
                   nlayers=thick,
                   origo=orientation_dict[orientation]['origio'][termination])
        add_vacuum(slab, 80)

        cwd = os.getcwd()

        wd_name = 'strained_'+str(round(strain,2)).replace('.','_')
        wd = cwd+'/'+wd_name
        if not os.path.exists(wd):
            os.makedirs(wd)

        os.chdir(wd)
        write_vasp('POSCAR', slab, vasp5=True, sort=True)
        os.chdir(cwd)

        with tarfile.open(wd_name+'.tar.gz', mode='w:gz') as archive:
            archive.add('./'+wd_name, recursive=True)

        try:
            shutil.rmtree('./'+wd_name)
        except:
            pass
        try:
            os.rmtree('./'+wd_name)
        except:
            pass

def collect_data(db):
    all_files = glob.glob("*.tar.gz")
    cwd = os.getcwd()

    for f in all_files:
        system_name=f.replace('.tar.gz','')
        strain_level=float(system_name.split('_')[-2]+'.'+system_name.split('_')[-1])

        tf = tarfile.open(f)
        tf.extractall()

        data={}
        kvp={}
        kvp['uid'] = system_name
        kvp['strain'] = strain_level

        os.chdir(system_name)

        try:
            calculator = Vasp()
            calculator.check_convergence()

            if calculator.completed:
                atoms = [a for a in read_vasp_xml(index=-1)][-1]
                kvp['total_energy'] = atoms.get_calculator().get_potential_energy()
                print(system_name + ' total energy: ' + str(kvp['total_energy']) + ' eV/atom')
            else:
                print(system_name + ' NOT CONVERGED!')
        except Exception as e:
            continue

        try:
            with zipfile.ZipFile('./phonon_G.zip') as z:
                with open("./OUTCAR_ph", 'wb') as f:
                    f.write(z.read("OUTCAR"))
            f.close()
            z.close()
        except:
            pass

        if os.path.isfile("./OUTCAR_ph"):
            outcar = Outcar('./OUTCAR_ph')
            outcar.read_lepsilon_ionic()
            print('dielectric ionic tensor')
            print(outcar.dielectric_ionic_tensor)
            data['dielectric_ionic_tensor'] = outcar.dielectric_ionic_tensor
            #print('Born Effective Charge')
            #print(outcar.born)
            populate_db(db, atoms, kvp, data)
            os.remove('./OUTCAR_ph')
        os.chdir(cwd)

        try:
            shutil.rmtree(system_name)
        except:
            pass
        try:
            os.rmtree(system_name)
        except:
            pass

def strain_dielectric_constant_plot(db,all_uids):
    data_dict={}
    for uid in all_uids:
        row = None
        try:
            row = db.get(selection=[('uid', '=', uid)])
        except:
            pass

        #Getting the structural information for the asymptotic correction of 2D dielectric constants
        structure_2d = row.toatoms()
        supercell_c = structure_2d.get_cell_lengths_and_angles()[2]
        structure = map_ase_atoms_to_crystal(structure_2d)
        all_z_positions = np.array(
            [a.scaled_position.z for a in structure.asymmetric_unit[0].atoms])
        all_z_positions = all_z_positions - np.round(all_z_positions)
        all_z_positions = [z * supercell_c for z in all_z_positions]
        slab_thick = max(all_z_positions) - min(all_z_positions)

        #getting the information for dielectric tensors
        try:
            twod_dielectric_tensor = row.data['dielectric_ionic_tensor']
        except:
            continue

        strain = row.key_value_pairs['strain']

        #correct the dielectric tensor to 2D limit
        twod_epsilon_inplane = None
        twod_epsilon_outplane = None
        from analysis.result_analysis import non_zero,trace_inplane,trace_outplane
        if non_zero(twod_dielectric_tensor):
            twod_epsilon_inplane = trace_inplane(twod_dielectric_tensor)
            #twod_epsilon_inplane = (supercell_c / slab_thick) * (
            #        twod_epsilon_inplane - 1.0) + 1.0

            twod_epsilon_outplane = trace_outplane(twod_dielectric_tensor)
            #wod_epsilon_outplane = 1.0 / ((supercell_c / slab_thick) * (
            #        1.0 / twod_epsilon_outplane - 1.0) + 1.0)

            data_dict[strain]={'inplane':twod_epsilon_inplane,'outplane':twod_epsilon_outplane}

    x=list(sorted(data_dict.keys()))

    inplane=[data_dict[_x]['inplane'] for _x in x]
    outplane=[data_dict[_x]['outplane'] for _x in x]

    plt.plot(x,inplane,'s-',c='#CB0000',label='in-plane',ms=4,markerfacecolor='none')
    #plt.plot(x,outplane,'s-',c='#102A49',label='out-of-plane',ms=4,markerfacecolor='none')

    plt.xlabel("In-Plane Strain $\Delta \epsilon$ (\%)")
    plt.ylabel("Static Ionic Dielectric Constants")
    #plt.legend()
    plt.tight_layout()
    plt.savefig("strain_dielectric.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scan of 2D properties with in-plane strain deformations',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-prep","--prepare",    action="store_true",help="build the deformed structures and prepare the relevant calculation folder")
    parser.add_argument("-collect","--collect", action="store_true",help='Collect the calculation results')
    parser.add_argument("-plot","--plot",action='store_true', help="Make plots for data analysis")

    parser.add_argument("-a",   "--a", type=str, default='Sr', help="A site cation")
    parser.add_argument("-b",   "--b", type=str, default='Ti', help="B site cation")
    parser.add_argument("-c",   "--c", type=str, default='O', help="C site anion")
    parser.add_argument("--db", type=str, default='strain.db',help="Name of the database that contains the results of the screenings.")
    parser.add_argument("--orient", type=str, default='100',help='Orientations of the two-d perovskite slabs')
    parser.add_argument("--terminations", type=str, default='AO',help='Surface termination type of the two-d slab')
    parser.add_argument("--thick", type=int, default=3, help='thickness of the 2D structures')

    args = parser.parse_args()
    db=connect(args.db)
    if args.prepare:
        setup_models(a=args.a,b=args.b,c=args.c,orientation=args.orient,thick=args.thick,termination=args.terminations,db=db)
    if args.collect:
        collect_data(db)
    if args.plot:
        all_uids = []
        _db = sqlite3.connect(args.db)
        cur = _db.cursor()
        cur.execute("SELECT * FROM systems")
        rows = cur.fetchall()

        for row in rows:
            for i in row:
                if 'uid' in str(i):
                    this_dict = json.loads(str(i))
                    this_uid = this_dict['uid']
                    all_uids.append(this_uid)
        strain_dielectric_constant_plot(db,all_uids)