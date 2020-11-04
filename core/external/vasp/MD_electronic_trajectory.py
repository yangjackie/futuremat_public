# Module containing codes for performing electronic structure calculations along the molecular dynamic trajectories

import argparse
from core.dao.vasp import VaspReader, VaspWriter
from core.calculators.vasp import Vasp
import os, shutil
from pymatgen.io.vasp.outputs import Vasprun
from os import path
import argparse
from pymatgen.electronic_structure.core import Spin, Orbital
from scipy.interpolate import interp1d
import numpy as np
import glob

# Getting the options in
parser = argparse.ArgumentParser()
parser.add_argument("--traj", help='VASP MD trajectory file', default='XDATCAR')
parser.add_argument("--batch", action='store_true', help='wether to do calculations in batch')
parser.add_argument("--batch_size", help='number of frames to calculate in a batch', type=int, default=100)
parser.add_argument("--part", help='which batch of calculation to run for a long trajectory', type=int)
parser.add_argument("--save", action='store_true', help='whether to store the vasprun.xml files')
# parser.add_argument("--output", type=str, default='gap_dynamics.dat', help='name of the outputfile storing the band positions for each frame')
args = parser.parse_args()

def get_cb(dos,fermi=None,tol=2):
    cb = None
    for pp in range(5000):
        e = fermi+5/5000*pp
        if (abs(dos(e))-abs(dos(fermi)))/abs(dos(fermi)) > tol:
            cb = e
            break
    return cb

def get_vb(dos,fermi=None,tol=2):
    vb = None
    for pp in range(5000):
        e = fermi-5/5000*pp
        if (abs(dos(e))-abs(dos(fermi)))/abs(dos(fermi)) > tol:
            vb = e
            break
    return vb

# default options for doing DOS calculations
incar_dict = {'EDIFF': 1e-05,
              'EDIFFG': -0.01,
              'IALGO': 38,
              'IBRION': -1,
              'ISIF': 0,
              'ISPIN': 1,
              'LVTOT': False,
              'LWAVE': False,
              'LCHARG': False,
              'LREAL': 'Auto',
              'NELM': 150,
              'NSW': 0,
              'NCORE': 48,
              'use_gw': True,
              'Gamma_centered': True,
              'MP_points': [1, 1, 1],
              'executable': 'vasp_gam'}

pwd = os.getcwd()

# get the trajectory
all_frames = VaspReader(args.traj).read_XDATCAR()

if not args.batch:
    output = 'gap_dynamics.dat'
else:
    output = 'gap_dynamics_' + str(args.part) + '.dat'

# figure out if this is a continuing calculation
if not os.path.exists(pwd + '/' + output):
    o = open(pwd + '/' + output, 'w+')
    o.write('Frame\t VBM \t CBM \t E_f \t E_g\n')
    o.close()
    last = 0
else:
    o = open(pwd + '/' + output, 'r')
    for l in o.readlines():
        last = l.split()[0]
    try:
        last = int(last)
    except:
        last = 0
    print('Last frame is '+str(last))
    o.close()

for i, frame in enumerate(all_frames):
    if (not args.batch) and (last > 0) and (i <= last):
        continue
    if (args.batch) and ((i < args.batch_size * args.part + last) or (i >= args.batch_size * (args.part + 1))):
        continue

    try:
        os.mkdir(pwd + '/frame_' + str(i))
    except:
        pass

    os.chdir(pwd + '/frame_' + str(i))

    # run the vasp calculation
    vasp = Vasp(**incar_dict)
    vasp.set_crystal(frame)
    vasp.execute()
    if vasp.completed:

        # analyse the density-of-states to get the final results
        vasprun = Vasprun("vasprun.xml")
        dos = vasprun.tdos.densities[Spin.up]

        xnew = np.linspace(min(vasprun.tdos.energies),
                           max(vasprun.tdos.energies), 500 * len(vasprun.tdos.energies))
        _new_dos = interp1d(vasprun.tdos.energies, dos)

        gap = 0.0
        tol = 0.1
        cb = get_cb(_new_dos, fermi=vasprun.tdos.efermi, tol=tol)
        vb = get_vb(_new_dos, fermi=vasprun.tdos.efermi, tol=tol)
        if (cb is not None) and (vb is not None):
            gap = cb - vb

        os.chdir(pwd)
        o = open(pwd + '/' + output, 'a+')
        o.write(str(i) + '\t' + "{:.4f}".format(vb) + '\t' + "{:.4f}".format(cb) + '\t' + "{:.4f}".format(
            vasprun.tdos.efermi) + '\t' + "{:.4f}".format(gap) + '\n')
        o.close()
    else:
        os.chdir(pwd)

    if not args.save:
        shutil.rmtree(pwd + '/frame_' + str(i))
    else:
        # save only the vasprun.xml file
        os.chdir(pwd + '/frame_' + str(i))
        files = ['CHG', 'CHGCAR', 'EIGENVAL', 'IBZKPT', 'PCDAT', 'POTCAR', 'WAVECAR', 'LOCPOT',
                 'node_info', "WAVECAR", "WAVEDER", 'DOSCAR', 'PROCAR', 'POSCAR', 'KPOINTS', 'INCAR', 'REPORT',
                 'OUTCAR', 'OSZICAR', 'CONTCAR', 'XDATCAR', 'vasp.log']
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass
        os.chdir(pwd)
        shutil.make_archive('frame_' + str(i), 'zip', 'frame_' + str(i))
        import subprocess

        subprocess.Popen(['rm', '-rf', 'frame_' + str(i)])

if args.save:
    if not args.batch:
        try:
            os.mkdir(pwd + '/all_frames')
        except:
            pass

        for z in glob.glob('frame*zip'):
            shutil.move(z, pwd + '/all_frames')

        shutil.make_archive('all_frames', 'zip', 'all_frames')
        import subprocess

        subprocess.Popen(['rm', '-rf', 'all_frames'])
    else:
        try:
            os.mkdir(pwd + '/all_frames_part_' + str(args.part))
        except:
            pass

        for i, frame in enumerate(all_frames):
            if ((i < args.batch_size * args.part) or (i >= args.batch_size * (args.part + 1))):
                continue
            for z in glob.glob('frame_' + str(i) + '.zip'):
                shutil.move(z, pwd + '/all_frames_part_' + str(args.part))
        shutil.make_archive('all_frames_part_' + str(args.part), 'zip', 'all_frames_part_' + str(args.part))
        import subprocess

        subprocess.Popen(['rm', '-rf', 'all_frames_part_' + str(args.part)])
