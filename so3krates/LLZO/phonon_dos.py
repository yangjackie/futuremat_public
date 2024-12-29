import ase.cell

from core.external.vasp.analysis import velocity_autocorrelation_function, get_phonon_dos, vibrational_free_energies
from core.dao.vasp import VaspReader
from core.internal.builders.crystal import map_ase_atoms_to_crystal
import argparse
import warnings

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc

rc('text', usetex=True)
import matplotlib.pylab as pylab

params = {'legend.fontsize': '11',
          'figure.figsize': (7, 4),
          'axes.labelsize': 16,
          'axes.titlesize': 16,
          'xtick.labelsize': 13,
          'ytick.labelsize': 13}
pylab.rcParams.update(params)

import numpy as np
import h5py
import typing
from tqdm import tqdm
from ase import Atoms
from ase.io import read

from scipy.ndimage import gaussian_filter


def prepare_ml_traj(traj_file: str,
                    cell: ase.cell.Cell,
                    save_from_i: typing.Optional[int] = 0,
                    save_every_i: typing.Optional[int] = 1,
                    save_up_to: typing.Optional[int] = None,
                    parallel_run: bool = False) -> list:
    ml_frames = []
    trajectory = h5py.File(traj_file)
    print(trajectory.keys())

    pos = np.array(trajectory['positions'][:])
    if save_up_to is None:
        save_up_to = len(pos)
    pos = pos.reshape(-1, *pos.shape[-2:])[save_from_i:save_up_to][::save_every_i]

    potential_energies = trajectory['potential_energy'][save_from_i:save_up_to:save_every_i]
    try:
        vel = np.array(trajectory['velocities'][:])
        vel = vel.reshape(-1, *vel.shape[-2:])[save_from_i:save_up_to][::save_every_i]
        # shape: (steps%save_every_i, n_atoms, 3)
    except KeyError:
        import logging
        logging.warning('No velocities detected in trajectory. Converting to trajectory  without them.')
        vel = np.zeros_like(pos)

    z = np.array(trajectory['atomic_numbers'][:])
    z = z.reshape(-1, z.shape[-1])[0]  # shape: (n_atoms)

    counter=0
    if parallel_run is not True:
        for p, v in zip(pos, vel):
            this_frame = Atoms(positions=p, numbers=z, velocities=v, cell=cell)
            crystal = map_ase_atoms_to_crystal(this_frame)
            crystal.potential_energy = potential_energies[counter]/crystal.total_num_atoms()
            ml_frames.append(crystal)
            counter+=1
    else:
        raise NotImplementedError
    return ml_frames


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='analysis script for phonon calculations with vasp and/or mlff',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-xdatcar', '--xdatcar', type=str, required=False, default=None,
                        help='xdatcar file containing the vasp MD result')
    parser.add_argument('-mltraj', '--mltraj', type=str, required=False, default=None,
                        help='trajectory files from running ML-MD calculation')
    parser.add_argument('--save_every_i', type=int, required=False, default=1)
    parser.add_argument('--save_from_i', type=int, required=False, default=0)
    parser.add_argument('--save_up_to', type=int, required=False, default=-1)
    parser.add_argument('-plt_dos', '--plt_dos', action='store_true', help='plot phonon DOS for inspection')
    parser.add_argument('-plt_autocorr', '--plt_autocorr', action='store_true',
                        help='plot velocity autocorrelation for inspection')
    parser.add_argument('-temp', '--temp', type=float, required=False,
                        help='temperature for plottingat which MD is run')
    # parser.add_argument('--multiple_start',action='store_true')
    parser.add_argument('--multitemp_ml', action='store_true')
    parser.add_argument('--multitemp_dft', action='store_true')
    parser.add_argument('--supercell', action='store_true')
    args = parser.parse_args()
    try:
        cell = read('./POSCAR').get_cell()  # this is a hack, as the trajectory.h5 does not store the unit cell information
    except:
        pass

    if args.xdatcar is not None:
        all_vasp_frames = VaspReader(input_location=args.xdatcar).read_XDATCAR()

        # vasp_frames=ase.io.read(args.xdatcar, index=':')
        # all_vasp_frames = []
        # for a in tqdm(vasp_frames):
        #   all_vasp_frames.append(map_ase_atoms_to_crystal(a))

        if args.plt_autocorr:
            dft_corr = velocity_autocorrelation_function(frames=all_vasp_frames, potim=1)

        if args.plt_dos:
            omega_vasp, ph_dos_vasp = get_phonon_dos(all_vasp_frames, potim=1, nblock=1, unit='meV')
            ph_dos_vasp = [i / max(ph_dos_vasp) for i in ph_dos_vasp]
            vibrational_free_energies(all_vasp_frames, temp=args.temp, potim=1, nblock=1)

    if args.mltraj is not None:

        # if not args.multiple_start:
        save_from_i = args.save_from_i
        save_every_i = args.save_every_i
        save_up_to = args.save_up_to
        # else:
        #     save_from_i = 0
        #     save_every_i = 1
        #     save_up_to = -1

        all_ml_frames = prepare_ml_traj(args.mltraj, save_from_i=save_from_i, save_every_i=save_every_i,
                                        save_up_to=save_up_to, cell=cell)

        # if not args.multiple_start:
        if args.plt_autocorr:
            ml_corr = velocity_autocorrelation_function(all_ml_frames, potim=1)

        if args.plt_dos:
            omega_ml, ph_dos_ml = get_phonon_dos(all_ml_frames, potim=1, nblock=1, unit='meV')
            ph_dos_ml = [i / max(ph_dos_ml) for i in ph_dos_ml]
            vibrational_free_energies(all_ml_frames, temp=args.temp, potim=1, nblock=1)
            internal_energy = np.mean(np.array([c.potential_energy for c in all_ml_frames]))
            print("Internal energy:", internal_energy, 'eV/atom')

        # else:
        #     if args.plt_dos:
        #         all_set = []
        #         for start in np.random.choice(6000, 20, replace=False):
        #             print("random_start: "+str(start))
        #             omega_ml, ph_dos_ml = get_phonon_dos(all_ml_frames[start:start+4001], potim=1, nblock=1, unit='meV')
        #             ph_dos_ml = [i / max(ph_dos_ml) for i in ph_dos_ml]
        #             all_set.append([omega_ml, ph_dos_ml])

    if args.multitemp_ml: #this is just a quick hack to do some analysis
        temps=[300,500,700,900,1000,1100,1200,1300,1400,1500]
        #colors = plt.cm.YlOrRd(np.linspace(0, 1, len(temps)+6))
        #emps=[300,1500]
        #alphas = np.linspace(0.5, 1, len(temps))
        alphas=[1,1]
        colors=['#FFB55A','#F694BD']

        for i,temp in enumerate(temps):
            try:
                print(temp)
                mltraj = 'trajectory_'+str(temp)+'K.h5'
                all_ml_frames = prepare_ml_traj(mltraj, save_from_i=999, save_every_i=1,save_up_to=6000, cell=cell)
                omega_ml, ph_dos_ml = get_phonon_dos(all_ml_frames, potim=1, nblock=1, unit='meV')
                ph_dos_ml = [i / max(ph_dos_ml) for i in ph_dos_ml]
                ph_dos_ml = gaussian_filter(ph_dos_ml, sigma=2)
                vibrational_free_energies(all_ml_frames, temp=temp, potim=1, nblock=1)
                internal_energy = np.mean(np.array([c.potential_energy for c in all_ml_frames]))
                print("Internal energy: "+str(internal_energy)+' eV/atom')

                plt.plot(omega_ml, ph_dos_ml,color=colors[i],label=str(temp)+' K',alpha=alphas[i])
            except Exception as e:
                print(e)
                pass

        plt.xlim([0, 100])
        plt.ylim([0, 1.0])
        plt.xlabel('Energy (meV)')
        plt.ylabel('Normalised Phonon DOS')
        plt.legend()
        plt.tight_layout()
        plt.savefig('phonon_dos_multitemps_so3krates.pdf')


    if args.multitemp_dft: #this is just a quick hack to do some analysis
        temps=[300,500,700,900,1000,1100,1200,1300,1400,1500]
        colors = plt.cm.YlOrRd(np.linspace(0, 1, len(temps)+6))
        alphas = np.linspace(0.5, 1, len(temps))

        for i,temp in enumerate(temps):
            try:
                potential_energies=[]
                oszicar = open('MD_'+str(temp)+'_NVE/OSZICAR','r')
                for l in oszicar.readlines():
                    if 'T=' in l:
                        potential_energies.append(float(l.split()[4]))
                traj = 'MD_'+str(temp)+'_NVE/XDATCAR'
                all_frames = VaspReader(input_location=traj).read_XDATCAR()
                omega, ph_dos = get_phonon_dos(all_frames, potim=1, nblock=1, unit='meV')
                ph_dos = [i / max(ph_dos) for i in ph_dos]
                ph_dos = gaussian_filter(ph_dos, sigma=2)
                print(temp)
                plt.plot(omega, ph_dos,color=colors[i+4],label=str(temp)+'K',alpha=alphas[i])

                print("Internal energy: "+str(np.mean(np.array(potential_energies))/all_frames[0].total_num_atoms())+' eV/atom')
                vibrational_free_energies(all_frames, temp=temp, potim=1, nblock=1)
            except:
                pass

        plt.xlim([0, 100])
        plt.ylim([0, 1.0])
        plt.xlabel('Energy (meV)')
        plt.ylabel('Normalised Phonon DOS')
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig('phonon_dos_multitemps_dft.pdf')

    if args.supercell:
        #WARNING - this is very slow, takes a long time!
        a=[1,2,3,2,3,2,3,2,2,2,2,2]
        b=[1,1,1,2,2,2,3,2,7,2,3,2]
        c=[1,1,1,1,1,2,1,3,1,4,3,5]

        #a.reverse()
        #b.reverse()
        #c.reverse()

        colors = plt.cm.PuBu(np.linspace(0, 1, len(a)+6))
        alphas = np.linspace(0.6, 1, len(a))
        for i in range(len(a)):
            multiplicity = a[i] * b[i] * c[i]
            directory = './super_'+str(a[i])+'_'+str(b[i])+'_'+str(c[i])
            atoms = read(directory+'/POSCAR_start')
            cell = atoms.get_cell()
            num_atoms = atoms.get_number_of_atoms()
            print("number of atoms: "+str(num_atoms))

            all_ml_frames = prepare_ml_traj(directory+'/trajectory.h5', save_from_i=999, save_every_i=1, save_up_to=-1, cell=cell)
            omega_ml, ph_dos_ml = get_phonon_dos(all_ml_frames, potim=1, nblock=1, unit='meV')
            ph_dos_ml = [i / max(ph_dos_ml) for i in ph_dos_ml]
            ph_dos_ml = gaussian_filter(ph_dos_ml, sigma=2)
            plt.plot(omega_ml, ph_dos_ml, color=colors[i+4], label=str(num_atoms) + ' atoms', alpha=alphas[i])

        plt.xlim([0, 100])
        plt.ylim([0, 1.0])
        plt.xlabel('Energy (meV)')
        plt.ylabel('Normalised Phonon DOS')
        plt.legend()
        plt.tight_layout()
        plt.savefig('phonon_dos_supercell_mlff.pdf')

    if args.plt_dos:
        print("Plotting phonon DOS ...")
        if args.xdatcar is not None:
            ph_dos_vasp = gaussian_filter(ph_dos_vasp, sigma=2)
            plt.plot(omega_vasp, ph_dos_vasp, '--', c='#EA4335', alpha=0.7, label='phonon DOS (DFT)')
        if args.mltraj is not None:
            ph_dos_ml = gaussian_filter(ph_dos_ml, sigma=2)
            plt.plot(omega_ml, ph_dos_ml, '-', c='#1975BA', alpha=1, label='phonon DOS (so3krates)')
        plt.xlim([0, 150])
        plt.ylim([0, 0.9])
        plt.xlabel('Energy (meV)')
        plt.ylabel('Normalised Phonon DOS')
        plt.legend()
        plt.tight_layout()
        plt.savefig('phonon_dos.pdf')

    # if args.plt_dos and args.multiple_start:
    #     for i in range(len(all_set)):
    #         plt.plot(all_set[i][0],all_set[i][1],'k--',alpha=0.3)
    #
    #     mean_dos = np.mean([all_set[i][1] for i in range(len(all_set))], axis=0)
    #     plt.plot(all_set[0][0], mean_dos, 'b-', alpha=0.8)
    #     plt.xlim([0, 150])
    #     plt.ylim([0, 1.05])
    #     plt.xlabel('Energy (meV)')
    #     plt.ylabel('phonon DOS')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig('phonon_av.pdf')

    if args.plt_autocorr:
        print("Plotting phonon autocorrelation function ...")
        if args.xdatcar is not None:
            plt.plot([i for i in range(len(dft_corr))], dft_corr, 'k--', alpha=0.3, label='phonon DOS (DFT)')
        if args.mltraj is not None:
            plt.plot([i for i in range(len(ml_corr))], ml_corr, 'b-', alpha=0.8, label='phonon DOS (MLFF)')
        plt.xlim([3800, 4200])
        plt.legend()
        plt.tight_layout()
        plt.savefig('phonon_autocorr.pdf')

