import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib import rc
import matplotlib.pylab as pylab
from pymatgen.io.vasp.outputs import Vasprun, Procar, BSVasprun
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.electronic_structure.core import Spin, Orbital
from pymatgen.electronic_structure.plotter import BSPlotter
import numpy as np
import math
import scipy
from scipy.interpolate import interp1d
from scipy.fftpack import fft, ifft, fftfreq, fftshift
import glob

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
params = {'legend.fontsize': '10',
          'figure.figsize': (6, 5),
         'axes.labelsize': 20,
         'axes.titlesize':20,
         'xtick.labelsize':12,
         'ytick.labelsize':12}
#         'axes.prop_cycle':cycler.cycler('color',color)}
pylab.rcParams.update(params)

k_B_ev = 8.617e-5
hbar_ev = 4.136e-15
N_A = 6.0221409e+23


def get_projected_plot_dots_local(bs, dictio, ylim=[-5, 5], color_codes=None, zero_to_efermi=True, spin=Spin.up,
                                  filename=None):
    from plotter import BSPlotterProjected as LocalBSPlotterProjected
    from pymatgen.electronic_structure.core import Spin, Orbital, OrbitalType
    from pymatgen.util.plotting import pretty_plot
    import math
    from matplotlib.patches import Patch

    local_plotter = LocalBSPlotterProjected(bs)
    band_linewidth = 1

    proj = local_plotter._get_projections_by_branches(dictio)
    data = local_plotter.bs_plot_data(zero_to_efermi)

    plt = pretty_plot(10)
    plt.subplot(100 * math.ceil(0.5) + 20)
    local_plotter._maketicks(plt)
    e_min = -4
    e_max = 4
    if local_plotter._bs.is_metal():
        e_min = -10
        e_max = 10

    for b in range(len(data['distances'])):
        for i in range(local_plotter._nb_bands):
            plt.plot(data['distances'][b],
                     [data['energy'][b][str(spin)][i][j]
                      for j in range(len(data['distances'][b]))],
                     'b:',
                     linewidth=band_linewidth, alpha=0.6)

    # fix this, this is default for STO-SnTe project!
    # color_codes = {"Ti":'#AEBD38','Sn':'#8D230F','Te':'#E6D72A'}
    legend_elements = []

    counter = -1

    for el in dictio.keys():
        for o in dictio[el]:
            print(el, o)
            counter += 1
            legend_elements.append(
                Patch(facecolor=color_codes[el], edgecolor=color_codes[el], label=el + '-$' + o + "$"))
            for b in range(len(data['distances'])):
                for i in range(local_plotter._nb_bands):
                    for j in range(len(data['energy'][b][str(spin)][i])):
                        plt.plot(data['distances'][b][j],
                                 data['energy'][b][str(spin)][i][j],
                                 'o',
                                 markeredgecolor=color_codes[el], markerfacecolor=color_codes[el],
                                 markersize=proj[b][str(spin)][i][j][str(el)][o] * 10, alpha=0.5)

    if ylim is None:
        if local_plotter._bs.is_metal():
            if zero_to_efermi:
                plt.ylim(e_min, e_max)
            else:
                plt.ylim(local_plotter._bs.efermi + e_min, local_plotter._bs.efermi + e_max)
    else:
        plt.ylim(ylim)
    plt.ylabel('$E-E_{f}$ (eV)', size=18)
    plt.legend(handles=legend_elements, loc=1)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def plot_orbital_projected_band_structure(filename='vasprun.xml', ylim=[-5, 5], orbitals=None, color_codes=None,
                                          spin=Spin.up, save_filename=None):
    if orbitals is None:
        plot_simple_smoothed_band_structure(filename=filename, ylim=ylim)
    else:
        v = BSVasprun(filename, parse_projected_eigen=True)
        print('read in vasprun xml file')
        bs = v.get_band_structure(line_mode=True)
        get_projected_plot_dots_local(bs, orbitals, color_codes=color_codes, ylim=ylim, spin=spin,
                                      filename=save_filename)


def plot_simple_smoothed_band_structure(ylim=[-1.5, 3.5], filename=None):
    vasprun = Vasprun('./vasprun.xml')
    bs = vasprun.get_band_structure(line_mode=True)
    if filename is None:
        # BSPlotter(bs).get_plot(smooth=True,ylim=ylim)
        BSPlotter(bs).show(smooth=True, ylim=ylim)
    else:
        BSPlotter(bs).save_plot(filename)

def plot_total_density_of_states(xlim=None, ylim=None, filename=None):
    vasprun = Vasprun('./vasprun.xml')
    if not vasprun.is_spin:
        dos = vasprun.tdos.densities[Spin.up]
    else:
        dos = vasprun.tdos.densities[Spin.up] + vasprun.tdos.densities[Spin.down]

    xnew = np.linspace(min(vasprun.tdos.energies - vasprun.tdos.efermi),
                       max(vasprun.tdos.energies - vasprun.tdos.efermi), 100 * len(vasprun.tdos.energies))
    power_smooth = interp1d(vasprun.tdos.energies - vasprun.tdos.efermi, dos)

    #from scipy import signal
    #dydx = signal.savgol_filter(power_smooth(xnew), window_length=11, polyorder=2, deriv=1)

    plt.plot(xnew, power_smooth(xnew), 'b')
    #plt.plot(xnew, dydx, 'r')

    #plt.plot(vasprun.tdos.energies - vasprun.tdos.efermi, dos, 'b')

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)
    #xelse:
        #plt.ylim([0,max(dos)+0.05*max(dos)])

    plt.ylabel("Density-of-states (states/eV/cell)", fontsize=12)
    plt.xlabel("$E-E_{F}$ (eV)", fontsize=12)
    plt.tight_layout()

    plt.xlim([-5,5])

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

def get_dos_gap():
    vasprun = Vasprun("vasprun.xml")
    dos = vasprun.tdos.densities[Spin.up]
    xnew = np.linspace(min(vasprun.tdos.energies),
                       max(vasprun.tdos.energies), 500 * len(vasprun.tdos.energies))
    _new_dos = interp1d(vasprun.tdos.energies, dos)
    from scipy import signal
    dydx = signal.savgol_filter(_new_dos(xnew), window_length=11, polyorder=2, deriv=1)

    vbm=vasprun.tdos.efermi
    cbm=vasprun.tdos.efermi

    for i,x in enumerate(xnew):
        this_dos = _new_dos(x)
        this_diff = dydx[i]
        if x<vasprun.tdos.efermi: continue
        if (abs(this_diff)<0.01):
            vbm = x
            break

    for i,x in enumerate(xnew):
        this_dos = _new_dos(x)
        this_diff = dydx[i]
        if x<=vbm: continue
        if (abs(this_diff)<0.01):
            cbm = x
        else:
            break

    print('vbm is ', vbm)
    print('cbm is ', cbm)
    gap = cbm-vbm
    return gap

def get_cb(dos,fermi=None,tol=0.2):
    cb = None
    _dos = dos(fermi)
    for pp in range(5000):
        e = fermi+5/5000*pp
        if dos(e)>_dos:
            cb = e
            break
        _dos=dos(e)
        #if (dos(e)-dos(fermi))/dos(fermi) > tol:
        #    cb = e
        #    break
    return cb

def get_vb(dos,fermi=None,tol=0.2):
    vb = None
    _dos = dos(fermi)
    for pp in range(5000):
        e = fermi-5/5000*pp
        if dos(e)>_dos:
            vb = e
            break
        _dos=dos(e)
        #if (dos(e)-dos(fermi))/dos(fermi) > tol:
        #    vb = e
        #    break
    return vb


def plot_MD_energies_and_temperature_evolution(time_step=0.001):
    temp = []
    energies = []

    data = open('OSZICAR', 'r')
    for line in data.readlines():
        if 'T=' in line:
            temp.append(float(line.split()[2]))
            energies.append(float(line.split()[4]))
    plt.subplot(2, 1, 1)
    plt.plot([x * time_step for x in range(len(temp))], temp, 'b-')
    plt.xlabel('Time (ps)')
    plt.ylabel('Temperature (K)')

    plt.subplot(2, 1, 2)
    plt.plot([x * time_step for x in range(len(energies))], energies, 'r-')
    plt.xlabel('Time (ps)')
    plt.ylabel('Energy (eV)')
    plt.tight_layout()
    plt.show()


def pair_correlation_function_averaged(frames, bins=50, A='', B='', cross_term=False):
    if not A:
        A = frames[0].asymmetric_unit[0].atoms[0].label
    if not B:
        B = A

    pos_A = []
    pos_B = []
    index_A = []
    index_B = []

    for i in range(len(frames)):
        _pos_A = []
        _pos_B = []
        for p, atom in enumerate(frames[i].asymmetric_unit[0].atoms):
            if atom.label == A:
                _pos_A.append(np.array([atom.scaled_position.x, atom.scaled_position.y, atom.scaled_position.z]))
                if i == 0:
                    index_A.append(p)
            if atom.label == B:
                _pos_B.append(np.array([atom.scaled_position.x, atom.scaled_position.y, atom.scaled_position.z]))
                if i == 0:
                    index_B.append(p)
        # print(len(_pos_A),len(_pos_B))
        pos_A.append(_pos_A)
        pos_B.append(_pos_B)

    if (B == A) or cross_term:
        rABs = [pos_A[i][k] - pos_B[i][j] for i in range(len(frames)) for k in range(len(pos_A[i])) for j in
                range(len(pos_B[i])) if
                index_A[k] != index_B[j]]

    if (B != A) and (not cross_term):
        pos = [pos_A[i] + pos_B[i] for i in range(len(frames))]

        index = index_A + index_B

        rABs = [pos[i][k] - pos[i][j] for i in range(len(frames)) for k in range(len(pos[i])) for j in
                range(len(pos[i])) if
                index[k] != index[j]]

    rABs = np.array(rABs)

    for i in range(len(rABs)):
        for j in range(3):
            if rABs[i][j] > 0.5:
                rABs[i][j] = rABs[i][j] - 1.0
            if rABs[i][j] < -0.5:
                rABs[i][j] = rABs[i][j] + 1.0

    _lv = frames[0].lattice.lattice_vectors
    lv = np.array(
        [[_lv[0][0], _lv[0][1], _lv[0][2]], [_lv[1][0], _lv[1][1], _lv[1][2]], [_lv[2][0], _lv[2][1], _lv[2][2]]])

    rABs = np.linalg.norm(np.dot(lv, rABs.T), axis=0)
    val, b = np.histogram(rABs, bins=bins)

    # print("total of histogram "+str(sum(val)/len(frames)))

    rho = len(frames[0].asymmetric_unit[0].atoms) / frames[0].lattice.volume
    Na = len(index_A)
    Nb = len(index_B)

    if (B != A) and (not cross_term):
        Na = len(index)
        Nb = len(index)

    dr = b[1] - b[0]
    val = val * len(frames[0].asymmetric_unit[0].atoms) / (4 * 3.1415 * b[1:] ** 2 * dr) / (Na * Nb * rho) / len(frames)
    # val = val * len(frames[0].asymmetric_unit[0].atoms) / (Na * Nb) / len(frames)
    return val, b[1:]


def pair_correlation_function_single_frame(crystal, bins=50, A='', B='', cross_term=False):
    return pair_correlation_function_averaged([crystal], bins=bins, A=A, B=B, cross_term=cross_term)


def velocity_autocorrelation_function(frames, potim=8):
    def get_coord(frame_index, atom_index):
        position = frames[frame_index].asymmetric_unit[0].atoms[atom_index].scaled_position
        return np.array([position.x, position.y, position.z])

    num_atoms_in_cell = len(frames[0].asymmetric_unit[0].atoms)
    pos = np.array([[get_coord(i, j) for j in range(num_atoms_in_cell)] for i in range(len(frames))])
    pos = pos.ravel().reshape((-1, num_atoms_in_cell, 3))
    Niter = len(frames)
    dpos = np.diff(pos, axis=0)
    positionC = np.zeros_like(pos)

    dpos[dpos > 0.5] -= 1.0
    dpos[dpos < -0.5] += 1.0

    # Velocity in Angstrom per femtosecond

    _lv = frames[0].lattice.lattice_vectors
    lv = np.array(
        [[_lv[0][0], _lv[0][1], _lv[0][2]], [_lv[1][0], _lv[1][1], _lv[1][2]], [_lv[2][0], _lv[2][1], _lv[2][2]]])

    for i in range(Niter - 1):
        positionC[i, :, :] = np.dot(pos[i, :, :], lv)
        dpos[i, :, :] = np.dot(dpos[i, :, :], lv) / potim

    positionC[-1, :, :] = np.dot(pos[-1, :, :], lv)
    velocity = dpos

    VAF2 = np.zeros((Niter - 1) * 2 - 1)
    for i in range(num_atoms_in_cell):
        for j in range(3):
            VAF2 += np.correlate(velocity[:, i, j], velocity[:, i, j], 'full')
    # two-sided VAF
    VAF2 /= np.sum(velocity ** 2)
    # VAF = VAF2[Niter - 2:]
    return VAF2




def atomic_displacement_from_ref_frame(frames, ref_frame, ref_atomic_label, direction='x'):
    if direction == 'x':
        direction = 0
    if direction == 'y':
        direction = 1
    if direction == 'z':
        direction = 2

    if isinstance(ref_atomic_label, str):
        ref_atomic_label = [ref_atomic_label]

    _lv = ref_frame.lattice.lattice_vectors
    lv = np.array(
        [[_lv[0][0], _lv[0][1], _lv[0][2]], [_lv[1][0], _lv[1][1], _lv[1][2]], [_lv[2][0], _lv[2][1], _lv[2][2]]])

    ref_coords = np.array(
        [[a.scaled_position.x, a.scaled_position.y, a.scaled_position.z] for a in ref_frame.asymmetric_unit[0].atoms
         if a.label in ref_atomic_label])

    all_coords = np.array([np.array(
        [[a.scaled_position.x, a.scaled_position.y, a.scaled_position.z] for a in frames[i].asymmetric_unit[0].atoms
         if a.label in ref_atomic_label]) for i in range(len(frames))])

    _all_diff = np.array([all_coords[i, :] - ref_coords[:] for i in range(all_coords.shape[0])])
    # impose periodic boundary conditions, no atomic displacement in fractional coordinates should be greater than
    # the length of each unit cell dimension
    _all_diff[_all_diff < -0.5] += 1
    _all_diff[_all_diff > 0.5] -= 1

    for i in range(_all_diff.shape[0]):
        _all_diff[i, :, :] = np.dot(_all_diff[i, :, :], lv)
    all_diff = _all_diff[:, :, direction]

    return all_diff

def plot_atomic_displacement_trajectories(frames, ref_frame, ref_atomic_label, direction='x', potim=8, nblock=10,
                                          filename=None):
    all_diff = atomic_displacement_from_ref_frame(frames, ref_frame, ref_atomic_label, direction=direction)
    num_atoms = all_diff.shape[1]
    time_step = potim * nblock / 1000

    for i in range(num_atoms):
        if i != num_atoms - 1:
            plt.plot([j * time_step for j in range(len(all_diff[:, i]))], all_diff[:, i], '-', c='#90AFC5', alpha=0.3)
        else:
            plt.plot([j * time_step for j in range(len(all_diff[:, i]))], all_diff[:, i], '-', c='#90AFC5', alpha=0.3,
                     label="$\\Delta " + str(direction) + "$ for individual atom")
    plt.plot([j * time_step for j in range(len(all_diff[:, 0]))], np.mean(all_diff, axis=1), '-', c='r',
             label="$\\Delta " + str(direction) + "$ averaged across all atoms")

    plt.ylabel("$\\Delta " + str(direction) + "$ $(\\mbox{\AA})$")
    plt.xlabel('Time (ps)')

    label=''
    for i in ref_atomic_label:
        label+=str(i)+' '

    #plt.annotate(label, xy=(0.85,0.85), xycoords='axes fraction',fontsize=16)
    plt.ylim([-2,2])
    plt.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

def plot_atomic_displacement_statistics(frames, ref_frame, ref_atomic_label, direction='x', potim=8, nblock=10,
                                          filename=None):
    all_diff = atomic_displacement_from_ref_frame(frames, ref_frame, ref_atomic_label, direction=direction)
    val, b = np.histogram(all_diff[abs(all_diff)<0.2], bins=int(0.4/0.01), density=False)
    val = val / max(val)
    def gaus(x, a, x0, sigma):
        return a * scipy.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    from scipy.optimize import curve_fit

    mean = sum(b[1:] * val) / len(val)
    sigma = sum(val * (b[1:] - mean) ** 2) / len(val)

    popt=None
    try:
        popt, pcov = curve_fit(gaus, b[1:], val, p0=[1, mean, sigma])
    except:
        print("Gaussian fit to the histogram failed!!")
        pass

    val, b = np.histogram(all_diff, bins=int((all_diff.max()-all_diff.min())/0.01), density=False)

    if popt is not None:
        plt.plot(gaus(b[1:], *popt), b[1:], 'r-', label="$\\mathcal{G}(\\Delta " + str(direction) + ")$")

    val = val / max(val)
    plt.plot(val/max(val),b[1:],'b-',label="$\\mathcal{P}(\\Delta " + str(direction) + ")$",alpha=0.6)

    plt.xlabel('Probability Density')
    plt.ylabel("$\\Delta " + str(direction) + "$ $(\\mbox{\AA})$")

    label = ''
    for i in ref_atomic_label:
        label += str(i) + ' '

    plt.annotate(label, xy=(0.85,0.15), xycoords='axes fraction',fontsize=16)
    plt.legend()
    plt.ylim([-2,2])
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

def get_phonon_dos(frames, potim=8, nblock=10, unit='THz'):
    potim = potim * nblock
    N = len(frames) - 1
    # Frequency in THz
    omega = fftfreq(2 * N - 1, potim * 1e-15) * 1E-12
    # Frequency in cm^-1
    if unit.lower() == 'cm-1':
        omega *= 33.35640951981521
    if unit.lower() == 'mev':
        omega *= 4.13567
    VAF2 = velocity_autocorrelation_function(frames, potim=potim)
    pdos = np.abs(fft(VAF2 - np.average(VAF2))) ** 2

    return omega[:N], pdos[:N]


def vibrational_free_energies(frames, temp=300, potim=8, nblock=10):
    num_atoms_in_cell = len(frames[0].asymmetric_unit[0].atoms)
    _omega, _pdos = get_phonon_dos(frames, potim=potim, nblock=nblock, unit='meV')
    omega = []
    pdos = []
    for i in range(len(_omega)):
        if _omega[i] != 0.0:
            omega.append(_omega[i] * 1e-3)
            pdos.append(_pdos[i])

    integrand = [math.log(2 * math.sinh(hbar_ev * omega[i] / (2 * k_B_ev * temp))) * pdos[i] for i in range(len(omega))]
    fe = np.trapz(integrand, omega)
    fe = 3 * num_atoms_in_cell * k_B_ev * temp * fe
    return fe


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings('ignore')
    import argparse

    parser = argparse.ArgumentParser(description='analysis script for VASP electronic structure calculations',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--simple_band", action='store_true',
                        help='Plot a simple band structure from vasp static calculation, without projecting onto atomic orbitals')
    parser.add_argument("--total_dos", action='store_true',
                        help='Plot total density of states from vasp static calculation')
    parser.add_argument("--Emin", type=float, default=-1.5,
                        help='lowest energy limit in returned plot, unit: eV')
    parser.add_argument("--Emax", type=float, default=3.5,
                        help='highest energy limit in returned plot, unit: eV')
    parser.add_argument("--output", type=str, default=None,
                        help='output filename for diagrams.')
    parser.add_argument("--md", action='store_true',
                        help='plot energy and temperature change in MD simulation from OSZICAR')
    parser.add_argument("--md_frames", nargs='+', type=str, default='XDATCAR',
                        help='name of the VASP md trajectory file')
    parser.add_argument("--ref_frame", type=str, default='POSCAR_0',
                        help="name of the file containing the reference frame for analyzing atomic displacements")
    parser.add_argument("--ref_atoms", nargs='+', type=str,
                        help='name of the atoms for which atomic displacements will be analyzed')
    parser.add_argument("--direction", type=str, default='x',
                        help='direction of atomic displacements to be analyzed')
    parser.add_argument("-ts", "--time_step", type=float, default=0.001)
    parser.add_argument("--md_potim", type=float, default=8,
                        help='POTIM setting for MD time step in VASP, default: 8fs')
    parser.add_argument("--md_nblock", type=float, default=10,
                        help='NBLOCK setting for MD in VASP, frequency for writing out trajectory frames, default: 10')

    parser.add_argument("--displacement_traj", action='store_true',
                        help='plot trajectory of atom displacement to a given reference frame')
    parser.add_argument("--displacement_histograms", action='store_true',
                        help='plot histograms of atom displacement to a given reference frame')

    parser.add_argument("--dos_gap", action='store_true',
                        help='Extract the density of state gap')

    args = parser.parse_args()

    if args.simple_band:
        plot_simple_smoothed_band_structure(ylim=[args.Emin, args.Emax], filename=args.output)
    if args.total_dos:
        plot_total_density_of_states(filename=args.output)
    if args.md:
        plot_MD_energies_and_temperature_evolution(time_step=args.time_step)
    if args.displacement_traj or args.displacement_histograms:
        from entdecker.core.io.vasp import VaspReader

        print(args.ref_atoms)
        args.md_frames=['XDATCAR_'+str(i+1) for i in range(len(args.md_frames))]
        print(args.md_frames)
        all_frames=[]
        for f in args.md_frames:
            if 'equ' not in f:
                all_frames += VaspReader(input_location=f).read_XDATCAR()
        ref_frame = VaspReader(input_location=args.ref_frame).read_POSCAR()

        if args.displacement_traj:
            plot_atomic_displacement_trajectories(frames=all_frames, ref_frame=ref_frame,
                                                  ref_atomic_label=args.ref_atoms,
                                                  direction=args.direction, potim=args.md_potim, nblock=args.md_nblock,
                                                  filename=args.output)
        if args.displacement_histograms:
            plot_atomic_displacement_statistics(frames=all_frames, ref_frame=ref_frame,
                                                  ref_atomic_label=args.ref_atoms,
                                                  direction=args.direction, potim=args.md_potim, nblock=args.md_nblock,
                                                  filename=args.output)

    if args.dos_gap:
        gap = get_dos_gap()
        print('The DOS gap is '+str(gap)+" eV.")