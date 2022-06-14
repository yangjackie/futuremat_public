from dscribe.descriptors import SOAP, EwaldSumMatrix
from dscribe.kernels import REMatchKernel, AverageKernel
from sklearn.preprocessing import normalize
import numpy as np
from ase.db import connect
import argparse, os, pickle, sqlite3, json
from sklearn.decomposition import KernelPCA

import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
import matplotlib.pylab as pylab

params = {'legend.fontsize': '15',
          'figure.figsize': (7, 6),
          'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)

from core.internal.builders.crystal import map_ase_atoms_to_crystal
from core.models.element import atomic_numbers
from perovskite_screenings.analysis import halide_A, halide_B, halide_C, chalco_A, chalco_B, chalco_C


def system_loader(X):
    all_systems = []
    if X == 'halides':
        A = halide_A
        B = halide_B
        C = halide_C
    elif X == 'chalcogenides':
        A = chalco_A
        B = chalco_B
        C = chalco_C

    for c in C:
        db = connect('perovskites_updated_' + c + '.db')
        for b_count, b in enumerate(B):
            for a_count, a in enumerate(A):
                system_name = a + b + c
                uid = system_name + '_Pm3m'
                row = None
                pm3m_formation_e = None

                try:
                    row = db.get(selection=[('uid', '=', uid)])
                except:
                    continue

                if row is not None:
                    try:
                        pm3m_formation_e = row.key_value_pairs['formation_energy']
                    except KeyError:
                        continue

                if pm3m_formation_e is not None:
                    atoms = row.toatoms()
                    indicies = [atom.index for atom in atoms if atom.symbol == a]
                    indicies += [atom.index for atom in atoms if atom.symbol == b]
                    indicies += [atom.index for atom in atoms if atom.symbol == c]
                    print("adding system: " + str(system_name))
                    all_systems.append({'name': system_name, 'atoms': atoms, 'indicies': indicies})
    return all_systems


def double_perovskites_system_loader():
    all_uids = []
    all_systems = []
    dbname = "double_halide_pv.db"
    _db = sqlite3.connect(dbname)
    cur = _db.cursor()
    cur.execute("SELECT * FROM systems")
    rows = cur.fetchall()

    for row in rows:
        for i in row:
            if 'uid' in str(i):
                this_dict = json.loads(str(i))
                this_uid = this_dict['uid']
                if 'dpv' in this_uid:
                    all_uids.append(this_uid)

    db = connect(dbname)

    for uid in all_uids:
        row = None
        formation_energy = None
        sigma = None
        sigma_mode_averaged = None
        try:
            row = db.get(selection=[('uid', '=', uid)])
        except:
            continue
        if row is not None:
            atoms = row.toatoms()

            try:
                formation_energy = row.key_value_pairs['formation_energy']
                print('system ' + uid + ' Formation Energy ' + str(formation_energy) + ' eV/atom')
            except KeyError:
                continue

            try:
                sigma = row.key_value_pairs['sigma_300K_single']
            except KeyError:
                continue

            try:
                sigma_mode_averaged = row.key_value_pairs['sigma_mode_averaged_300K']
            except KeyError:
                continue

            if formation_energy is not None:
                all_systems.append({'name': uid, 'atoms': atoms,'sigma':sigma,'sigma_mode_averaged':sigma_mode_averaged})
    return all_systems


def build_kernels_for_perovskite_systems(X='halides', kernel='REMatch', save_kernel=True, descriptor='SOAP'):
    # all_systems=system_loader(X)
    all_systems = double_perovskites_system_loader()
    all_features = []

    # Calculate all the SOAP descriptors
    for counter, system in enumerate(all_systems):

        print('Creating descriptor for system :' + str(counter + 1) + '/' + str(len(all_systems)))
        _system = map_ase_atoms_to_crystal(system['atoms'])

        if descriptor.lower() == 'soap':
            _atomic_numbers = list(set([atomic_numbers[a.label] for a in _system.all_atoms(sort=True, unique=True)]))
            desc = SOAP(species=_atomic_numbers,
                        rcut=4, nmax=6, lmax=6, sigma=0.1, periodic=True, crossover=False,
                        sparse=False)
            feature = desc.create(system['atoms'], positions=system['indicies'])
            feature = normalize(feature)
        elif descriptor.lower() == 'ewald':
            desc = EwaldSumMatrix(n_atoms_max=20, permutation='none', flatten=True)
            feature = desc.create(system['atoms'])

        all_features.append(feature)

    if kernel.lower() == 'rematch':
        # build the REMatch kernel
        re = REMatchKernel(metric="cosine", alpha=0.6, threshold=1e-6, gamma=1)
        print("Building Kernel")
        _kernel = re.create(all_features)
        print("kernel build: " + str(np.shape(_kernel)))
    if kernel.lower() == 'average':
        k = AverageKernel(metric="cosine")
        _kernel = k.create(all_features)
        print("kernel build: " + str(np.shape(_kernel)))

    if save_kernel:
        print('saving kernel')
        kernel_name = X + '_' + kernel + '.bp'
        pickle.dump([_kernel, all_systems], open(kernel_name, 'wb'))

    return kernel


def map_perovskites(kernel_pickle=None, color_by='sigma'):
    color_dict = {'F': '#061283', 'Cl': '#FD3C3C', 'Br': '#FFB74C', 'I': '#138D90'}

    data = pickle.load(open(kernel_pickle, 'rb'))
    kernel = data[0]
    system_info = data[1]

    colors = []

    for i in system_info:
        _system = map_ase_atoms_to_crystal(i['atoms'])
        a = [_a.label for _a in _system.all_atoms(sort=True, unique=True)]

        if color_by == 'sigma':
            if i['sigma'] == None:
                colors.append(100000)
            else:
                colors.append(i['sigma_mode_averaged'])

        if color_by == 'B':
            aa = [_a for _a in a if _a not in ['Li', 'Na', 'K', 'Rb', 'Cs', 'F', 'Cl', 'Br', 'I']]
            atomic_number_list = [atomic_numbers[_a] for _a in aa]
            averaged_B_number = sum(atomic_number_list) / len(atomic_number_list)
            colors.append(averaged_B_number)

        if color_by == 'X':
            if 'F' in a:
                colors.append('#F1F3F2')
            if 'Cl' in a:
                colors.append('#F1F3F2')
            if 'Br' in a:
                colors.append('#F1F3F2')
            if 'I' in a:
                colors.append(color_dict['I'])
        # if color_by == 'B':
        #    B = [item for item in a if item in halide_B]
        #    colors.append(atomic_numbers[B[0]])
        if color_by == 'A':
            if 'Li' in a:
                # colors.append('#344d90')
                colors.append('#F1F3F2')
            elif 'Na' in a:
                colors.append('#F1F3F2')
                # colors.append('#5cc5ef')
            elif 'K' in a:
                colors.append('#F1F3F2')
                # colors.append("#ffb745")
            elif 'Rb' in a:
                colors.append('#F1F3F2')
                # colors.append("#ffbebd")
            elif 'Cs' in a:
                # colors.append('#F1F3F2')
                colors.append("#CB0000")

    from matplotlib.patches import Patch

    if color_by == 'X':
        legend_elements = [  # Patch(facecolor=color_dict['F'], edgecolor='k', label='X=F')]
            # Patch(facecolor=color_dict['Cl'], edgecolor='k', label='X=Cl')
            # Patch(facecolor=color_dict['Br'], edgecolor='k', label='X=Br')]
            Patch(facecolor=color_dict['I'], edgecolor='k', label='X=I')]
    if color_by == 'A':
        legend_elements = [  # Patch(facecolor='#344d90', edgecolor='k', label='A=Li')]
            # Patch(facecolor='#5cc5ef', edgecolor='k', label='A=Na')]
            # Patch(facecolor="#ffb745", edgecolor='k', label='A=K')]
            # Patch(facecolor="#ffbebd", edgecolor='k', label='A=Rb')]
            Patch(facecolor="#CB0000", edgecolor='k', label='A=Cs')]

    kpca = KernelPCA(n_components=None, kernel="precomputed", fit_inverse_transform=False)
    X_kpca = kpca.fit_transform(kernel)

    fig = plt.figure()
    if color_by == 'B':
        fig, ax = plt.subplots()
        cm = plt.cm.get_cmap('YlGnBu')
        aplot = ax.scatter(X_kpca[:, 0], X_kpca[:, 1], c=colors, alpha=0.5, cmap=cm)
        cbaxes = ax.inset_axes([0.06, 0.12, 0.4, 0.02])
        cbar = fig.colorbar(aplot, cax=cbaxes, orientation='horizontal')
        cbar.set_label('$\\bar{Z_{B}}$')
    elif color_by == 'sigma':
        fig, ax = plt.subplots()
        cm = plt.cm.get_cmap('Blues')
        aplot = ax.scatter(X_kpca[:, 0], X_kpca[:, 1], c=colors, alpha=0.7, cmap=cm,  vmin=-1, vmax=5)
        cbaxes = ax.inset_axes([0.06, 0.12, 0.4, 0.02])
        cbar = fig.colorbar(aplot, cax=cbaxes, orientation='horizontal')
        #cbar.set_label('$\\sigma^{(2)}$')
        cbar.set_label('$\\langle\\omega\\rangle_{\\sigma}$ (THz)')
    else:
        plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=colors, alpha=0.7)

    if color_by == 'X':
        plt.legend(handles=legend_elements, loc=3, fontsize=12, ncol=2)
    if color_by == 'A':
        plt.legend(handles=legend_elements, loc=3, fontsize=12, ncol=2)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('map.pdf')


def main():
    parser = argparse.ArgumentParser(
        description='Switches for analyzing the screening results of bulk cubic perovskites',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--C", type=str, default='halides',
                        help="Anion in ABCs.")
    parser.add_argument("--descriptor", type=str, default='SOAP')
    parser.add_argument("--kernel", type=str, default='REMatch')
    parser.add_argument("--build_kernel", action='store_true', help='build and store the similarity kernel')
    parser.add_argument("--build_map", action='store_true', help='build the low-dimensional map of the landscape')
    parser.add_argument("--kernel_pickle", type=str, default='halides_REMatch.bp')

    args = parser.parse_args()

    if args.build_kernel:
        build_kernels_for_perovskite_systems(kernel=args.kernel, descriptor=args.descriptor)
    if args.build_map:
        map_perovskites(kernel_pickle=args.kernel_pickle, color_by='sigma')


if __name__ == "__main__":
    main()
