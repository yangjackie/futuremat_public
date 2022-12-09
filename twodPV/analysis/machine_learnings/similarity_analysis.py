from dscribe.descriptors import SOAP, MBTR
from dscribe.kernels import REMatchKernel, AverageKernel
from sklearn.preprocessing import normalize
import numpy as np
from ase.db import connect
import argparse
import os
import math
import pickle

import matplotlib.pyplot as plt
from matplotlib import rc

from core.internal.builders.crystal import map_ase_atoms_to_crystal
from core.models import atomic_numbers
from core.models.element import ionic_radii

rc('text', usetex=True)
import matplotlib.pylab as pylab
from matplotlib.patches import Patch

params = {'legend.fontsize': '15',
          'figure.figsize': (7, 6),
          'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)

from twodPV.bulk_library import A_site_list, B_site_list, C_site_list

from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import KernelPCA

charge_state_A_site = {0: 1, 1: 1, 2: 2, 3: 1}
charge_state_B_site = {0: 2, 1: 2, 2: 4, 3: 5}
charge_state_C_site = {0: -1, 1: -1, 2: -2, 3: -2}


def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)


def map_structures_with_given_orientation(db, orientation='100', kernel='average', rcut=4, nmax=8, lmax=9, sigma=0.05,
                                          centers=['A','B','C'], color_by='B', descriptor='soap',save_kernel_kpca=True):
    termination_types = {'100': ['BO2', 'AO'], '110': ['O2', 'ABO'], '111': ['AO3', 'B']}
    anion_color_dict = {'F': '#6FB98F', 'Cl': '#2C7873', 'Br': '#004445', 'I': '#021C1E', 'O': '#8D230F',
                        'S': '#Fcc875', 'Se': '#E8A735', 'Te': '#E29930'}
    thick_color_dict = {3: '#258039', 5: '#F5BE41', 7: '#31A9B8', 9: '#CF3721'}
    b_color_dict = {0: 'r', 1: 'b', 2: 'y', 3: 'b'}
    all_systems_rows = []
    center_lists = []

    color_attributes = []

    if save_kernel_kpca:
        data_dict = {'system':[],'energy':[],'tolerance_factor':[],'termination':[],'kpca1':[],'kpca2':[],'structures':[]}

    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:
                    system_name = a + b + c
                    pm3m_formation_e = None

                    system_name = a + b + c
                    uid = system_name + '3_pm3m'
                    row = None
                    row = db.get(selection=[('uid', '=', uid)])

                    if row is not None:
                        try:
                            pm3m_formation_e = row.key_value_pairs['formation_energy']
                        except KeyError:
                            continue
                    if pm3m_formation_e is None:
                        continue
                    bulk_atoms = row.toatoms()
                    lattice_const = bulk_atoms.cell[0][0]

                    tolerance_f = ionic_radii[a][charge_state_A_site[i]] + ionic_radii[c][charge_state_C_site[i]]
                    tolerance_f /= ionic_radii[b][charge_state_B_site[i]] + ionic_radii[c][charge_state_C_site[i]]
                    tolerance_f /= math.sqrt(2)

                    for t, term in enumerate(termination_types[orientation]):
                        for thick in [3, 5, 7, 9]:
                            uid = system_name + '3_' + str(orientation) + "_" + str(term) + "_" + str(thick)

                            row = None
                            row = db.get(selection=[('uid', '=', uid)])
                            formation_e=None
                            if row is not None:
                                try:
                                    formation_e = row.key_value_pairs['formation_energy']
                                except KeyError:
                                    continue

                            if formation_e is None:
                                continue
                            print(uid)
                            all_systems_rows.append(row)

                            _this_center = []
                            for center in centers:
                                if center == "A":
                                    _this_center.append(a)
                                elif center == "B":
                                    _this_center.append(b)
                                elif center == "C":
                                    _this_center.append(c)
                            center_lists.append(_this_center)

                            if color_by == 'C':
                                color_attributes.append(anion_color_dict[c])
                            elif color_by == 'B':
                                color_attributes.append(b_color_dict[i])
                            elif color_by == 'thickness':
                                color_attributes.append(thick_color_dict[thick])
                            elif color_by == 'term':
                                if t == 0:
                                    tc = '#FFD662FF'
                                elif t == 1:
                                    tc = '#00539CFF'
                                color_attributes.append(tc)
                            elif color_by == 'lattice_const':
                                color_attributes.append(tolerance_f)
                            elif color_by == 'energy':
                                color_attributes.append(formation_e-pm3m_formation_e)

                            if save_kernel_kpca:
                                data_dict['system'].append(uid)
                                _atoms=row
                                data_dict['structures'].append(_atoms)
                                data_dict['energy'].append(formation_e)
                                data_dict['tolerance_factor'].append(tolerance_f)
                                data_dict['termination'].append(term)

    if descriptor == 'soap':
        all_desc = [
            row_to_soap_descriptor(all_systems_rows[i], centers=center_lists[i], rcut=rcut, nmax=nmax, lmax=lmax,
                                     sigma=sigma)
            for i in range(len(all_systems_rows))]
    elif descriptor == 'mbtr':
        all_desc = [row_to_mbtr_descriptor(row) for row in all_systems_rows]

    kernel_matrix = [[1.0 for _ in range(len(all_desc))] for _ in range(len(all_desc))]

    for i in range(len(all_desc) - 1):
        print("calculating kernel " + str(i) + '/' + str(len(all_desc) - 1))

        for j in range(i + 1, len(all_desc)):
            if descriptor == 'soap':
                this_kernel = sum([local_kernel(centre_I_descs=all_desc[i][k], centre_II_descs=all_desc[j][k],
                                                kernel=kernel, descriptor=descriptor) for k in
                                   range(len(centers))]) / len(centers)
            elif descriptor == 'mbtr':
                this_kernel = cosine_similarity(all_desc[i], all_desc[j])[0][0]
            kernel_matrix[i][j] = this_kernel
            kernel_matrix[j][i] = this_kernel

    kpca = KernelPCA(n_components=None, kernel="precomputed", fit_inverse_transform=False)
    X_kpca = kpca.fit_transform(kernel_matrix)

    if save_kernel_kpca:
        data_dict['kpca1'] = X_kpca[:, 0]
        data_dict['kpca2'] = X_kpca[:, 1]

        pickle.dump(data_dict,open("similarity_kernel_" + str(orientation) + '_' + str(descriptor) + '.bp','wb'))

    if color_by not in ['lattice_const','energy']:
        plt.scatter(X_kpca[:, 0], X_kpca[:, 1], alpha=0.85, c=color_attributes)
    else:
        if color_by == 'lattice_const':
            plt.scatter(X_kpca[:, 0], X_kpca[:, 1], alpha=0.95, c=color_attributes, cmap='YlGnBu')
        elif color_by == 'energy':
            import matplotlib.colors as colors
            plt.scatter(X_kpca[:, 0], X_kpca[:, 1], alpha=0.95, c=color_attributes, cmap='YlGnBu_r',vmin=-1.5,vmax=1.5,
                        norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                              vmin=-1.5, vmax=1.5, base=10))

    legend_elements = None
    if color_by == 'C':
        legend_elements = [Patch(facecolor=anion_color_dict[k], edgecolor='none', label=k) for k in
                           anion_color_dict.keys()]
    elif color_by == 'B':
        legend_elements = [Patch(facecolor=b_color_dict[k], edgecolor='none') for k in b_color_dict.keys()]
    elif color_by == 'thickness':
        legend_elements = [Patch(facecolor=thick_color_dict[k], edgecolor='none', label='$n=$' + str(k)) for k in
                           thick_color_dict.keys()]
    elif color_by == 'term':
        legend_elements = [Patch(facecolor='#FFD662FF', edgecolor='none',
                                 label=termination_types[orientation][0].replace('O', 'X').replace('2', '$_2$').replace(
                                     '3', '$_3$')),
                           Patch(facecolor='#00539CFF', edgecolor='none',
                                 label=termination_types[orientation][1].replace('O', 'X').replace('2', '$_2$').replace(
                                     '3', '$_3$'))]
    if color_by not in ['lattice_const','energy']:
        plt.legend(handles=legend_elements)
    else:
        if color_by == 'energy':
            ticks = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
            #cbar = plt.colorbar(ticks=ticks)
            labels = []
            for n,t in enumerate(ticks):
                if t==-1:
                    labels.append('$-1$')
                elif t==-0.1:
                    labels.append('$-0.1$')
                elif t==0:
                    labels.append('$0$')
                elif t==0.1:
                    labels.append('$0.1$')
                elif t==1:
                    labels.append('$1$')
                elif t == 0.5:
                    labels.append('$0.5$')
                elif t == -0.5:
                    labels.append('$-0.5$')
                else:
                    labels.append('')
            #cbar.ax.set_yticklabels(labels)
        else:
            plt.colorbar()

    plt.axis('off')
    plt.tight_layout()
    plt.savefig("similarity_map_" + str(orientation) + '_' + str(descriptor) + '.pdf')


def row_to_soap_descriptor(row, rcut=4, nmax=8, lmax=9, sigma=0.05, centers=['B']):
    """
    This helper function take a row from the database and calculate the corresponding SOAP
    descriptor for the corresponding chemical structure stored in that row.
    """
    global_descriptor = []
    print("Generating the descriptor for :" + row.key_value_pairs['uid'])
    atoms = row.toatoms()
    crystal = map_ase_atoms_to_crystal(atoms)
    __atomic_numbers = list(set([atomic_numbers[a.label] for a in crystal.all_atoms(sort=True, unique=True)]))

    # initializing the descriptor
    desc = SOAP(species=__atomic_numbers, rcut=rcut, nmax=nmax, lmax=lmax, sigma=sigma, periodic=True,
                crossover=False, sparse=False)

    # find out the atomic numbers corresponding to the type of atom around which we want to expand the chemical environment

    for center in centers:
        this_site_indices = [atom.index for atom in atoms if atom.symbol == center]
        feature = desc.create(atoms, positions=this_site_indices)
        feature = normalize(feature)
        global_descriptor.append(feature)

    return global_descriptor


def row_to_mbtr_descriptor(row):
    """
    This helper function take a row from the database and return a many-body tensor representation of the structure.
    For details, see Haoyan Huo and Matthias Rupp. Unified representation of molecules and crystals for
    machine learning. arXiv e-prints, pages arXiv:1704.06439, Apr 2017.
    """
    print("Generating the descriptor for :" + row.key_value_pairs['uid'])
    atoms = row.toatoms()
    crystal = map_ase_atoms_to_crystal(atoms)
    __atomic_numbers = list(set([atomic_numbers[a.label] for a in crystal.all_atoms(sort=True, unique=True)]))

    # this is just borrowed from https://singroup.github.io/dscribe/0.3.x/tutorials/mbtr.html#mbtr
    k1 = {
        "geometry": {"function": "atomic_number"},
        "grid": {"min": 1, "max": 200, "sigma": 0.05, "n": 200}  # take into account all elements in the periodic table
    }

    k2 = {
        "geometry": {"function": "inverse_distance"},
        "grid": {"min": 0.1, "max": 2, "sigma": 0.05, "n": 50},
        "weighting": {"function": "exp", "scale": 0.75, "cutoff": 1e-2}
    }

    k3 = {
        "geometry": {"function": "angle"},
        "grid": {"min": 0, "max": 180, "sigma": 5, "n": 50},
        "weighting": {"function": "exp", "scale": 0.5, "cutoff": 1e-3}
    }

    desc = MBTR(species=__atomic_numbers, periodic=True, k1=k1, k2=k2, k3=k3, flatten=True, normalization='n_atoms')
    global_descriptor = desc.create(atoms)
    # print(max(global_descriptor[0][:50]))
    return global_descriptor


def compare_b_site_coordination_environment_to_bulk(db, orientation='100', termination='AO', kernel='average', rcut=4,
                                                    nmax=8, lmax=9, sigma=0.05):
    similarity_dict = {3: [], 5: [], 7: [], 9: []}
    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:

                    pm3m_formation_e = None
                    # get the corresponding bulk structures
                    system_name = a + b + c
                    uid = system_name + '3_pm3m'

                    row = None
                    row = db.get(selection=[('uid', '=', uid)])
                    if row is not None:
                        try:
                            pm3m_formation_e = row.key_value_pairs['formation_energy']
                        except KeyError:
                            continue
                    if pm3m_formation_e is None:
                        continue

                    bulk_atoms = row.toatoms()
                    _bulk_system = map_ase_atoms_to_crystal(bulk_atoms)
                    _atomic_numbers = list(
                        set([atomic_numbers[a.label] for a in _bulk_system.all_atoms(sort=True, unique=True)]))
                    b_site_indicies = [atom.index for atom in bulk_atoms if atom.symbol == b]

                    desc = SOAP(species=_atomic_numbers, rcut=rcut, nmax=nmax, lmax=lmax, sigma=sigma, periodic=True,
                                crossover=False, sparse=False)
                    bulk_feature = desc.create(bulk_atoms, positions=b_site_indicies)
                    bulk_feature = normalize(bulk_feature)

                    for thick in [3, 5, 7, 9]:
                        uid = system_name + '3_' + str(orientation) + "_" + str(termination) + "_" + str(thick)
                        row = None
                        row = db.get(selection=[('uid', '=', uid)])
                        twod_formation_e = None
                        if row is not None:
                            try:
                                twod_formation_e = row.key_value_pairs['formation_energy']
                            except KeyError:
                                continue

                        if twod_formation_e is not None:
                            twod_atoms = row.toatoms()
                            twod_system = map_ase_atoms_to_crystal(twod_atoms)
                            _twod_atomic_numbers = list(
                                set([atomic_numbers[a.label] for a in twod_system.all_atoms(sort=True, unique=True)]))
                            twod_b_site_indicies = [atom.index for atom in twod_atoms if atom.symbol == b]
                            desc = SOAP(species=_twod_atomic_numbers, rcut=rcut, nmax=nmax, lmax=lmax, sigma=sigma,
                                        periodic=True,
                                        crossover=False, sparse=False)
                            two_d_feature = desc.create(twod_atoms, positions=twod_b_site_indicies)
                            two_d_feature = normalize(two_d_feature)

                            similarity = local_kernel(bulk_feature, two_d_feature, kernel=kernel)
                            similarity_dict[thick].append(similarity)
                            print(uid, similarity)

    color_dict = {3: '#344D90', 5: '#5CC5EF', 7: '#FFB745', 9: '#E7552C'}
    x_grid = np.linspace(0.9875, 1.0025, 500)
    for thick in [3, 5, 7, 9]:
        pdf = kde_scipy(np.array(similarity_dict[thick]), x_grid, bandwidth=0.0005)
        plt.plot(x_grid, pdf, '-', lw=1.5, c=color_dict[thick], label='$n=$' + str(thick))
    plt.xlabel("SOAP Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("similarity_distr_" + str(orientation) + "_" + str(termination) + '.pdf')


def local_kernel(centre_I_descs=None, centre_II_descs=None, kernel='average', descriptor='soap') -> float:
    """
    Calculate the similarity kernel around a given type of atomic centers in ABX3 perovskites (i.e. either A-, B- or X- sites)
    between two chemical structures I and II. Each chemical structure can contain multiple atomic centres of a given type (i.e.
    there may exists multiple atoms belong to a given site.

    :param centre_I_descs: A list of list containing the descriptors around the atomic centers in structure I.
    :param centre_II_descs: A list of list containing the descriptors around the atomic centers in structure II.
    :param kernel: Types of kernel similarity to be calculated. Options are "average", "rematch", ...
    :return local_similarity: A float that corresponds to the similarity between the two chemical enivronments.
    """
    if (kernel == 'average') and (descriptor == 'soap'):
        similarity = [np.dot(a, b) for a in centre_I_descs for b in centre_II_descs]
        return sum(similarity) / len(similarity)
    elif kernel == 'rematch' and (descriptor == 'soap'):
        re = REMatchKernel(metric="cosine", alpha=0.6, threshold=1e-6, gamma=1)
        similarity = re.create([centre_I_descs, centre_II_descs])
        return similarity[0][1]
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Machine-Learning Analysis for 2D perovskites',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--db", type=str, default=os.getcwd() + '/2dpv.db',
                        help="Name of the database that contains the results of the screenings.")
    parser.add_argument("--orient", type=str, default='100',
                        help='Orientations of the two-d perovskite slabs')
    parser.add_argument("--terminations", type=str, default='AO',
                        help='Surface termination type of the two-d slab')
    parser.add_argument("-cb", "--compare_to_bulk", action='store_true',
                        help='Compare the B-site coordination environment to bulk')
    parser.add_argument("-map", "--map", action='store_true',
                        help='Produce a map of the structures')

    # Something got to do with the similarity kernels
    parser.add_argument('-des', "--descriptor", type=str, default='soap',
                        help='Type of descriptor to be used', choices=['soap', 'mbtr'])
    parser.add_argument("-kl", "--kernel", type=str, default='average',
                        help='Types of kernels used to compare the similarities between two structures',
                        choices=['average,rematch'])
    parser.add_argument("-rc", "--rcut", type=float, default=4.0,
                        help="Cut off radius for finding the atomic environment around a given center")
    parser.add_argument("-lmax", "--lmax", type=int, default=9,
                        help="Maximum angular momentum for expanding the SOAP descriptor, maximum is 9.")
    parser.add_argument("-nmax", "--nmax", type=int, default=8,
                        help="Maximum number of Gaussian functions to expand the radial part of the SOAP descriptor")
    parser.add_argument("-sigma", "--sigma", type=float, default=1.0,
                        help="Gaussian width for atom-centered density.")
    parser.add_argument("-c", "--color_by", type=str, default='C',
                        help="Color the map according to certain attributes",
                        choices=['C', 'B', 'thickness', 'term', 'lattice_const', 'energy'])
    args = parser.parse_args()

    if os.path.exists(args.db):
        args.db = connect(args.db)
    else:
        raise Exception("Database " + args.db + " does not exists, cannot proceed!")

    if args.compare_to_bulk:
        compare_b_site_coordination_environment_to_bulk(args.db, orientation=args.orient, termination=args.terminations,
                                                        kernel=args.kernel, rcut=args.rcut, lmax=args.lmax,
                                                        nmax=args.nmax, sigma=args.sigma)
    elif args.map:
        map_structures_with_given_orientation(args.db, orientation=args.orient, kernel=args.kernel,
                                              descriptor=args.descriptor, rcut=args.rcut,
                                              lmax=args.lmax, nmax=args.nmax, sigma=args.sigma, color_by=args.color_by)
