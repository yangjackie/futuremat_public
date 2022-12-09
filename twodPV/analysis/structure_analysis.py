from core.models import cVector3D
from twodPV.bulk_library import A_site_list, B_site_list, C_site_list
from twodPV.analysis.bulk_energy_landscape import *
from core.models.element import *
from ase.db import connect
from ase.geometry.analysis import Analysis
import argparse
import os
import numpy as np

from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

rc('text', usetex=True)
params = {'legend.fontsize': '12',
          'figure.figsize': (6, 5),
          'axes.labelsize': 20,
          'axes.titlesize': 13,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12}
pylab.rcParams.update(params)


def BO_bond_vectors_in_bulk_perovskite(db, a, b, c):
    system_name = a + b + c
    uid = system_name + '3_pm3m'
    row = db.get(selection=[('uid', '=', uid)])
    atoms = row.toatoms()
    analyzer = Analysis(atoms)
    all_vectors = []
    for bonds in analyzer.get_bonds(b, c)[0]:
        v = cVector3D(*atoms.get_distances(bonds[0], [bonds[1]], mic=True, vector=True)[0])
        v = cVector3D(*v)
        all_vectors.append(v)
    return all_vectors


def BO_bond_vectors_in_twod_perovskites(db, a, b, c, orientation, termination, thickness):
    system_name = a + b + c
    uid = system_name + '3_' + str(orientation) + "_" + str(termination) + "_" + str(thickness)
    print(uid)
    try:
        row = db.get(selection=[('uid', '=', uid)])
    except:
        return None
    atoms = row.toatoms()
    analyzer = Analysis(atoms)
    all_vectors = []
    for bonds in analyzer.get_bonds(b, c)[0]:
        v = cVector3D(*atoms.get_distances(bonds[0], [bonds[1]], mic=True, vector=True)[0])
        v = cVector3D(*v)
        all_vectors.append(v)
    return all_vectors


def polarisation(db, a, b, c, orientation, termination, thickness):
    system_name = a + b + c
    uid = system_name + '3_' + str(orientation) + "_" + str(termination) + "_" + str(thickness)
    print(uid)
    pol = None
    try:
        row = db.get(selection=[('uid', '=', uid)])
        pol = row.key_value_pairs['e_pol'] + row.key_value_pairs['nu_pol']
    except:
        pass
    return pol



def polarisation_out_distributions(db):
    termination_types = {'100': ['AO', 'BO2'], '110': ['ABO', 'O2'], '111': ['AO3', 'B']}
    two_d_out_of_plane_vec = cVector3D(0.0, 0.0, 1.0)
    S_out_of_plane = []
    polarisations = []
    colors = []
    sizes = []



    in_plane_statistics = {'100': {3: [], 5: [], 7: [], 9: []},
                           '110': {3: [], 5: [], 7: [], 9: []},
                           '111': {3: [], 5: [], 7: [], 9: []}}
    out_of_plane_statistics = {'100': {3: [], 5: [], 7: [], 9: []},
                               '110': {3: [], 5: [], 7: [], 9: []},
                               '111': {3: [], 5: [], 7: [], 9: []}}

    # color_dict = {3: '#07000E', 5: '#D75404', 7: '#522E75', 9: '#D50B53'}

    color_dict = {'100': '#F2A104', '110': '#00743F', '111': '#1D65A6'}
    counter = 0

    plt.figure(figsize=(14, 9))

    for id, orientation in enumerate(['100', '110', '111']):
        _or = [float(c) for c in orientation]
        _or_vec = cVector3D(_or[0], _or[1], _or[2]).normalise()
        for term_id, term_type in enumerate(termination_types[orientation]):
            S_out_of_plane = []
            polarisations = []

            sto_pol = []
            sto_s = []
            sto_sizes = []

            bto_pol = []
            bto_s = []
            bto_sizes = []

            cpb_pol = []
            cpb_s = []
            cpb_sizes = []

            for i in range(len(A_site_list)):
                for a in A_site_list[i]:
                    for b in B_site_list[i]:
                        for c in C_site_list[i]:
                            bo_vecs_in_pm3m = BO_bond_vectors_in_bulk_perovskite(db, a, b, c)
                            bulk_out_comps = [abs(bv.dot(_or_vec)) for bv in bo_vecs_in_pm3m]

                            bulk_inplane_comps = [bv - _or_vec.vec_scale(bv.dot(_or_vec) / _or_vec.l2_norm()) for bv in
                                                  bo_vecs_in_pm3m]
                            bulk_inplane_comps = [v.l2_norm() for v in bulk_inplane_comps]

                            bulk_out_comp_max = max(bulk_out_comps)
                            bulk_in_plane_max = max(bulk_inplane_comps)

                            _delta_inplane_bulk = [v / bulk_in_plane_max for v in bulk_inplane_comps]
                            _delta_inplane_bulk = -1.0 * sum(_delta_inplane_bulk) / len(_delta_inplane_bulk)

                            _delta_out_of_plane_bulk = [v / bulk_out_comp_max for v in bulk_out_comps]
                            _delta_out_of_plane_bulk = -1.0 * sum(_delta_out_of_plane_bulk) / len(
                                _delta_out_of_plane_bulk)

                            for thick in [3, 5, 7, 9]:
                                counter += 1
                                print(counter)
                                two_d_BO_bonds_vec = BO_bond_vectors_in_twod_perovskites(db, a, b, c, orientation,
                                                                                         term_type, thick)
                                _pol = polarisation(db, a, b, c, orientation, term_type, thick)

                                two_d_out_of_plane_components = [abs(v.dot(two_d_out_of_plane_vec)) / bulk_out_comp_max
                                                                 for v in two_d_BO_bonds_vec]
                                two_d_in_plane_components = [
                                    v - two_d_out_of_plane_vec.vec_scale(v.dot(two_d_out_of_plane_vec)) for v in
                                    two_d_BO_bonds_vec]
                                two_d_in_plane_components = [v.l2_norm() / bulk_in_plane_max for v in
                                                             two_d_in_plane_components]

                                _delta_out_of_plane_two_d = sum(two_d_out_of_plane_components) / len(
                                    two_d_out_of_plane_components)
                                _delta_inplane_two_d = sum(two_d_in_plane_components) / len(two_d_in_plane_components)

                                _S_out = _delta_out_of_plane_two_d + _delta_out_of_plane_bulk
                                _S_in = _delta_inplane_two_d + _delta_inplane_bulk

                                if _pol is not None:
                                    S_out_of_plane.append(_S_out)
                                    polarisations.append(_pol)
                                    colors.append(color_dict[orientation])
                                    sizes.append(thick * 15)

                                if (a=='Sr') and (b=='Ti') and (c=='O'):
                                    sto_pol.append(_pol)
                                    sto_s.append(_S_out)
                                    sto_sizes.append(thick * 15)
                                if (a=='Ba') and (b=='Ti') and (c=='O'):
                                    bto_pol.append(_pol)
                                    bto_s.append(_S_out)
                                    bto_sizes.append(thick * 15)
                                if (a=='Cs') and (b=='Pb') and (c=='Br'):
                                    cpb_pol.append(_pol)
                                    cpb_s.append(_S_out)
                                    cpb_sizes.append(thick * 15)

            slot = id + 1 + term_id * 3
            plt.subplot(2, 3, slot)
            if term_id == 0:
                plt.scatter(S_out_of_plane, polarisations, marker='o', alpha=0.2, edgecolor='None',
                           facecolor=color_dict[orientation], s=sizes)
                plt.scatter(sto_s, sto_pol, marker='o', edgecolor='k',
                            facecolor='None', s=sto_sizes, alpha=0.8)
                plt.scatter(bto_s, bto_pol, marker='s', edgecolor='k',
                            facecolor='None', s=bto_sizes, alpha=0.8)
                plt.scatter(cpb_s, cpb_pol, marker='d', edgecolor='k',
                            facecolor='None', s=cpb_sizes, alpha=0.8)
            else:
                plt.scatter(S_out_of_plane, polarisations, marker='s', alpha=0.2, edgecolor='None',
                           facecolor=color_dict[orientation], s=sizes)
                plt.scatter(sto_s, sto_pol, marker='o', edgecolor='k',
                            facecolor='None', s=sto_sizes, alpha=0.8)
                plt.scatter(bto_s, bto_pol, marker='s', edgecolor='k',
                            facecolor='None', s=bto_sizes, alpha=0.8)
                plt.scatter(cpb_s, cpb_pol, marker='d', edgecolor='k',
                            facecolor='None', s=cpb_sizes, alpha=0.8)
            if slot == 1:
                legend_elements = [Patch(facecolor=color_dict['100'], edgecolor='k', label='$[100]$'),
                                   Patch(facecolor=color_dict['110'], edgecolor='k', label='$[110]$'),
                                   Patch(facecolor=color_dict['111'], edgecolor='k', label='$[111]$'),
                                   Line2D([0], [0], marker='o', color='k', label='SrTiO$_3$',
                                          markerfacecolor='none', markersize=12),
                                   Line2D([0], [0], marker='s', color='k', label='BaTiO$_3$',
                                          markerfacecolor='none', markersize=12),
                                   Line2D([0], [0], marker='d', color='k', label='CsPbBr$_3$',
                                          markerfacecolor='none', markersize=12)
                                   ]
                plt.legend(handles=legend_elements, loc=4, fontsize=13, ncol=1)
            plt.xlabel("$\Delta S_{\perp}$")
            plt.ylabel("$P_{z}$")
            plt.ylim([-40,40])
            #plt.xlim([-1.2,0.55])

            if slot == 1: textstr = "AX-termination"
            if slot == 2: textstr = 'ABX-termination'
            if slot == 3: textstr = 'AX$_{3}$-termination'
            if slot == 4: textstr = 'BX$_{2}$-termination'
            if slot == 5: textstr = 'X$_{2}$-termination'
            if slot == 6: textstr = 'B-termination'
            plt.title(textstr)

    # plt.scatter(S_out_of_plane, polarisations, marker='o', s=sizes, alpha=0.5, edgecolor='None', facecolor=colors)
    plt.tight_layout()
    plt.savefig('S_out_pol.pdf')


def bond_lengths_distributions(db, direction='in'):
    termination_types = {'100': ['AO', 'BO2'], '110': ['O2', 'ABO'], '111': ['AO3', 'B']}
    two_d_out_of_plane_vec = cVector3D(0.0, 0.0, 1.0)
    S_out_of_plane = []
    S_in_plane = []
    tolerance_factors = []
    colors = []
    sizes = []

    in_plane_statistics = {'100': {3: [], 5: [], 7: [], 9: []},
                           '110': {3: [], 5: [], 7: [], 9: []},
                           '111': {3: [], 5: [], 7: [], 9: []}}
    out_of_plane_statistics = {'100': {3: [], 5: [], 7: [], 9: []},
                               '110': {3: [], 5: [], 7: [], 9: []},
                               '111': {3: [], 5: [], 7: [], 9: []}}

    # color_dict = {3: '#07000E', 5: '#D75404', 7: '#522E75', 9: '#D50B53'}

    color_dict = {'100': '#F2A104', '110': '#00743F', '111': '#1D65A6'}
    counter = 0
    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:

                    tolerance_f = ionic_radii[a][charge_state_A_site[i]] + ionic_radii[c][charge_state_C_site[i]]
                    tolerance_f /= ionic_radii[b][charge_state_B_site[i]] + ionic_radii[c][charge_state_C_site[i]]
                    tolerance_f /= math.sqrt(2)

                    bo_vecs_in_pm3m = BO_bond_vectors_in_bulk_perovskite(db, a, b, c)
                    for orientation in ['100', '110', '111']:
                        _or = [float(c) for c in orientation]
                        _or_vec = cVector3D(_or[0], _or[1], _or[2]).normalise()

                        bulk_out_comps = [abs(bv.dot(_or_vec)) for bv in bo_vecs_in_pm3m]

                        bulk_inplane_comps = [bv - _or_vec.vec_scale(bv.dot(_or_vec) / _or_vec.l2_norm()) for bv in
                                              bo_vecs_in_pm3m]
                        bulk_inplane_comps = [v.l2_norm() for v in bulk_inplane_comps]

                        bulk_out_comp_max = max(bulk_out_comps)
                        bulk_in_plane_max = max(bulk_inplane_comps)

                        _delta_inplane_bulk = [v / bulk_in_plane_max for v in bulk_inplane_comps]
                        _delta_inplane_bulk = -1.0 * sum(_delta_inplane_bulk) / len(_delta_inplane_bulk)

                        _delta_out_of_plane_bulk = [v / bulk_out_comp_max for v in bulk_out_comps]
                        _delta_out_of_plane_bulk = -1.0 * sum(_delta_out_of_plane_bulk) / len(_delta_out_of_plane_bulk)

                        for term_type in termination_types[orientation]:
                            for thick in [9, 5, 7, 3]:
                                counter += 1
                                print(counter)
                                two_d_BO_bonds_vec = BO_bond_vectors_in_twod_perovskites(db, a, b, c, orientation,
                                                                                         term_type, thick)

                                two_d_out_of_plane_components = [abs(v.dot(two_d_out_of_plane_vec)) / bulk_out_comp_max
                                                                 for v in two_d_BO_bonds_vec]
                                two_d_in_plane_components = [
                                    v - two_d_out_of_plane_vec.vec_scale(v.dot(two_d_out_of_plane_vec)) for v in
                                    two_d_BO_bonds_vec]
                                two_d_in_plane_components = [v.l2_norm() / bulk_in_plane_max for v in
                                                             two_d_in_plane_components]

                                _delta_out_of_plane_two_d = sum(two_d_out_of_plane_components) / len(
                                    two_d_out_of_plane_components)
                                _delta_inplane_two_d = sum(two_d_in_plane_components) / len(two_d_in_plane_components)

                                _S_out = _delta_out_of_plane_two_d + _delta_out_of_plane_bulk
                                S_out_of_plane.append(_S_out)

                                out_of_plane_statistics[orientation][thick].append(_S_out)

                                _S_in = _delta_inplane_two_d + _delta_inplane_bulk
                                S_in_plane.append(_S_in)

                                in_plane_statistics[orientation][thick].append(_S_in)

                                tolerance_factors.append(tolerance_f)
                                colors.append(color_dict[orientation])
                                sizes.append(pylab.rcParams['lines.markersize'] ** 2 * thick / 3)

    legend_elements = [Patch(facecolor=color_dict['100'], edgecolor='k', label='$[100]$'),
                       Patch(facecolor=color_dict['110'], edgecolor='k', label='$[110]$'),
                       Patch(facecolor=color_dict['111'], edgecolor='k', label='$[111]$')]

    gs = gridspec.GridSpec(1, 4, width_ratios=[3.5, 1, 1, 1])
    gs.update(wspace=0.025, hspace=0.05)
    fig = plt.subplots(figsize=(15, 6))
    ax = plt.subplot(gs[0])

    if direction == 'in':
        y = S_in_plane
        ylabel = '$\Delta S_{\parallel}$'
    elif direction == 'out':
        y = S_out_of_plane
        ylabel = '$\Delta S_{\perp}$'
    ax.scatter(tolerance_factors, y, marker='o', edgecolor='None', facecolor=colors, alpha=0.5, s=sizes)
    ax.set_xlabel('Tolerance factor')
    ax.set_ylabel(ylabel)

    if direction == 'in':
        ax.set_ylim([-0.6, 0.3])
    elif direction == 'out':
        ax.set_ylim([-1.1, 0.6])

    ax.legend(handles=legend_elements, loc=3, fontsize=12, ncol=1)

    from scipy.stats import gaussian_kde
    import numpy as np

    def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
        # Kernel Density Estimation with Scipy
        # Note that scipy weights its bandwidth by the covariance of the
        # input data.  To make the results comparable to the other methods,
        # we divide the bandwidth by the sample standard deviation here.
        kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
        return kde.evaluate(x_grid)

    if direction == 'in':
        y_grid = np.linspace(-0.6, 0.3, 1000)
        _statistics = in_plane_statistics
    elif direction == 'out':
        y_grid = np.linspace(-1.1, 0.6, 1000)
        _statistics = out_of_plane_statistics

    # -------------------------------------------------------------------------------------------------------------------
    ax1 = plt.subplot(gs[1])
    _or = '100'

    total = _statistics[_or][3] + _statistics[_or][5] + _statistics[_or][7] + _statistics[_or][9]

    total_pdf = kde_scipy(np.array(total), y_grid, bandwidth=0.05)
    ax1.plot(total_pdf, y_grid, '-', lw=4, c=color_dict['100'])

    pdf = kde_scipy(np.array(_statistics[_or][3]), y_grid, bandwidth=0.05)
    ax1.plot(pdf / 4.0, y_grid, '--', lw=1, c='r')
    pdf = kde_scipy(np.array(_statistics[_or][5]), y_grid, bandwidth=0.05)
    ax1.plot(pdf / 4.0, y_grid, '--', lw=1, c='b')
    pdf = kde_scipy(np.array(_statistics[_or][7]), y_grid, bandwidth=0.05)
    ax1.plot(pdf / 4.0, y_grid, '--', lw=1, c='m')
    pdf = kde_scipy(np.array(_statistics[_or][9]), y_grid, bandwidth=0.05)
    ax1.plot(pdf / 4.0, y_grid, '--', lw=1, c='y')

    ax1.plot(total_pdf, [0 for _ in total_pdf], 'k--')

    if direction == 'in':
        ax1.set_ylim([-0.6, 0.3])
    elif direction == 'out':
        ax1.set_ylim([-1.1, 0.6])

    labels = [item.get_text() for item in ax1.get_yticklabels()]
    empty_string_labels = [''] * len(labels)
    ax1.set_yticklabels(empty_string_labels)
    ax1.set_xlim([0, max(total_pdf)])
    # -------------------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------------------
    ax1 = plt.subplot(gs[2])
    _or = '110'

    total = _statistics[_or][3] + _statistics[_or][5] + _statistics[_or][7] + _statistics[_or][9]

    total_pdf = kde_scipy(np.array(total), y_grid, bandwidth=0.05)
    ax1.plot(total_pdf, y_grid, '-', lw=4, c=color_dict['110'])

    pdf = kde_scipy(np.array(_statistics[_or][3]), y_grid, bandwidth=0.05)
    ax1.plot(pdf / 4.0, y_grid, '--', lw=1, c='r')
    pdf = kde_scipy(np.array(_statistics[_or][5]), y_grid, bandwidth=0.05)
    ax1.plot(pdf / 4.0, y_grid, '--', lw=1, c='b')
    pdf = kde_scipy(np.array(_statistics[_or][7]), y_grid, bandwidth=0.05)
    ax1.plot(pdf / 4.0, y_grid, '--', lw=1, c='m')
    pdf = kde_scipy(np.array(_statistics[_or][9]), y_grid, bandwidth=0.05)
    ax1.plot(pdf / 4.0, y_grid, '--', lw=1, c='y')

    ax1.plot(total_pdf, [0 for _ in total_pdf], 'k--')

    if direction == 'in':
        ax1.set_ylim([-0.6, 0.3])
    elif direction == 'out':
        ax1.set_ylim([-1.1, 0.6])

    labels = [item.get_text() for item in ax1.get_yticklabels()]
    empty_string_labels = [''] * len(labels)
    ax1.set_yticklabels(empty_string_labels)
    ax1.set_xlim([0, max(total_pdf)])
    # -------------------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------------------
    ax1 = plt.subplot(gs[3])
    _or = '111'

    total = _statistics[_or][3] + _statistics[_or][5] + _statistics[_or][7] + _statistics[_or][9]

    total_pdf = kde_scipy(np.array(total), y_grid, bandwidth=0.05)
    ax1.plot(total_pdf, y_grid, '-', lw=4, c=color_dict['111'], label='Total')

    pdf = kde_scipy(np.array(_statistics[_or][3]), y_grid, bandwidth=0.05)
    ax1.plot(pdf / 4.0, y_grid, '--', lw=1, c='r', label='$n=3$')
    pdf = kde_scipy(np.array(_statistics[_or][5]), y_grid, bandwidth=0.05)
    ax1.plot(pdf / 4.0, y_grid, '--', lw=1, c='b', label='$n=5$')
    pdf = kde_scipy(np.array(_statistics[_or][7]), y_grid, bandwidth=0.05)
    ax1.plot(pdf / 4.0, y_grid, '--', lw=1, c='m', label='$n=7$')
    pdf = kde_scipy(np.array(_statistics[_or][9]), y_grid, bandwidth=0.05)
    ax1.plot(pdf / 4.0, y_grid, '--', lw=1, c='y', label='$n=9$')

    ax1.plot(total_pdf, [0 for _ in total_pdf], 'k--')

    ax1.legend()

    if direction == 'in':
        ax1.set_ylim([-0.6, 0.3])
    elif direction == 'out':
        ax1.set_ylim([-1.1, 0.6])

    labels = [item.get_text() for item in ax1.get_yticklabels()]
    empty_string_labels = [''] * len(labels)
    ax1.set_yticklabels(empty_string_labels)
    ax1.set_xlim([0, max(total_pdf)])
    # -------------------------------------------------------------------------------------------------------------------

    plt.tight_layout()
    plt.savefig('BO_bond_change_' + str(direction) + '.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Switches for analyzing the structures of 2D perovskites',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--db", type=str, default=os.getcwd() + '/2dpv.db',
                        help="Name of the database that contains the results of the screenings.")
    parser.add_argument('--direction', type=str, default='in')
    args = parser.parse_args()

    if os.path.exists(args.db):
        args.db = connect(args.db)
    else:
        raise Exception("Database " + args.db + " does not exists, cannot proceed!")

    bond_lengths_distributions(args.db, direction=args.direction)

    #polarisation_out_distributions(args.db)
