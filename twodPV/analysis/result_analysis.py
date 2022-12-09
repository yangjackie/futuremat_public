import argparse
import os
from ase.db import connect
from matplotlib import rc, patches
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.lines import Line2D
import numpy as np
import math

from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from twodPV.bulk_library import A_site_list, B_site_list, C_site_list
from core.models.element import ionic_radii
from core.internal.builders.crystal import map_ase_atoms_to_crystal

rc('text', usetex=True)
params = {'legend.fontsize': '12',
          'figure.figsize': (6, 5),
          'axes.labelsize': 20,
          'axes.titlesize': 16,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)

termination_types = {'100': ['AO', 'BO2'], '110': ['O2', 'ABO'], '111': ['AO3', 'B']}

charge_state_A_site = {0: 1, 1: 1, 2: 2, 3: 1}
charge_state_B_site = {0: 2, 1: 2, 2: 4, 3: 5}
charge_state_C_site = {0: -1, 1: -1, 2: -2, 3: -2}


def plot_thickness_dependent_formation_energies(db, orientation='100', output=None):
    """
    Plot the thickness dependent formation energies of two-dimensional perovskite slabs in a specific orientation.
    Sructures will be categorized according to the chemistries of A/B/C sites (columns) and surface terminations
    (rows)
    """
    figs, ((ax11, ax12, ax13, ax14), (ax21, ax22, ax23, ax24)) = plt.subplots(ncols=4, nrows=2, figsize=(20, 12))
    thicknesses = [3, 5, 7, 9]

    for i in range(len(A_site_list)):

        ylabel = None
        if i == 0:
            axs = [ax11, ax21]
            color = '#7AC7A9'
            title = '$A^{I}B^{II}_{M}X_{3}$'
            ylabel = '$\\min[E_{f}^{2D,n}-E_{f}^{Pm\\bar{3}m}]$ (eV/atom)'
        if i == 1:
            axs = [ax12, ax22]
            color = '#90CA57'
            title = '$A^{I}B^{II}_{TM}X_{3}$'
        if i == 2:
            axs = [ax13, ax23]
            color = '#F1D628'
            title = '$A^{II}B^{IV}C_{3}$'
        if i == 3:
            axs = [ax14, ax24]
            color = '#2B8283'
            title = '$A^{I}B^{X}C_{3}$'

        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:
                    # Energy of the bulk material in the perovskite structure
                    system_name = a + b + c
                    uid = system_name + '3_pm3m'
                    row = db.get(selection=[('uid', '=', uid)])
                    pm3m_formation_e = row.key_value_pairs['formation_energy']

                    two_d_en_diff = [[] for _ in range(len(termination_types[orientation]))]
                    for term_type_id, term_type in enumerate(termination_types[orientation]):

                        for t in thicknesses:
                            uid = system_name + '3_' + str(orientation) + "_" + str(term_type) + "_" + str(t)
                            try:
                                row = db.get(selection=[('uid', '=', uid)])
                                twod_formation_e = row.key_value_pairs['formation_energy']
                                two_d_en_diff[term_type_id].append(twod_formation_e - pm3m_formation_e)
                            except KeyError:
                                print('WARNING: No value for ' + uid + ' Skip!')
                                continue

                        axs[term_type_id].plot(thicknesses, two_d_en_diff[term_type_id], 'o:', c=color, alpha=0.9)
                        axs[term_type_id].plot([3, 9], [0, 0], 'k--')

                        if ylabel is not None:
                            axs[term_type_id].set_ylabel(ylabel)

                    axs[0].set_title(title)
                    axs[1].set_xlabel('Layers')

    if orientation == '111':
        ax21.set_ylim([-1, 2])
        ax22.set_ylim([-2, 2.5])
        ax23.set_ylim([-0.5, 2])

    if output is None:
        output = 'two_d_' + str(orientation) + '_' + 'thickness_dependent.png'
    plt.tight_layout()
    plt.savefig(output)
    plt.show()


def plot_thickness_dependent_lowest_imaginary_frequency(db, orientation='100', output=None):
    """
    Plot the thickness dependent formation energies of two-dimensional perovskite slabs in a specific orientation.
    Sructures will be categorized according to the chemistries of A/B/C sites (columns) and surface terminations
    (rows)
    """
    figs, ((ax11, ax12, ax13, ax14), (ax21, ax22, ax23, ax24)) = plt.subplots(ncols=4, nrows=2, figsize=(20, 12))
    thicknesses = [3, 5, 7, 9]

    for i in range(len(A_site_list)):

        ylabel = None
        if i == 0:
            axs = [ax11, ax21]
            color = '#7AC7A9'
            title = '$A^{I}B^{II}_{M}X_{3}$'
            ylabel = '$\\min[E_{f}^{2D,n}-E_{f}^{Pm\\bar{3}m}]$ (eV/atom)'
        if i == 1:
            axs = [ax12, ax22]
            color = '#90CA57'
            title = '$A^{I}B^{II}_{TM}X_{3}$'
        if i == 2:
            axs = [ax13, ax23]
            color = '#F1D628'
            title = '$A^{II}B^{IV}C_{3}$'
        if i == 3:
            axs = [ax14, ax24]
            color = '#2B8283'
            title = '$A^{I}B^{X}C_{3}$'

        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:
                    # Energy of the bulk material in the perovskite structure
                    system_name = a + b + c

                    two_d_omega = [[] for _ in range(len(termination_types[orientation]))]
                    for term_type_id, term_type in enumerate(termination_types[orientation]):
                        _thicknesses = []
                        for t in thicknesses:
                            uid = system_name + '3_' + str(orientation) + "_" + str(term_type) + "_" + str(t)
                            min_omega = None
                            try:
                                row = db.get(selection=[('uid', '=', uid)])
                                gamma_point_freqs = row.data['gamma_phonon_freq']
                                try:
                                    gamma_point_freqs = [(f ** 2).real for f in gamma_point_freqs if f.imag != 0.0]
                                    min_omega = min(gamma_point_freqs)
                                except ValueError:
                                    # print("No imaginary frequency for " + str(uid))
                                    pass

                            except KeyError:
                                print('WARNING: No value for ' + uid + ' Skip!')
                                continue

                            if min_omega is not None:
                                _thicknesses.append(t)
                                two_d_omega[term_type_id].append(min_omega)

                        axs[term_type_id].plot(_thicknesses, two_d_omega[term_type_id], 'o:', c=color, alpha=0.9)
                        axs[term_type_id].plot([3, 9], [0, 0], 'k--')
                        axs[term_type_id].set_ylim([-5.5, 0.5])
                        if ylabel is not None:
                            axs[term_type_id].set_ylabel(ylabel)

                    axs[0].set_title(title)
                    axs[1].set_xlabel('Layers')

    if output is None:
        output = 'two_d_' + str(orientation) + '_' + 'thickness_dependent_freq.pdf'
    plt.tight_layout()
    plt.savefig(output)
    plt.show()


def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)


def plot_energy_versus_bulk_summary(db, output=None, tolerance_factor=False):
    fig, [ax_hist, ax] = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [1, 4]})
    # color_dict = {0: '#7AC7A9', 1: '#90CA57', 2: '#F1D628', 3: '#2B8283'}
    color_dict = {0: '#07000E', 1: '#D75404', 2: '#522E75', 3: '#D50B53'}
    marker_dict = {0: 'o', 1: 's', 2: 'p', 3: 'D'}
    all_twod_e_diff = []

    stable_counter = 0
    unstable_bulk_counter = 0
    total = 0

    for orientation in ['100', '110', '111']:
        for term_type_id, term_type in enumerate(termination_types[orientation]):
            for i in range(len(A_site_list)):
                bulk_e_diff = []
                twod_e_diff = []

                if tolerance_factor: tolerance_factors = []

                for count_a, a in enumerate(A_site_list[i]):
                    for b in B_site_list[i]:
                        for c in C_site_list[i]:

                            if tolerance_factor:
                                tolerance_f = ionic_radii[a][charge_state_A_site[i]] + ionic_radii[c][
                                    charge_state_C_site[i]]
                                tolerance_f /= ionic_radii[b][charge_state_B_site[i]] + ionic_radii[c][
                                    charge_state_C_site[i]]
                                tolerance_f /= math.sqrt(2)

                            system_name = a + b + c
                            uid = system_name + '3_pm3m'
                            row = db.get(selection=[('uid', '=', uid)])
                            pm3m_formation_e = row.key_value_pairs['formation_energy']

                            # Find the lowest energy amongst all the relaxed random structures
                            lowest_relaxed = -100000
                            for k in range(10):
                                uid = system_name + '3_random_str_' + str(k + 1)
                                try:
                                    row = db.get(selection=[('uid', '=', uid)])
                                    randomised_formation_e = row.key_value_pairs['formation_energy']
                                    if (randomised_formation_e > lowest_relaxed):  # and (
                                        # (pm3m_formation_e - randomised_formation_e) > -1.0):
                                        lowest_relaxed = randomised_formation_e
                                except KeyError:
                                    continue

                            uid = system_name + '3_' + str(orientation) + "_" + str(term_type) + "_3"
                            try:
                                row = db.get(selection=[('uid', '=', uid)])
                                twod_formation_e = row.key_value_pairs['formation_energy']

                                # if (twod_formation_e - pm3m_formation_e) <= 10.0:
                                if twod_formation_e - pm3m_formation_e <= 0.2:
                                    stable_counter += 1
                                    if pm3m_formation_e - lowest_relaxed >= 0.0:
                                        unstable_bulk_counter += 1
                                total += 1
                                # bulk_e_diff.append(pm3m_formation_e - lowest_relaxed)
                                bulk_e_diff.append(pm3m_formation_e - 0.0)
                                # twod_e_diff.append(twod_formation_e - pm3m_formation_e)
                                twod_e_diff.append(twod_formation_e - 0.0)

                                if tolerance_factor: tolerance_factors.append(tolerance_f)

                                if (twod_formation_e - pm3m_formation_e > -1.1) and (
                                        twod_formation_e - pm3m_formation_e < 2.0):
                                    # all_twod_e_diff.append(twod_formation_e - pm3m_formation_e)
                                    all_twod_e_diff.append(twod_formation_e - 0.0)
                                    # names[term_type_id].append(system_name + '$_{3}$')
                            except KeyError:
                                print('WARNING: No value for ' + uid + ' Skip!')
                                continue

                print("termination id :" + str(term_type_id) + "  A_site id:" + str(i))

                if term_type_id == 0:
                    if not tolerance_factor:
                        ax.scatter(twod_e_diff, bulk_e_diff, marker=marker_dict[i], edgecolors=color_dict[i],
                                   facecolors=color_dict[i], alpha=0.5, s=60)
                    else:
                        ax.scatter(twod_e_diff, tolerance_factors, marker=marker_dict[i], edgecolors=color_dict[i],
                                   facecolors=color_dict[i], alpha=0.5, s=60)
                if term_type_id == 1:
                    if not tolerance_factor:
                        ax.scatter(twod_e_diff, bulk_e_diff, marker=marker_dict[i], edgecolors=color_dict[i],
                                   facecolors='none', alpha=0.7, s=60)
                    else:
                        ax.scatter(twod_e_diff, tolerance_factors, marker=marker_dict[i], edgecolors=color_dict[i],
                                   facecolors='none', alpha=0.7, s=60)

    print("Number of two-d perovskites with relative stabilities less than 0.2 eV/ atom: " + str(
        stable_counter) + '/' + str(total))
    print("In which " + str(unstable_bulk_counter) + " structures have an unstable bulk perovskites")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_dict[0], edgecolor='none', label='$A^{I}B^{II}_{M}X_{3}$-(AX,X$_{2}$,AX$_{3}$)'),
        Patch(edgecolor=color_dict[0], facecolor='none', label='$A^{I}B^{II}_{M}X_{3}$-(BX$_{2}$,ABX,B)'),
        Patch(facecolor=color_dict[1], edgecolor='none', label='$A^{I}B^{II}_{TM}X_{3}$-(AX,X$_{2}$,AX$_{3}$)'),
        Patch(edgecolor=color_dict[1], facecolor='none', label='$A^{I}B^{II}_{TM}X_{3}$-(BX$_{2}$,ABX,B)'),
        Patch(facecolor=color_dict[2], edgecolor='none', label='$A^{II}B^{IV}C_{3}$-(AC,C$_{2}$,AC$_{3}$)'),
        Patch(edgecolor=color_dict[2], facecolor='none', label='$A^{II}B^{IV}C_{3}$-(BC$_{2}$,ABC,B)'),
        Patch(facecolor=color_dict[3], edgecolor='none', label='$A^{I}B^{X}C_{3}$-(AC,C$_{2}$,AC$_{3}$)'),
        Patch(edgecolor=color_dict[3], facecolor='none', label='$A^{I}B^{X}C_{3}$-(BC$_{2}$,ABC,B)')
    ]

    if not tolerance_factor:
        # ax.set_ylabel('$\\min[E_{f}^{Pm\\bar{3}m}-E_{f}^{\mbox{\\Large{full relax}}}]$ (eV/atom)')
        ax.set_ylabel('$E_{f}^{Pm\\bar{3}m}$ (eV/atom)')
        ax.set_ylim([-2, 0.5])
    else:
        ax.set_ylabel('Tolerance Factor')
        ax.set_ylim([0.5, 1.2])
    # ax.set_xlabel("$E_f^{2D,n=3}-E_{f}^{Pm\\bar{3}m}$ (eV/atom)")
    ax.set_xlabel("$E_f^{2D,n=3}$ (eV/atom)")
    ax.set_xlim([-2.6, 2.1])

    ax.axvspan(-2.6, 0.2, alpha=0.3, color='#EECC8D')

    ax.legend(handles=legend_elements)

    ax_hist.hist(all_twod_e_diff, density=True, bins=150, color='#F2C083')

    x_grid = np.linspace(-2.6, 2.0, 400)
    pdf = kde_scipy(np.array(all_twod_e_diff), x_grid, bandwidth=0.05)
    ax_hist.plot(x_grid, pdf, '-', lw=1.5, c='k')
    ax_hist.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(output)


def plot_size_dependent_energy_histograms(db, output=None):
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(17, 12))
    index = 0
    for orient_id, orientation in enumerate(['100', '110', '111']):
        for i in range(len(A_site_list)):
            index += 1
            twod_e_diff_3 = []
            twod_e_diff_9 = []

            for count_a, a in enumerate(A_site_list[i]):
                for b in B_site_list[i]:
                    for c in C_site_list[i]:
                        system_name = a + b + c
                        uid = system_name + '3_pm3m'
                        row = db.get(selection=[('uid', '=', uid)])
                        pm3m_formation_e = row.key_value_pairs['formation_energy']

                        for term_type_id, term_type in enumerate(termination_types[orientation]):
                            uid = system_name + '3_' + str(orientation) + "_" + str(term_type) + "_3"
                            try:
                                row = db.get(selection=[('uid', '=', uid)])
                                twod_formation_e = row.key_value_pairs['formation_energy']
                                twod_e_diff_3.append(twod_formation_e - pm3m_formation_e)
                            except KeyError:
                                print('WARNING: No value for ' + uid + ' Skip!')
                                continue

                            uid = system_name + '3_' + str(orientation) + "_" + str(term_type) + "_9"
                            try:
                                row = db.get(selection=[('uid', '=', uid)])
                                twod_formation_e = row.key_value_pairs['formation_energy']
                                twod_e_diff_9.append(twod_formation_e - pm3m_formation_e)
                            except KeyError:
                                print('WARNING: No value for ' + uid + ' Skip!')
                                continue
            twod_e_diff_3 = [e for e in twod_e_diff_3 if (e < 2.0) and (e > -1.1)]
            twod_e_diff_9 = [e for e in twod_e_diff_9 if (e < 2.0) and (e > -1.1)]

            plt.subplot(3, 4, index)
            x_grid = np.linspace(-1.1, 2.0, 400)
            pdf = kde_scipy(np.array(twod_e_diff_3), x_grid, bandwidth=0.05)
            plt.plot(x_grid, pdf, 'k-')
            plt.fill_between(x_grid, pdf, color='#F1BA46', alpha=0.6)

            pdf = kde_scipy(np.array(twod_e_diff_9), x_grid, bandwidth=0.05)
            plt.plot(x_grid, pdf, 'k-')
            plt.fill_between(x_grid, pdf, color='#D72F01', alpha=0.6)

            if index == 1:
                legend_elements = [
                    Patch(facecolor='#F1BA46', edgecolor='k', label='$n=3$'),
                    Patch(facecolor='#D72F01', edgecolor='k', label='$n=9$')
                ]
                plt.legend(handles=legend_elements)
            if index == 1:
                plt.ylabel('[100] // Frequency')
            if index == 5:
                plt.ylabel('[110] // Frequency')
            if index == 9:
                plt.ylabel('[111] // Frequency')
            if index in [9, 10, 11, 12]:
                plt.xlabel("$E_f^{2D}-E_{f}^{Pm\\bar{3}m}$ (eV/atom)")
            if index not in [9, 10, 11, 12]:
                plt.xticks(color='w')
            if index == 1:
                plt.title('$A^{I}B^{II}_{M}X_{3}$')
            if index == 2:
                plt.title('$A^{I}B^{II}_{TM}X_{3}$')
            if index == 3:
                plt.title('$A^{II}B^{IV}C_{3}$')
            if index == 4:
                plt.title('$A^{I}B^{X}C_{3}$')
            plt.axvline(x=0.0, c='k')
    plt.tight_layout()
    plt.savefig(output)


def plot_energy_versus_bulk(db, orientation='100', thick=3, output=None):
    """
    Correlations between the formation energies (Ef) for bulk cubic perovskites (given by the minimum energy difference
    with respect to the fully relaxed bulk structure of the lowest Ef) and Ef for mono-unit-cell-thick (by default),
    (given with respect to Ef for bulk cubic perovskites). The data is separately shown for each one of the four types
    of perovskites investigated here.

    For each orientation, there are two different termination type, which will all be plotted.
    """
    figs, ((ax11, ax12, ax13, ax14), (ax21, ax22, ax23, ax24)) = plt.subplots(ncols=4, nrows=2, figsize=(20, 12))

    for i in range(len(A_site_list)):

        bulk_e_diff = [[] for _ in range(len(termination_types[orientation]))]
        twod_e_diff = [[] for _ in range(len(termination_types[orientation]))]
        names = [[] for _ in range(len(termination_types[orientation]))]

        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:

                    # Energy of the bulk material in the perovskite structure
                    system_name = a + b + c
                    uid = system_name + '3_pm3m'
                    row = db.get(selection=[('uid', '=', uid)])
                    pm3m_formation_e = row.key_value_pairs['formation_energy']

                    # Find the lowest energy amongst all the relaxed random structures
                    lowest_relaxed = -100000
                    for k in range(10):
                        uid = system_name + '3_random_str_' + str(k + 1)
                        try:
                            row = db.get(selection=[('uid', '=', uid)])
                            randomised_formation_e = row.key_value_pairs['formation_energy']
                            if (randomised_formation_e > lowest_relaxed) and (
                                    (pm3m_formation_e - randomised_formation_e) > -1.0):
                                lowest_relaxed = randomised_formation_e
                        except KeyError:
                            continue

                    # Find the formation energy of the corresponding two-D structures
                    for term_type_id, term_type in enumerate(termination_types[orientation]):
                        uid = system_name + '3_' + str(orientation) + "_" + str(term_type) + "_" + str(thick)
                        try:
                            row = db.get(selection=[('uid', '=', uid)])
                            twod_formation_e = row.key_value_pairs['formation_energy']

                            if twod_formation_e - pm3m_formation_e <= 10.0:
                                bulk_e_diff[term_type_id].append(pm3m_formation_e - lowest_relaxed)
                                twod_e_diff[term_type_id].append(twod_formation_e - pm3m_formation_e)
                                names[term_type_id].append(system_name + '$_{3}$')
                        except KeyError:
                            print('WARNING: No value for ' + uid + ' Skip!')
                            continue
        ylabel = None
        if i == 0:
            axs = [ax11, ax21]
            color = '#7AC7A9'
            title = '$A^{I}B^{II}_{M}X_{3}$'
            ylabel = '$\\min[E_{f}^{Pm\\bar{3}m}-E_{f}^{\mbox{\\Large{full relax}}}]$ (eV/atom)'
        if i == 1:
            axs = [ax12, ax22]
            color = '#90CA57'
            title = '$A^{I}B^{II}_{TM}X_{3}$'
        if i == 2:
            axs = [ax13, ax23]
            color = '#F1D628'
            title = '$A^{II}B^{IV}C_{3}$'
        if i == 3:
            axs = [ax14, ax24]
            color = '#2B8283'
            title = '$A^{I}B^{X}C_{3}$'

        for _p in range(len(axs)):
            _x = twod_e_diff[_p]
            _y = bulk_e_diff[_p]
            axs[_p].plot(_x, _y, 'o', c=color, alpha=0.9, ms=15)
            for u, _ in enumerate(_x):
                x = _x[u]
                y = _y[u]
                if (y > 0.2) or (y < -0.2):
                    axs[_p].text(x + 0.01, y + 0.01, names[_p][u], fontsize=14, color='#688291')

            # if max(_x)>10:
            #    axs[_p].set_xlim([min(_x) - 0.02, 3])
            # else:
            axs[_p].set_xlim([min(_x) - 0.02, max(_x) + 0.12])
            axs[_p].set_ylim([min(_y) - 0.02, max(_y) + 0.05])

            if ylabel is not None:
                axs[_p].set_ylabel(ylabel)
            axs[_p].annotate(str(termination_types[orientation][_p]) + '-termination', xy=(1, 0.9),
                             xycoords='axes fraction',
                             fontsize=16, horizontalalignment='right', verticalalignment='bottom')
        axs[0].set_title(title)
        axs[1].set_xlabel("$E_f^{2D,n=" + str(thick) + "}-E_{f}^{Pm\\bar{3}m}$ (eV/atom)")

    if output is None:
        output = 'two_d_' + str(orientation) + '_' + str(thick) + '_compare_bulk.png'
    plt.tight_layout()
    plt.savefig(output)
    plt.show()


def plot_super_cell_dependent_formation_energies(db, orientation='100', thick=3, output=None):
    figs, ((ax11), (ax21)) = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))

    for i in range(len(A_site_list)):

        small_cell_e_diff = [[] for _ in range(len(termination_types[orientation]))]
        large_cell_e_diff = [[] for _ in range(len(termination_types[orientation]))]

        if i == 0:
            color = '#7AC7A9'
        if i == 1:
            color = '#90CA57'
        if i == 2:
            color = '#F1D628'
        if i == 3:
            color = '#2B8283'

        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:
                    # Energy of the bulk material in the perovskite structure
                    system_name = a + b + c
                    uid = system_name + '3_pm3m'
                    row = db.get(selection=[('uid', '=', uid)])
                    pm3m_formation_e = row.key_value_pairs['formation_energy']

                    # Find the formation energy of the corresponding two-D structures
                    for term_type_id, term_type in enumerate(termination_types[orientation]):
                        uid_small = system_name + '3_' + str(orientation) + "_" + str(term_type) + "_" + str(thick)
                        uid_large = system_name + '3_' + str(orientation) + "_" + str(term_type) + "_" + str(
                            thick) + "_large_cell_full_B_octa"
                        twod_formation_e_small = None
                        twod_formation_e_large = None

                        try:
                            row = db.get(selection=[('uid', '=', uid_small)])
                            twod_formation_e_small = row.key_value_pairs['formation_energy']
                        except:
                            pass

                        try:
                            row = db.get(selection=[('uid', '=', uid_large)])
                            twod_formation_e_large = row.key_value_pairs['formation_energy']
                        except:
                            pass

                        if (twod_formation_e_large is not None) and (twod_formation_e_small is not None):
                            _a = twod_formation_e_small - pm3m_formation_e
                            _b = twod_formation_e_large - pm3m_formation_e
                            small_cell_e_diff[term_type_id].append(_a)
                            large_cell_e_diff[term_type_id].append(_b)
                            if abs(_a - _b) > 0.09:
                                from core.models.element import ionic_radii
                                from twodPV.analysis.bulk_energy_landscape import charge_state_A_site, \
                                    charge_state_B_site, charge_state_C_site
                                import math
                                tolerance_f = ionic_radii[a][charge_state_A_site[i]] + ionic_radii[c][
                                    charge_state_C_site[i]]
                                tolerance_f /= ionic_radii[b][charge_state_B_site[i]] + ionic_radii[c][
                                    charge_state_C_site[i]]
                                tolerance_f /= math.sqrt(2)
                                print(system_name, term_type, abs(_a - _b), tolerance_f)

        ax11.plot(small_cell_e_diff[0], large_cell_e_diff[0], 'o', c=color)
        ax11.plot(small_cell_e_diff[0], small_cell_e_diff[0], 'k-')
        ax11.set_xlabel(termination_types[orientation][0] + " (small cell) $E_{f}^{2D}-E_{f}^{Pm\\bar{2}m}$ (eV)")
        ax11.set_ylabel(termination_types[orientation][0] + " (large cell) $E_{f}^{2D}-E_{f}^{Pm\\bar{2}m}$ (eV)")

        ax21.plot(small_cell_e_diff[1], large_cell_e_diff[1], 'o', c=color)
        ax21.plot(small_cell_e_diff[1], small_cell_e_diff[1], 'k-')
        ax21.set_xlabel(termination_types[orientation][1] + " (small cell) $E_{f}^{2D}-E_{f}^{Pm\\bar{2}m}$ (eV)")
        ax21.set_ylabel(termination_types[orientation][1] + " (large cell) $E_{f}^{2D}-E_{f}^{Pm\\bar{2}m}$ (eV)")
    plt.tight_layout()
    plt.savefig(output)
    plt.show()


def plot_thickness_dependent_imaginary_frequency_distribution_summary(db, output=None):
    fig, axs = plt.subplots(2, 3, figsize=(24, 12))
    tf, tb = __frequency_histogram(db, '100', 'AO')
    print('[100]//AO-termination')
    __histogram_plotter_v2(axs[0, 0], tf, tb, '[100]//AO-termination')

    tf, tb = __frequency_histogram(db, '100', 'BO2')
    print('[100]//BO$_{2}$-termination')
    __histogram_plotter_v2(axs[1, 0], tf, tb, '[100]//BO$_{2}$-termination')

    tf, tb = __frequency_histogram(db, '110', 'ABO')
    print('[110]//ABO-termination')
    __histogram_plotter_v2(axs[0, 1], tf, tb, '[110]//ABO-termination')

    tf, tb = __frequency_histogram(db, '110', 'O2')
    print('[110]//O2-termination')
    __histogram_plotter_v2(axs[1, 1], tf, tb, '[110]//O$_{2}$-termination')

    tf, tb = __frequency_histogram(db, '111', 'AO3')
    print('[111]//AO3-termination')
    __histogram_plotter_v2(axs[0, 2], tf, tb, '[111]//AO$_{3}$-termination')

    tf, tb = __frequency_histogram(db, '111', 'B')
    print('[111]//B-termination')
    __histogram_plotter_v2(axs[1, 2], tf, tb, '[111]//B-termination')

    plt.tight_layout()
    plt.savefig(output)


def plot_thickness_dependent_imaginary_frequency_distribution_single_summary(db, output=None):
    import math
    plt.figure(figsize=(8, 6))
    thickness_freqs = {3: [], 5: [], 7: [], 9: []}
    thickness_bins = {3: [], 5: [], 7: [], 9: []}
    for thick in [3, 5, 7, 9]:
        this_freq = []
        for orientation in ['100', '110', '111']:
            for term_type in termination_types[orientation]:
                for i in range(len(A_site_list)):
                    for a in A_site_list[i]:
                        for b in B_site_list[i]:
                            for c in C_site_list[i]:
                                system_name = a + b + c
                                uid = system_name + '3_' + str(orientation) + "_" + str(term_type) + "_" + str(thick)
                                row = db.get(selection=[('uid', '=', uid)])
                                gamma_point_freqs = None
                                try:
                                    gamma_point_freqs = row.data['gamma_phonon_freq']

                                    # HERE ONLY TAKE IMAGINARY FREQUENCIES!
                                    # try:
                                    #    gamma_point_freqs = [(f ** 2).real for f in gamma_point_freqs if f.imag != 0.0]
                                    #    this_freq.append(min(gamma_point_freqs))
                                    # except ValueError:
                                    #    pass
                                except KeyError:
                                    # print("Phonon calculation failed for " + str(uid))
                                    pass
                                if gamma_point_freqs is not None:
                                    gamma_point_freqs = [(f ** 2).real for f in gamma_point_freqs]
                                    if not math.isnan(np.mean(gamma_point_freqs)):
                                        this_freq.append(np.mean(gamma_point_freqs))
        # this_freq = [f for f in this_freq if f > -5]
        hist, bin_edge = np.histogram(this_freq, bins=[-5 + 2 * j for j in range(103)], range=(-5, 200))
        thickness_freqs[thick] = list(hist)
        thickness_bins[thick] = list(bin_edge)[:-1]
    __histogram_plotter(thickness_freqs, thickness_bins)
    plt.savefig(output)


def plot_thickness_dependent_imaginary_frequency_distribution(db, orientation='100', output=None, term_type='AO'):
    plt.figure(figsize=(8, 6))

    thickness_freqs, thickness_bins = __frequency_histogram(db, orientation, term_type)
    if term_type == 'AO3':
        term = "AO$_{3}$"
    elif term_type == 'BO2':
        term = "BO$_{2}$"
    elif term_type == 'O2':
        term = "O$_{2}$"
    else:
        term = term_type
    __histogram_plotter(thickness_freqs, thickness_bins, title='[' + str(orientation) + ']//' + term + '-termination')
    plt.tight_layout()
    plt.savefig(output)


def __histogram_plotter(thickness_freqs, thickness_bins, title):
    color_dict = {3: '#A3586D', 5: '#5C4A72', 7: '#F3B05A', 9: '#F4874B'}
    plt.bar(thickness_bins[3], thickness_freqs[3], 0.1, alpha=0.7, label='$n=3$', color=color_dict[3])
    plt.bar(thickness_bins[5], thickness_freqs[5], 0.1, alpha=0.7, bottom=thickness_freqs[3], label='$n=5$',
            color=color_dict[5])
    plt.bar(thickness_bins[7], thickness_freqs[7], 0.1, alpha=0.7,
            bottom=[thickness_freqs[3][l] + thickness_freqs[5][l] for l in range(len(thickness_freqs[3]))],
            label='$n=7$', color=color_dict[7])
    plt.bar(thickness_bins[9], thickness_freqs[9], 0.1, alpha=0.7,
            bottom=[thickness_freqs[3][l] + thickness_freqs[5][l] + thickness_freqs[7][l] for l in
                    range(len(thickness_freqs[3]))], label='$n=9$', color=color_dict[9])
    # plt.xlabel("$\omega^2_{\min}$ (THz$^2$)")
    plt.xlabel("$\omega^2_{\min}$ (THz$^2$)$")
    plt.ylabel("Occurences")
    plt.title(title)
    plt.legend()


def __histogram_plotter_v2(ax, thickness_freqs, thickness_bins, title):
    color_dict = {3: '#A3586D', 5: '#5C4A72', 7: '#F3B05A', 9: '#F4874B'}
    ax.bar(thickness_bins[3], thickness_freqs[3], 0.09, alpha=0.7, label='$n=3$', color=color_dict[3])
    ax.bar(thickness_bins[5], thickness_freqs[5], 0.09, alpha=0.7, bottom=thickness_freqs[3], label='$n=5$',
           color=color_dict[5])
    ax.bar(thickness_bins[7], thickness_freqs[7], 0.09, alpha=0.7,
           bottom=[thickness_freqs[3][l] + thickness_freqs[5][l] for l in range(len(thickness_freqs[3]))],
           label='$n=7$', color=color_dict[7])
    ax.bar(thickness_bins[9], thickness_freqs[9], 0.09, alpha=0.7,
           bottom=[thickness_freqs[3][l] + thickness_freqs[5][l] + thickness_freqs[7][l] for l in
                   range(len(thickness_freqs[3]))], label='$n=9$', color=color_dict[9])
    ax.set_xlabel("$\omega^2_{\min}$ (THz$^2$)")
    ax.set_ylabel("Occurences")
    ax.set_title(title)
    ax.legend()


def __frequency_histogram(db, orientation=None, term_type=None):
    thickness_freqs = {3: [], 5: [], 7: [], 9: []}
    thickness_bins = {3: [], 5: [], 7: [], 9: []}

    for thick in [3, 5, 7, 9]:
        _this_freq = []
        for i in range(len(A_site_list)):
            for a in A_site_list[i]:
                for b in B_site_list[i]:
                    for c in C_site_list[i]:
                        system_name = a + b + c
                        uid = system_name + '3_' + str(orientation) + "_" + str(term_type) + "_" + str(thick)
                        row = db.get(selection=[('uid', '=', uid)])
                        try:
                            gamma_point_freqs = row.data['gamma_phonon_freq']
                            try:
                                gamma_point_freqs = [(f ** 2).real for f in gamma_point_freqs if f.imag != 0.0]
                                _this_freq.append(min(gamma_point_freqs))
                                if uid == 'BaTiO3_100_AO_3':
                                    print(uid, min(gamma_point_freqs))
                            except ValueError:
                                # print("No imaginary frequency for " + str(uid))
                                pass
                        except KeyError:
                            # print("Phonon calculation failed for " + str(uid))
                            pass
        this_freq = [f for f in _this_freq if f > -5]
        hist, bin_edge = np.histogram(this_freq, bins=[-5 + j * 0.1 for j in range(55)], range=(-5, 0))
        thickness_freqs[thick] = list(hist)
        thickness_bins[thick] = list(bin_edge)[:-1]
        print(hist, thickness_bins[thick])
        # print(len(thickness_bins[thick]), len(thickness_freqs[thick]))
        print(str(orientation), str(term_type), str(thick), len([f for f in this_freq if f >= -2]))
    return thickness_freqs, thickness_bins


def __thickness_dependent_imaginary_frequency_vs_energy(db, orientation='100', term_type='AO'):
    energy_thick = {3: [[] for _ in range(len(A_site_list))],
                    5: [[] for _ in range(len(A_site_list))],
                    7: [[] for _ in range(len(A_site_list))],
                    9: [[] for _ in range(len(A_site_list))]}

    freq_thick = {3: [[] for _ in range(len(A_site_list))],
                  5: [[] for _ in range(len(A_site_list))],
                  7: [[] for _ in range(len(A_site_list))],
                  9: [[] for _ in range(len(A_site_list))]}

    def most_frequent(List):
        return max(set(List), key=List.count)

    for i in range(len(A_site_list)):
        stable_chemical_space = [[], [], []]

        min_energy = 100
        max_freq = -50

        good_A = None
        good_B = None
        good_C = None
        good_thick = None

        for thick in [3, 5, 7, 9]:
            for a in A_site_list[i]:
                for b in B_site_list[i]:
                    for c in C_site_list[i]:
                        system_name = a + b + c
                        uid = system_name + '3_' + str(orientation) + "_" + str(term_type) + "_" + str(thick)
                        row = db.get(selection=[('uid', '=', uid)])
                        min_gamma_freq = None
                        try:
                            gamma_point_freqs = row.data['gamma_phonon_freq']
                            try:
                                gamma_point_freqs = [(f ** 2).real for f in gamma_point_freqs if f.imag != 0.0]
                                min_gamma_freq = min(gamma_point_freqs)
                            except ValueError:
                                # print("No imaginary frequency for " + str(uid))
                                pass
                        except KeyError:
                            # print("Phonon calculation failed for " + str(uid))
                            pass

                        two_d_formation_energy = row.key_value_pairs['formation_energy']

                        uid = system_name + '3_pm3m'
                        row = db.get(selection=[('uid', '=', uid)])
                        pm3m_formation_e = row.key_value_pairs['formation_energy']

                        if (pm3m_formation_e is not None) and (min_gamma_freq is not None) and (
                                two_d_formation_energy is not None):
                            energy_thick[thick][i].append(two_d_formation_energy - 0)
                            freq_thick[thick][i].append(min_gamma_freq)

                            if (two_d_formation_energy <= 0.2) and (min_gamma_freq >= -2):
                                stable_chemical_space[0].append(a)
                                stable_chemical_space[1].append(b)
                                stable_chemical_space[2].append(c)

                                if (two_d_formation_energy <= min_energy) and (min_gamma_freq >= max_freq):
                                    min_energy = two_d_formation_energy
                                    max_freq = min_gamma_freq
                                    good_A = a
                                    good_B = b
                                    good_C = c
                                    good_thick = thick

        print(most_frequent(stable_chemical_space[0]), most_frequent(stable_chemical_space[1]),
              most_frequent(stable_chemical_space[2]))
        print(good_A, good_B, good_C, good_thick, min_energy, max_freq)
    return energy_thick, freq_thick


def __freq_energy_plotter(ax, energy_thick, freq_thick, title, legend=False):
    color_dict = {3: '#A3586D', 5: '#5C4A72', 7: '#F3B05A', 9: '#F4874B'}
    chem_color_dict = {0: '#07000E', 1: '#D75404', 2: '#522E75', 3: '#D50B53'}

    set_y = False
    for i in [0, 1, 2, 3]:
        for thick in [3, 5, 7, 9]:
            ax.plot(freq_thick[thick][i], energy_thick[thick][i], 'o', color=chem_color_dict[i], alpha=0.5,
                    label="$n=$" + str(thick), ms=5 + 1.5 * thick)

            counter = 0
            for j in range(len(freq_thick[thick][i])):
                if (freq_thick[thick][i][j] >= -2) and (energy_thick[thick][i][j] <= 0.2):
                    counter += 1
            print(str(i) + ' ' + str(thick) + "  no. of stable structures " + str(counter))
        # if max(energy_thick[thick][i]) > 10:
        #    set_y = True

    ax.add_patch(patches.Rectangle((-2, -2.5), 2.3, 2.7, alpha=0.4, facecolor='#F1BA46'))

    ax.set_xlim([-5.3, 0.3])
    ax.set_ylim([-2.5, 1])
    ax.set_xlabel("$\omega^2_{\min}$ (THz$^2$)", fontsize=30)
    ax.set_ylabel("$E_f^{2D}$ (eV/atom)", fontsize=30)
    ax.set_title(title, fontsize=25)
    if legend:
        legend_elements = [Line2D([0], [0], marker='o', color='#D75404', label='$n=3$', markerfacecolor='#D75404',
                                  markersize=5 + 1.5 * 3, lw=0),
                           Line2D([0], [0], marker='o', color='#D75404', label='$n=5$', markerfacecolor='#D75404',
                                  markersize=5 + 1.5 * 5, lw=0),
                           Line2D([0], [0], marker='o', color='#D75404', label='$n=7$', markerfacecolor='#D75404',
                                  markersize=5 + 1.5 * 7, lw=0),
                           Line2D([0], [0], marker='o', color='#D75404', label='$n=9$', markerfacecolor='#D75404',
                                  markersize=5 + 1.5 * 9, lw=0),
                           Line2D([0], [0], marker='o', color='#07000E', label='$A^{I}B_{M}^{II}X_{3}$',
                                  markerfacecolor='#07000E', markersize=5 + 1.5 * 3, lw=0),
                           Line2D([0], [0], marker='o', color='#D75404', label='$A^{I}B_{TM}^{II}X_{3}$',
                                  markerfacecolor='#D75404', markersize=5 + 1.5 * 3, lw=0),
                           Line2D([0], [0], marker='o', color='#522E75', label='$A^{II}B^{IV}C_{3}$',
                                  markerfacecolor='#522E75', markersize=5 + 1.5 * 3, lw=0),
                           Line2D([0], [0], marker='o', color='#D50B53', label='$A^{I}B^{X}C_{3}$',
                                  markerfacecolor='#D50B53', markersize=5 + 1.5 * 3, lw=0)]
        ax.legend(handles=legend_elements, fontsize=19)


def plot_thickness_dependent_imaginary_frequency_energy_summary(db, output=None):
    fig, axs = plt.subplots(2, 3, figsize=(24, 12))
    et, ft = __thickness_dependent_imaginary_frequency_vs_energy(db, '100', 'AO')
    print('[100]//AX-termination')
    __freq_energy_plotter(axs[0, 0], et, ft, '[100]//AX-termination', legend=True)

    et, ft = __thickness_dependent_imaginary_frequency_vs_energy(db, '100', 'BO2')
    print('[100]//BO$_{2}$-termination')
    __freq_energy_plotter(axs[1, 0], et, ft, '[100]//BX$_{2}$-termination')

    et, ft = __thickness_dependent_imaginary_frequency_vs_energy(db, '110', 'ABO')
    print('[110]//ABO-termination')
    __freq_energy_plotter(axs[0, 1], et, ft, '[110]//ABX-termination')

    et, ft = __thickness_dependent_imaginary_frequency_vs_energy(db, '110', 'O2')
    print('[110]//O2-termination')
    __freq_energy_plotter(axs[1, 1], et, ft, '[110]//X$_{2}$-termination')

    et, ft = __thickness_dependent_imaginary_frequency_vs_energy(db, '111', 'AO3')
    print('[111]//AO3-termination')
    __freq_energy_plotter(axs[0, 2], et, ft, '[111]//AX$_{3}$-termination')

    et, ft = __thickness_dependent_imaginary_frequency_vs_energy(db, '111', 'B')
    print('[111]//B-termination')
    __freq_energy_plotter(axs[1, 2], et, ft, '[111]//B-termination')

    plt.tight_layout()
    plt.savefig(output)


def plot_thickness_dependent_imaginary_frequency_energy_single_summary(db, output=None):
    plt.figure(figsize=(8, 6))
    color_dict = {3: '#A3586D', 5: '#5C4A72', 7: '#F3B05A', 9: '#F4874B'}
    thickness_freqs = {3: [], 5: [], 7: [], 9: []}
    thickness_energies = {3: [], 5: [], 7: [], 9: []}
    for thick in [3, 5, 7, 9]:
        for orientation in ['100', '110', '111']:
            for term_type in termination_types[orientation]:
                for i in range(len(A_site_list)):
                    for a in A_site_list[i]:
                        for b in B_site_list[i]:
                            for c in C_site_list[i]:
                                min_gamma_freq = None
                                system_name = a + b + c
                                uid = system_name + '3_' + str(orientation) + "_" + str(term_type) + "_" + str(thick)
                                row = db.get(selection=[('uid', '=', uid)])
                                gamma_point_freqs = None
                                try:
                                    gamma_point_freqs = row.data['gamma_phonon_freq']
                                    # try:
                                    #    gamma_point_freqs = [(f ** 2).real for f in gamma_point_freqs if f.imag != 0.0]
                                    #    min_gamma_freq = min(gamma_point_freqs)
                                    # except ValueError:
                                    #    # print("No imaginary frequency for " + str(uid))
                                    #    pass
                                except KeyError:
                                    # print("Phonon calculation failed for " + str(uid))
                                    pass
                                if gamma_point_freqs is not None:
                                    gamma_point_freqs = [(f ** 2).real for f in gamma_point_freqs]
                                    min_gamma_freq = np.mean(gamma_point_freqs)

                                two_d_formation_energy = row.key_value_pairs['formation_energy']

                                uid = system_name + '3_pm3m'
                                row = db.get(selection=[('uid', '=', uid)])
                                pm3m_formation_e = row.key_value_pairs['formation_energy']

                                if (pm3m_formation_e is not None) and (min_gamma_freq is not None) and (
                                        two_d_formation_energy is not None):
                                    thickness_energies[thick].append(two_d_formation_energy)  # - pm3m_formation_e)
                                    thickness_freqs[thick].append(min_gamma_freq)
        plt.plot(thickness_freqs[thick], thickness_energies[thick], 'o', color=color_dict[thick], alpha=0.7,
                 label="$n=$" + str(thick), ms=5)

    plt.xlim([-5, 205])
    # plt.ylim([-1, 2])
    plt.ylim([-2.0, 0.5])
    # plt.xlabel("$\omega^2_{\min}$ (THz$^2$)")
    plt.xlabel("$\langle\omega^2\\rangle$ (THz$^2$)")
    # plt.ylabel("$E_f^{2D}-E_{f}^{Pm\\bar{3}m}$ (eV/atom)")
    plt.ylabel("$E_f^{2D}$ (eV/atom)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output)


def non_zero(tensor):
    return (tensor[0][0] != 0.0) and (tensor[1][1] != 0.0) and (tensor[2][2] != 0.0)


def trace(tensor):
    return (tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3.0


def trace_inplane(tensor):
    return (tensor[0][0] + tensor[1][1]) / 2.0


def trace_outplane(tensor):
    return tensor[2][2]


# def tolerance_factor_bulk_ionic_dielectric_compare(db):
#     all_tolerance_factors=[]
#     all_dielectric_consts=[]
#     for i in range(len(A_site_list)):
#         for a in A_site_list[i]:
#             for b in B_site_list[i]:
#                 for c in C_site_list[i]:
#                     system_name = a + b + c
#                     uid = system_name + '3_pm3m'
#                     row = db.get(selection=[('uid', '=', uid)])
#                     print(uid)
#                     tolerance_f = ionic_radii[a][charge_state_A_site[i]] + ionic_radii[c][charge_state_C_site[i]]
#                     tolerance_f /= ionic_radii[b][charge_state_B_site[i]] + ionic_radii[c][charge_state_C_site[i]]
#                     tolerance_f /= math.sqrt(2)
#
#                     bulk_dielectric_tensor = None
#                     try:
#                         bulk_dielectric_tensor = row.data['dielectric_ionic_tensor']
#                     except:
#                         continue
#
#                     if bulk_dielectric_tensor is not None:
#                         bulk_dielectric_constant = trace(bulk_dielectric_tensor)
#                         print(bulk_dielectric_constant)
#                         all_tolerance_factors.append(tolerance_f)
#                         all_dielectric_consts.append(bulk_dielectric_constant)
#
#     plt.scatter(all_tolerance_factors,all_dielectric_consts,marker='o',alpha=0.7)
#     plt.yscale('symlog')
#
#    plt.tight_layout()
#    plt.savefig('bulk_epsilon_tolerance.pdf')

def plot_bulk_and_2D_ionic_dielectric_constants(db, output=None, stable_structure_only=True):
    import scipy
    termination_types = {'100': ['AO', 'BO2'], '110': ['ABO', 'O2'], '111': ['AO3', 'B']}
    thicknesses = [3, 5, 7, 9]
    color_dict = {'100': '#F2A104', '110': '#00743F', '111': '#1D65A6'}

    def corrected_trace_inplane(sys_name):
        row = db.get(selection=[('uid', '=', sys_name)])
        structure_2d = row.toatoms()
        supercell_c = structure_2d.get_cell_lengths_and_angles()[2]

        structure = map_ase_atoms_to_crystal(structure_2d)
        all_z_positions = np.array([a.scaled_position.z for a in structure.asymmetric_unit[0].atoms])
        all_z_positions = all_z_positions - np.round(all_z_positions)
        all_z_positions = [z * supercell_c for z in all_z_positions]
        slab_thick = max(all_z_positions) - min(all_z_positions)
        twod_dielectric_tensor = row.data['dielectric_ionic_tensor']
        twod_epsilon_inplane = trace_inplane(twod_dielectric_tensor)
        twod_epsilon_inplane = (supercell_c / slab_thick) * (twod_epsilon_inplane - 1.0) + 1.0
        return twod_epsilon_inplane

    plt.figure(figsize=(14, 9))
    # get the bulk data

    epsilon_SrTiO = trace(db.get(selection=[('uid', '=', 'SrTiO3_pm3m')]).data['dielectric_ionic_tensor'])
    print("SrTiO",epsilon_SrTiO)
    epsilon_BaTiO = trace(db.get(selection=[('uid', '=', 'BaTiO3_pm3m')]).data['dielectric_ionic_tensor'])
    print("BaTiO",epsilon_BaTiO)
    epsilon_BaZrO = trace(db.get(selection=[('uid', '=', 'BaZrO3_pm3m')]).data['dielectric_ionic_tensor'])
    print("BaZrO",epsilon_BaZrO)
    epsilon_CsSnF = trace(db.get(selection=[('uid', '=', 'CsSnF3_pm3m')]).data['dielectric_ionic_tensor'])
    print("CsSnF", epsilon_CsSnF)

    for i in range(len(A_site_list)):
        for orient_id, orient in enumerate(['100', '110', '111']):
            for term_id, term in enumerate(termination_types[orient]):
                slot = orient_id + 1 + term_id * 3
                plt.subplot(2, 3, slot)

                for thick in thicknesses:

                    bulk_epsilons = []
                    twod_epsilons_inplane = []
                    twod_epsilons_outplane = []
                    sizes = []
                    for a in A_site_list[i]:
                        for b in B_site_list[i]:
                            for c in C_site_list[i]:
                                system_name = a + b + c
                                uid = system_name + '3_pm3m'
                                row = db.get(selection=[('uid', '=', uid)])
                                try:
                                    bulk_dielectric_tensor = row.data['dielectric_ionic_tensor']
                                except:
                                    continue

                                bulk_epsilon = None
                                if non_zero(bulk_dielectric_tensor):
                                    bulk_epsilon = trace(bulk_dielectric_tensor)

                                uid_2d = system_name + '3_' + str(orient) + "_" + str(term) + "_" + str(thick)
                                row_2d = db.get(selection=[('uid', '=', uid_2d)])

                                structure_2d = row_2d.toatoms()
                                supercell_c = structure_2d.get_cell_lengths_and_angles()[2]

                                structure = map_ase_atoms_to_crystal(structure_2d)
                                all_z_positions = np.array(
                                    [a.scaled_position.z for a in structure.asymmetric_unit[0].atoms])
                                all_z_positions = all_z_positions - np.round(all_z_positions)
                                all_z_positions = [z * supercell_c for z in all_z_positions]
                                slab_thick = max(all_z_positions) - min(all_z_positions)

                                try:
                                    twod_dielectric_tensor = row_2d.data['dielectric_ionic_tensor']
                                except:
                                    continue

                                skip_this = False
                                if stable_structure_only:
                                    # check if this structure is both energetically and structurally stable
                                    fe_2d = None
                                    try:
                                        fe_2d = row_2d.key_value_pairs['formation_energy']
                                    except:
                                        pass

                                    min_omega = None
                                    try:
                                        freq_2d = row_2d.data['gamma_phonon_freq']
                                    except:
                                        pass

                                    try:
                                        gamma_point_freqs = [(f ** 2).real for f in freq_2d if f.imag != 0.0]
                                        min_omega = min(gamma_point_freqs)
                                    except:
                                        pass

                                    if ((fe_2d is not None) and (fe_2d > 0.2)) or (
                                            (min_omega is not None) and (min_omega < -2)):
                                        skip_this = True

                                if not skip_this:
                                    twod_epsilon_inplane = None
                                    twod_epsilon_outplane = None
                                    if non_zero(twod_dielectric_tensor):
                                        twod_epsilon_inplane = trace_inplane(twod_dielectric_tensor)
                                        twod_epsilon_inplane = (supercell_c / slab_thick) * (
                                                    twod_epsilon_inplane - 1.0) + 1.0

                                        twod_epsilon_outplane = trace_outplane(twod_dielectric_tensor)
                                        twod_epsilon_outplane = 1.0 / ((supercell_c / slab_thick) * (
                                                    1.0 / twod_epsilon_outplane - 1.0) + 1.0)

                                    if (bulk_epsilon is not None) and (twod_epsilon_inplane is not None) and (
                                            twod_epsilon_outplane is not None):
                                        # print(uid_2d, twod_epsilon_inplane, twod_epsilon_outplane)
                                        bulk_epsilons.append(bulk_epsilon)
                                        twod_epsilons_inplane.append(twod_epsilon_inplane)
                                        twod_epsilons_outplane.append(twod_epsilon_outplane)
                                        sizes.append(thick * 7.5)

                    # plt.scatter(bulk_epsilons, twod_epsilons_outplane, marker='o', alpha=0.75, edgecolor=color_dict[orient],
                    #            facecolor='None', s=sizes)

                    plt.scatter(bulk_epsilons, twod_epsilons_inplane, marker='o', alpha=0.35, edgecolor='None',
                                facecolor=color_dict[orient], s=sizes)

                    print(orient, term, thick, i,
                          "{:.3f}".format(scipy.stats.pearsonr(bulk_epsilons, twod_epsilons_inplane)[0]))

                plt.xlabel('$\\varepsilon_{\\mbox{bulk}}$')
                plt.ylabel('$\\varepsilon_{\\mbox{2D}}$')

                plot_this = True

                if plot_this:
                    plt.plot([min(bulk_epsilons), max(bulk_epsilons)], [min(bulk_epsilons), max(bulk_epsilons)], 'k--',
                             lw=1.5)

                    plt.axvline(epsilon_BaTiO, ls=':', c='r')
                    plt.text(epsilon_BaTiO, 5e4, 'BaTiO$_3$', rotation=90, verticalalignment='center', c='r',
                             fontsize=13)
                    plt.axvline(epsilon_BaZrO, ls=':', c='r')
                    plt.text(epsilon_BaZrO, 5e4, 'BaZrO$_3$', rotation=90, verticalalignment='center', c='r',
                             fontsize=13)
                    plt.axvline(epsilon_SrTiO, ls=':', c='r')
                    plt.text(epsilon_SrTiO, 5e4, 'SrTiO$_3$', rotation=90, verticalalignment='center', c='r',
                             fontsize=13)
                    plt.axvline(epsilon_CsSnF, ls=':', c='r')
                    plt.text(epsilon_CsSnF, 5e4, 'CsSnF$_3$', rotation=90, verticalalignment='center', c='r', fontsize=13)

                    plt.ylim([0.1, 3e6])
                else:
                    plt.ylim([5e-5, 1e1])

                plt.xlim([0.1, 1e5])

                plt.xscale('log')
                plt.yscale('log')

                if slot == 1:
                    legend_elements = [Line2D([0], [0], marker='o', color=color_dict['100'], label='$n=3$', alpha=0.35,
                                              markerfacecolor=color_dict['100'],
                                              markersize=4, lw=0),
                                       Line2D([0], [0], marker='o', color=color_dict['100'], label='$n=5$', alpha=0.35,
                                              markerfacecolor=color_dict['100'],
                                              markersize=5, lw=0),
                                       Line2D([0], [0], marker='o', color=color_dict['100'], label='$n=7$', alpha=0.35,
                                              markerfacecolor=color_dict['100'],
                                              markersize=6, lw=0),
                                       Line2D([0], [0], marker='o', color=color_dict['100'], label='$n=9$', alpha=0.35,
                                              markerfacecolor=color_dict['100'],
                                              markersize=7, lw=0),
                                       # Line2D([0], [0], marker='o', color=color_dict['100'],
                                       #       label='$\\varepsilon_{2D}(xy)$', alpha=0.35,
                                       #       markerfacecolor=color_dict['100'],
                                       #       markersize=7, lw=0),
                                       # Line2D([0], [0], marker='o', color=color_dict['100'], label='$\\varepsilon_{2D}(z)$',
                                       #       alpha=0.35,
                                       #       markerfacecolor='None',
                                       #       markersize=7, lw=0)
                                       ]
                    plt.legend(handles=legend_elements, loc=4, fontsize=13, ncol=1)

                if slot == 1: textstr = "AX-termination"
                if slot == 2: textstr = 'ABX-termination'
                if slot == 3: textstr = 'AX$_{3}$-termination'
                if slot == 4: textstr = 'BX$_{2}$-termination'
                if slot == 5: textstr = 'X$_{2}$-termination'
                if slot == 6: textstr = 'B-termination'

                if plot_this:
                    if slot == 1:
                        plt.scatter([epsilon_BaTiO for _ in range(4)],
                                    [corrected_trace_inplane('BaTiO3_100_AO_' + str(t)) for t
                                     in thicknesses], s=[t * 7.5 for t in thicknesses],
                                    facecolor='None', edgecolor='r')
                        plt.scatter([epsilon_SrTiO for _ in range(4)],
                                    [corrected_trace_inplane('SrTiO3_100_AO_' + str(t)) for t
                                     in thicknesses], s=[t * 7.5 for t in thicknesses],
                                    facecolor='None', edgecolor='r')
                        plt.scatter([epsilon_BaZrO for _ in range(4)],
                                    [corrected_trace_inplane('BaZrO3_100_AO_' + str(t)) for t
                                     in thicknesses], s=[t * 7.5 for t in thicknesses],
                                    facecolor='None', edgecolor='r')
                        plt.scatter([epsilon_CsSnF for _ in range(4)], [corrected_trace_inplane('CsSnF3_100_AO_' + str(t)) for t
                                                                         in thicknesses], s=[t * 7.5 for t in thicknesses],
                                    facecolor='None', edgecolor='r')
                    if slot == 2:
                        plt.scatter([epsilon_BaTiO for _ in range(4)],
                                    [corrected_trace_inplane('BaTiO3_110_ABO_' + str(t)) for t
                                     in thicknesses], s=[t * 7.5 for t in thicknesses],
                                    facecolor='None', edgecolor='r')
                        plt.scatter([epsilon_SrTiO for _ in range(4)],
                                    [corrected_trace_inplane('SrTiO3_110_ABO_' + str(t)) for t
                                     in thicknesses], s=[t * 7.5 for t in thicknesses],
                                    facecolor='None', edgecolor='r')
                        plt.scatter([epsilon_BaZrO for _ in range(4)],
                                    [corrected_trace_inplane('BaZrO3_110_ABO_' + str(t)) for t
                                     in thicknesses], s=[t * 7.5 for t in thicknesses],
                                    facecolor='None', edgecolor='r')
                        plt.scatter([epsilon_CsSnF for _ in range(4)], [corrected_trace_inplane('CsSnF3_110_ABO_' + str(t)) for t
                            in thicknesses], s=[t * 7.5 for t in thicknesses],
                                    facecolor='None', edgecolor='r')
                    if slot == 3:
                        plt.scatter([epsilon_BaTiO for _ in range(4)],
                                    [corrected_trace_inplane('BaTiO3_111_AO3_' + str(t)) for t
                                     in thicknesses], s=[t * 7.5 for t in thicknesses],
                                    facecolor='None', edgecolor='r')
                        plt.scatter([epsilon_SrTiO for _ in range(4)],
                                    [corrected_trace_inplane('SrTiO3_111_AO3_' + str(t)) for t
                                     in thicknesses], s=[t * 7.5 for t in thicknesses],
                                    facecolor='None', edgecolor='r')
                        plt.scatter([epsilon_BaZrO for _ in range(4)],
                                    [corrected_trace_inplane('BaZrO3_111_AO3_' + str(t)) for t
                                     in thicknesses], s=[t * 7.5 for t in thicknesses],
                                    facecolor='None', edgecolor='r')
                        plt.scatter([epsilon_CsSnF for _ in range(4)], [corrected_trace_inplane('CsSnF3_111_AO3_' + str(t)) for t
                                                                         in thicknesses], s=[t * 7.5 for t in thicknesses],
                                    facecolor='None', edgecolor='r')
                    if slot == 4:
                        plt.scatter([epsilon_BaTiO for _ in range(4)],
                                    [corrected_trace_inplane('BaTiO3_100_BO2_' + str(t)) for t
                                     in thicknesses], s=[t * 7.5 for t in thicknesses],
                                    facecolor='None', edgecolor='r')
                        plt.scatter([epsilon_SrTiO for _ in range(4)],
                                    [corrected_trace_inplane('SrTiO3_100_BO2_' + str(t)) for t
                                     in thicknesses], s=[t * 7.5 for t in thicknesses],
                                    facecolor='None', edgecolor='r')
                        plt.scatter([epsilon_BaZrO for _ in range(4)],
                                    [corrected_trace_inplane('BaZrO3_100_BO2_' + str(t)) for t
                                     in thicknesses], s=[t * 7.5 for t in thicknesses],
                                    facecolor='None', edgecolor='r')

                        a = []
                        b = []
                        s = []
                        for t in thicknesses:
                            try:
                                e = corrected_trace_inplane('CsSnF3_100_BO2_' + str(t))
                                b.append(e)
                                a.append(epsilon_CsSnF)
                                s.append(t * 7.5)
                            except:
                                pass
                        plt.scatter(a, b, s=s, facecolor='None', edgecolor='r')

                    if slot == 5:
                        plt.scatter([epsilon_BaTiO for _ in range(4)],
                                    [corrected_trace_inplane('BaTiO3_110_O2_' + str(t)) for t
                                     in thicknesses], s=[t * 7.5 for t in thicknesses],
                                    facecolor='None', edgecolor='r')
                        plt.scatter([epsilon_SrTiO for _ in range(4)],
                                    [corrected_trace_inplane('SrTiO3_110_O2_' + str(t)) for t
                                     in thicknesses], s=[t * 7.5 for t in thicknesses],
                                    facecolor='None', edgecolor='r')
                        plt.scatter([epsilon_BaZrO for _ in range(4)],
                                    [corrected_trace_inplane('BaZrO3_110_O2_' + str(t)) for t
                                     in thicknesses], s=[t * 7.5 for t in thicknesses],
                                    facecolor='None', edgecolor='r')
                        plt.scatter([epsilon_CsSnF for _ in range(4)], [corrected_trace_inplane('CsSnF3_110_O2_' + str(t)) for t
                            in thicknesses], s=[t * 7.5 for t in thicknesses],
                                    facecolor='None', edgecolor='r')
                    if slot == 6:
                        plt.scatter([epsilon_BaTiO for _ in range(4)],
                                    [corrected_trace_inplane('BaTiO3_111_B_' + str(t)) for t
                                     in thicknesses], s=[t * 7.5 for t in thicknesses],
                                    facecolor='None', edgecolor='r')
                        plt.scatter([epsilon_SrTiO for _ in range(4)],
                                    [corrected_trace_inplane('SrTiO3_111_B_' + str(t)) for t
                                     in thicknesses], s=[t * 7.5 for t in thicknesses],
                                    facecolor='None', edgecolor='r')
                        plt.scatter([epsilon_BaZrO for _ in range(4)],
                                    [corrected_trace_inplane('BaZrO3_111_B_' + str(t)) for t
                                     in thicknesses], s=[t * 7.5 for t in thicknesses],
                                    facecolor='None', edgecolor='r')
                        plt.scatter([epsilon_CsSnF for _ in range(4)], [corrected_trace_inplane('CsSnF3_111_B_' + str(t)) for t
                            in thicknesses], s=[t * 7.5 for t in thicknesses],
                                    facecolor='None', edgecolor='r')

                plt.title(textstr)

    plt.tight_layout()
    plt.savefig(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Switches for analyzing the energy landscapes of 2D perovskites',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--orient", type=str, default='100',
                        help='Orientations of the two-d perovskite slabs')
    parser.add_argument("--terminations", type=str, default='AO',
                        help='Surface termination type of the two-d slab')
    parser.add_argument("--size_dependent_energies", action='store_true',
                        help="Plot size dependent formation energies.")
    parser.add_argument("--size_dependent_frequencies", action='store_true',
                        help="Plot size dependent gamma point frequencies.")
    parser.add_argument("--size_dependent_histogram", action='store_true',
                        help="Plot size dependent formation energies histogram.")
    parser.add_argument("--vbulk", action='store_true',
                        help="Plot energy compared to the bulk perovskite")
    parser.add_argument("--summary", action='store_true')
    parser.add_argument("--tolerance_factor", action='store_true')
    #    parser.add_argument("--supercell_effect", action='store_true',
    #                        help="Compare the formation energies calculated with different supercell size")
    parser.add_argument("--db", type=str, default=os.getcwd() + '/2dpv.db',
                        help="Name of the database that contains the results of the screenings.")
    parser.add_argument("--freq_distribute", action='store_true',
                        help='Distribution of imaginary phonon frequencies with respect to thickness.')
    parser.add_argument("--freq_energy_plot", action='store_true',
                        help='Distribution of imaginary phonon frequencies with respect to formation energy.')
    parser.add_argument("--ionic_dielectric", action='store_true',
                        help='Plot ionic dielectric constants, comparison between bulk and 2D')
    parser.add_argument("--bulk_ionic_dielectric", action='store_true')
    parser.add_argument("--output", type=str, default=None,
                        help='Output file name for figure generated.')

    args = parser.parse_args()

    if os.path.exists(args.db):
        args.db = connect(args.db)
    else:
        raise Exception("Database " + args.db + " does not exists, cannot proceed!")

    if args.vbulk:
        if not args.summary:
            plot_energy_versus_bulk(args.db, orientation=args.orient, output=args.output)
        if args.summary:
            plot_energy_versus_bulk_summary(args.db, output=args.output, tolerance_factor=args.tolerance_factor)

    if args.size_dependent_energies:
        plot_thickness_dependent_formation_energies(args.db, orientation=args.orient, output=args.output)

    if args.size_dependent_frequencies:
        plot_thickness_dependent_lowest_imaginary_frequency(args.db, orientation=args.orient, output=args.output)

    if args.freq_distribute:
        if not args.summary:
            plot_thickness_dependent_imaginary_frequency_distribution(args.db, orientation=args.orient,
                                                                      output=args.output,
                                                                      term_type=args.terminations)
        if args.summary:
            # plot_thickness_dependent_imaginary_frequency_distribution_summary(args.db, output=args.output)
            plot_thickness_dependent_imaginary_frequency_distribution_single_summary(args.db, output=args.output)

    if args.freq_energy_plot:
        if args.summary:
            plot_thickness_dependent_imaginary_frequency_energy_summary(args.db, output=args.output)
            # plot_thickness_dependent_imaginary_frequency_energy_single_summary(args.db, output=args.output)

    if args.size_dependent_histogram:
        plot_size_dependent_energy_histograms(args.db, output=args.output)

    if args.ionic_dielectric:
        plot_bulk_and_2D_ionic_dielectric_constants(args.db, output=args.output, stable_structure_only=False)

#    if args.supercell_effect:
#        plot_super_cell_dependent_formation_energies(args.db, orientation='100', output=args.output)

#    if args.bulk_ionic_dielectric:
#        tolerance_factor_bulk_ionic_dielectric_compare(args.db)