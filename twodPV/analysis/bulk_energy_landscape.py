from twodPV.bulk_library import A_site_list, B_site_list, C_site_list
from core.models.element import ionic_radii

from ase.db import connect
import os
import math
import argparse

from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

import matplotlib.pylab as pylab

params = {'legend.fontsize': '8',
          'figure.figsize': (6, 5),
          'axes.labelsize': 24,
          'axes.titlesize': 24,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)

charge_state_A_site = {0: 1, 1: 1, 2: 2, 3: 1}
charge_state_B_site = {0: 2, 1: 2, 2: 4, 3: 5}
charge_state_C_site = {0: -1, 1: -1, 2: -2, 3: -2}

color_dict = {0: '#A3586D', 1: '#5C4A72', 2: '#F3B05A', 3: '#F4874B'}

def bulk_energy_landscape():
    cwd = os.getcwd()
    db = connect(cwd + '/2dpv.db')

    energy_differences = []
    tolerance_factors = []
    colors = []
    en_diff_1 = []
    en_diff_2 = []
    en_diff_3 = []
    en_diff_4 = []
    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:

                    tolerance_f = ionic_radii[a][charge_state_A_site[i]] + ionic_radii[c][charge_state_C_site[i]]
                    tolerance_f /= ionic_radii[b][charge_state_B_site[i]] + ionic_radii[c][charge_state_C_site[i]]
                    tolerance_f /= math.sqrt(2)

                    system_name = a + b + c
                    uid = system_name + '3_pm3m'
                    row = db.get(selection=[('uid', '=', uid)])
                    pm3m_formation_e = row.key_value_pairs['formation_energy']

                    for k in range(10):
                        uid = system_name + '3_random_str_' + str(k + 1)
                        try:
                            row = db.get(selection=[('uid', '=', uid)])
                        except KeyError:
                            print(uid + '   failed')
                            continue
                            randomised_formation_e = row.key_value_pairs[   'formation_energy']

                        tolerance_factors.append(tolerance_f)

                        en_diff = pm3m_formation_e - randomised_formation_e
                        energy_differences.append(en_diff)

                        colors.append(color_dict[i])
                        if i == 0:
                            if en_diff > -1:
                                en_diff_1.append(en_diff)
                        if i == 1:
                            if en_diff > -1:
                                en_diff_2.append(en_diff)
                        if i == 2:
                            if en_diff > -1:
                                en_diff_3.append(en_diff)
                        if i == 3:
                            if en_diff > -1:
                                en_diff_4.append(en_diff)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_dict[0], edgecolor='k', label='$A^{I}B^{II}_{M}X_{3}$'),
                       Patch(facecolor=color_dict[1], edgecolor='k', label='$A^{I}B^{II}_{TM}X_{3}$'),
                       Patch(facecolor=color_dict[2], edgecolor='k', label='$A^{II}B^{IV}C_{3}$'),
                       Patch(facecolor=color_dict[3], edgecolor='k', label='$A^{I}B^{X}C_{3}$')]

    gs = gridspec.GridSpec(1, 5, width_ratios=[3.5, 1, 1, 1, 1])
    gs.update(wspace=0.025, hspace=0.05)
    fig = plt.subplots(figsize=(15, 6))
    ax = plt.subplot(gs[0])
    ax.scatter(tolerance_factors, energy_differences, marker='o', edgecolor='None', facecolor=colors, s=80, alpha=0.6)
    ax.set_ylim([-0.8, 0.8])
    ax.set_xlabel('Tolerance factor')
    ax.set_ylabel('$E_{f}^{Pm\\bar{3}m}-E_{f}^{\mbox{\\Large{full relax}}}$ (eV/atom)')

    from scipy.stats import gaussian_kde
    import numpy as np

    def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
        # Kernel Density Estimation with Scipy
        # Note that scipy weights its bandwidth by the covariance of the
        # input data.  To make the results comparable to the other methods,
        # we divide the bandwidth by the sample standard deviation here.
        kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
        return kde.evaluate(x_grid)

    ax.legend(handles=legend_elements, loc=3, fontsize=12, ncol=1)
    ax1 = plt.subplot(gs[1])
    ax1.hist(en_diff_1, bins=50, orientation="horizontal", color=color_dict[0], lw=0, alpha=0.7, density=1);

    x_grid = np.linspace(-0.8, 0.8, 1000)
    pdf = kde_scipy(np.array(en_diff_1), x_grid, bandwidth=0.05)
    ax1.plot(pdf, x_grid, '-', lw=1, c='k')

    labels = [item.get_text() for item in ax1.get_yticklabels()]
    empty_string_labels = [''] * len(labels)
    ax1.set_yticklabels(empty_string_labels)

    ax1.set_ylim([-0.8, 0.8])
    ax2 = plt.subplot(gs[2])
    ax2.hist(en_diff_2, bins=50, orientation="horizontal", color=color_dict[1], lw=0, alpha=0.7, density=1);
    ax2.set_ylim([-0.8, 0.8])

    x_grid = np.linspace(-0.8, 0.8, 1000)
    pdf = kde_scipy(np.array(en_diff_2), x_grid, bandwidth=0.05)
    ax2.plot(pdf, x_grid, '-', lw=1, c='k')

    labels = [item.get_text() for item in ax2.get_yticklabels()]
    empty_string_labels = [''] * len(labels)
    ax2.set_yticklabels(empty_string_labels)

    ax3 = plt.subplot(gs[3])
    ax3.hist(en_diff_3, bins=50, orientation="horizontal", color=color_dict[2], lw=0, alpha=0.7, density=1);
    ax3.set_ylim([-0.8, 0.8])

    x_grid = np.linspace(-0.8, 0.8, 1000)
    pdf = kde_scipy(np.array(en_diff_3), x_grid, bandwidth=0.05)
    ax3.plot(pdf, x_grid, '-', lw=1, c='k')

    labels = [item.get_text() for item in ax3.get_yticklabels()]
    empty_string_labels = [''] * len(labels)
    ax3.set_yticklabels(empty_string_labels)

    ax4 = plt.subplot(gs[4])
    ax4.hist(en_diff_4, bins=50, orientation="horizontal", color=color_dict[3], lw=0, alpha=0.5, density=1);
    ax4.set_ylim([-0.8, 0.8])

    x_grid = np.linspace(-0.8, 0.8, 1000)
    pdf = kde_scipy(np.array(en_diff_4), x_grid, bandwidth=0.05)
    ax4.plot(pdf, x_grid, '-', lw=1, c='k')

    labels = [item.get_text() for item in ax4.get_yticklabels()]
    empty_string_labels = [''] * len(labels)
    ax4.set_yticklabels(empty_string_labels)

    plt.tight_layout()
    plt.savefig('Pm3m_energy_landscape_2.pdf')
    plt.show()


def plot_phonon_freq_vs_ef(db):
    pm3m_formation_es = []
    lowest_eigen_gammas = []
    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:
                    system_name = a + b + c
                    uid = system_name + '3_pm3m'
                    row = db.get(selection=[('uid', '=', uid)])
                    pm3m_formation_e = row.key_value_pairs['formation_energy']
                    lowest_eigen = min(row.data['gamma_phonon_freq'])

                    pm3m_formation_es.append(pm3m_formation_e)
                    lowest_eigen_gammas.append(lowest_eigen)
    plt.plot(pm3m_formation_es, lowest_eigen_gammas, 'bo')
    plt.tight_layout()
    plt.savefig('Pm3m_energy_phonon_freq.png')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Switches for analyzing the energy landscapes of bulk perovskites',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--energy_landscape", action='store_true')
    parser.add_argument("--phonon", action='store_true')
    parser.add_argument("--db", type=str, default=os.getcwd() + '/2dpv.db',
                        help="Name of the database that contains the results of the screenings.")
    args = parser.parse_args()

    if os.path.exists(args.db):
        args.db = connect(args.db)
    else:
        raise Exception("Database " + args.db + " does not exists, cannot proceed!")

    if args.energy_landscape:
        bulk_energy_landscape()

    if args.phonon:
        plot_phonon_freq_vs_ef(args.db)
