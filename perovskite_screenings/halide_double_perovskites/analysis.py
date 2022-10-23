import logging
import pickle

from ase.db import connect
import sqlite3
import json
import os
import math
import argparse

from matplotlib.lines import Line2D
from numpy import dot

from core.internal.builders.crystal import map_ase_atoms_to_crystal
from core.models import Crystal
from core.models.element import shannon_radii
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
from ase.geometry import *
from ase.neighborlist import *
from matplotlib.patches import Patch

rc('text', usetex=True)

import matplotlib.pylab as pylab

params = {'legend.fontsize': '14',
          'figure.figsize': (7.5, 6),
          'axes.labelsize': 24,
          'axes.titlesize': 24,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)

A_site = ['Li', 'Na', 'K', 'Rb', 'Cs']
X_site = ['F', 'Cl', 'Br', 'I']
M_site_mono = ['Pd', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Cu', 'Ag', 'Au', 'Hg', 'In', 'Tl']
M_site_tri = ['Pd', 'Ir', 'Pr', 'Rh', 'Ru', 'La', 'Mo', 'Nd', 'Ni', 'Nb', 'Lu', 'Ce', 'Mn', 'Co', 'Cr', 'Dy', 'Er',
              'Sc', 'Ta', "Tb", 'Eu', 'Y', 'Al', 'Gd', 'Ga', 'In', 'As', 'Sb', 'Bi', 'Fe', "Sb", "Sc", "Sm", "Ti",
              "Tl", "Tm", "V", "Y", 'Au']

M_site_mono_exclusive = [x for x in M_site_mono if x not in M_site_tri]
M_site_tri_exclusive = [x for x in M_site_tri if x not in M_site_mono]
M_site_variable = [x for x in M_site_tri if x in M_site_mono]

promising_pvs = ['K2InAgCl6','Rb2InAgCl6','Rb2InAgBr6','Cs2InAgCl6','Cs2InAgBr6','Rb2InAsCl6','Cs2InAsBr6','Rb2InBiCl6',
                 'Cs2InBiCl6','Cs2InBiBr6','Rb2InSbCl6','Rb2InSbBr6','Cs2InSbBr6','Rb2TlAsBr6','Cs2TlAsBr6','Cs2TlAsI6',
                 'Cs2TlBiI6','Cs2TlSbBr6','Cs2TlSbI6','Cs2AsAuCl6','Rb2SbAuCl6','K2ScAuI6','Rb2ScAuI6','Cs2ScAuI6','Cs2InGaI6',
                 'RbInBr3','CsInBr3','RbTlCl3','RbTlBr3','CsTlBr3']

def chemical_classifier(crystal: Crystal) -> dict:
    atom_dict = crystal.all_atoms_count_dictionaries()
    all_elements = list(atom_dict.keys())
    stochiometry = list(sorted([atom_dict[k] for k in all_elements]))
    chemical_dict = {'A_cation': None, 'M_cation_mono': None, 'M_cation_tri': None, 'X_anion': None}

    for e in all_elements:
        if e in X_site:
            chemical_dict['X_anion'] = e
            all_elements.remove(e)

    if (stochiometry == [1, 1, 3]) or (stochiometry == [2, 2, 6]):
        for e in all_elements:
            if (e in M_site_mono) and (e in M_site_tri):
                chemical_dict['M_cation_mono'] = e
                chemical_dict['M_cation_tri'] = e
                all_elements.remove(e)
        chemical_dict['A_cation'] = all_elements[-1]
        assert (None not in [chemical_dict[k] for k in chemical_dict.keys()])
    elif stochiometry == [1, 3, 6]:
        for e in all_elements:
            if (e in A_site) and (e in M_site_mono):
                chemical_dict['A_cation'] = e
                chemical_dict['M_cation_mono'] = e
                all_elements.remove(e)
        chemical_dict['M_cation_tri'] = all_elements[-1]
        assert (None not in [chemical_dict[k] for k in chemical_dict.keys()])
    elif stochiometry == [1, 1, 2, 6]:
        for e in all_elements:
            if (e in A_site) and (atom_dict[e] == 2):
                chemical_dict['A_cation'] = e
                all_elements.remove(e)

        M_site_elements = all_elements

        for e in M_site_elements:
            if e in M_site_mono_exclusive:
                chemical_dict['M_cation_mono'] = e
                M_site_elements.remove(e)
                if len(M_site_elements) == 1:
                    chemical_dict['M_cation_tri'] = M_site_elements[-1]
            elif e in M_site_tri_exclusive:
                chemical_dict['M_cation_tri'] = e
                M_site_elements.remove(e)
                if len(M_site_elements) == 1:
                    chemical_dict['M_cation_mono'] = M_site_elements[-1]
        if len(M_site_elements) == 2:
            if all([m in M_site_variable for m in M_site_elements]):
                # cannot really tell which one in which charge state, randomly assign one
                print('variable valence, randomly assigned')
                if 'Pd' not in M_site_elements:
                    chemical_dict['M_cation_mono'] = M_site_elements[0]
                    chemical_dict['M_cation_tri'] = M_site_elements[1]
                else:
                    chemical_dict['M_cation_tri'] = 'Pd'
                    M_site_elements.remove('Pd')
                    chemical_dict['M_cation_mono'] = M_site_elements[-1]
                M_site_elements = []
        assert (None not in [chemical_dict[k] for k in chemical_dict.keys()])

    return chemical_dict


def geometric_fingerprint(crystal: Crystal):
    chemistry = chemical_classifier(crystal)
    r_a = shannon_radii[chemistry['A_cation']]["1"]["VI"]['r_ionic']
    r_m = shannon_radii[chemistry['M_cation_mono']]["1"]["VI"]['r_ionic']
    r_mp = shannon_radii[chemistry['M_cation_tri']]["3"]["VI"]['r_ionic']
    r_x = shannon_radii[chemistry['X_anion']]["-1"]["VI"]['r_ionic']
    return chemistry, octahedral_factor(r_m, r_mp, r_x), octahedral_mismatch(r_m, r_mp,
                                                                             r_x), generalised_tolerance_factor(r_a,
                                                                                                                r_m,
                                                                                                                r_mp,
                                                                                                                r_x)


def octahedral_factor(r_m, r_mp, r_x):
    return (r_m + r_mp) / (2.0 * r_x)


def octahedral_mismatch(r_m, r_mp, r_x):
    return abs(r_m - r_mp) / (2.0 * r_x)


def generalised_tolerance_factor(r_a, r_m, r_mp, r_x):
    nominator = r_a + r_x
    denominator = (r_m + r_mp) / 2.0 + r_x
    denominator = denominator ** 2 + (r_m - r_mp) ** 2 / 4
    denominator = math.sqrt(denominator) * math.sqrt(2)
    return nominator / denominator


def formation_energy_landscape(db, uids, switch='sigma', plot_anion='F'):
    F_data_dict = {'formation_energies': [], 'octahedral_factors': [], 'octahedral_mismatch': [],
                   'tolerance_factors': [],
                   'A_site_cation': [], 'sigma': [], 'system': []}
    Cl_data_dict = {'formation_energies': [], 'octahedral_factors': [], 'octahedral_mismatch': [],
                    'tolerance_factors': [],
                    'A_site_cation': [], 'sigma': [], 'system': []}
    Br_data_dict = {'formation_energies': [], 'octahedral_factors': [], 'octahedral_mismatch': [],
                    'tolerance_factors': [],
                    'A_site_cation': [], 'sigma': [], 'system': []}
    I_data_dict = {'formation_energies': [], 'octahedral_factors': [], 'octahedral_mismatch': [],
                   'tolerance_factors': [],
                   'A_site_cation': [], 'sigma': [], 'system': []}

    all_data_dict = {'formation_energies': [], 'octahedral_factors': [], 'octahedral_mismatch': [],
                   'tolerance_factors': [],
                   'A_site_cation': [], 'sigma': [], 'system': []}

    if switch in ['A-site']:
        from perovskite_screenings.analysis import halide_C, halide_B, halide_A, tolerance_factor
        from perovskite_screenings.analysis import octahedral_factor as pv_octahedral_factor

        pv_tolerance_f = []
        pv_octahedral_f = []
        for c in halide_C:
            for a in halide_A:
                for b in halide_B:
                    tolerance_f = tolerance_factor(a, b, c, type='goldschmidt')
                    octahedral_f = pv_octahedral_factor(b, c)
                    pv_tolerance_f.append(tolerance_f)
                    pv_octahedral_f.append(octahedral_f)

        plt.scatter(pv_octahedral_f, pv_tolerance_f, alpha=0.3, marker='+', s=20, label='ABX$_{3}$')

    min_energy = 100000
    max_energy = -100000
    min_sigma = 100000
    max_sigma = -100000
    min_oct_mismatch = 100000
    max_oct_mismatch = -100000
    colors = []
    for uid in uids:
        row = None
        formation_energy = None
        sigma = None
        try:
            row = db.get(selection=[('uid', '=', uid)])
        except:
            continue
        if row is not None:
            atoms = row.toatoms()
            crystal = map_ase_atoms_to_crystal(atoms)

            try:
                formation_energy = row.key_value_pairs['formation_energy']
                print('system ' + uid + ' Formation Energy ' + str(formation_energy) + ' eV/atom')
            except KeyError:
                continue

            try:
                sigma = row.key_value_pairs['sigma_300K_single']
                if sigma>=2:
                    sigma=None
                print('system ' + uid + ' sigma ' + str(sigma))
            except KeyError:
                continue

        if (formation_energy is not None) and (sigma is not None):
            chemistry, octahedral_factor, octahedral_mismatch, generalised_tolerance_factor = geometric_fingerprint(
                crystal)
            if octahedral_factor >= octahedral_mismatch + 1 - math.sqrt(2):
                __octahedral_mismatch = octahedral_mismatch
            else:
                __octahedral_mismatch = -1

            print(octahedral_factor, octahedral_mismatch, generalised_tolerance_factor)

            all_data_dict['system'].append(uid.replace('dpv_', ''))
            all_data_dict['formation_energies'].append(formation_energy)
            all_data_dict['sigma'].append(sigma)
            all_data_dict['octahedral_factors'].append(octahedral_factor)
            all_data_dict['octahedral_mismatch'].append(__octahedral_mismatch)
            all_data_dict['tolerance_factors'].append(generalised_tolerance_factor)
            all_data_dict["A_site_cation"].append(chemistry['A_cation'])

            if chemistry['X_anion'] == 'F':
                F_data_dict['system'].append(uid.replace('dpv_', ''))
                F_data_dict['formation_energies'].append(formation_energy)
                F_data_dict['sigma'].append(sigma)
                F_data_dict['octahedral_factors'].append(octahedral_factor)
                F_data_dict['octahedral_mismatch'].append(__octahedral_mismatch)
                F_data_dict['tolerance_factors'].append(generalised_tolerance_factor)
                F_data_dict["A_site_cation"].append(chemistry['A_cation'])
            elif chemistry['X_anion'] == 'Cl':
                Cl_data_dict['system'].append(uid.replace('dpv_', ''))
                Cl_data_dict['formation_energies'].append(formation_energy)
                Cl_data_dict['sigma'].append(sigma)
                Cl_data_dict['octahedral_factors'].append(octahedral_factor)
                Cl_data_dict['octahedral_mismatch'].append(__octahedral_mismatch)
                Cl_data_dict['tolerance_factors'].append(generalised_tolerance_factor)
                Cl_data_dict["A_site_cation"].append(chemistry['A_cation'])
            elif chemistry['X_anion'] == 'Br':
                Br_data_dict['system'].append(uid.replace('dpv_', ''))
                Br_data_dict['formation_energies'].append(formation_energy)
                Br_data_dict['sigma'].append(sigma)
                Br_data_dict['octahedral_factors'].append(octahedral_factor)
                Br_data_dict['octahedral_mismatch'].append(__octahedral_mismatch)
                Br_data_dict['tolerance_factors'].append(generalised_tolerance_factor)
                Br_data_dict["A_site_cation"].append(chemistry['A_cation'])
            elif chemistry['X_anion'] == 'I':
                I_data_dict['system'].append(uid.replace('dpv_', ''))
                I_data_dict['formation_energies'].append(formation_energy)
                I_data_dict['sigma'].append(sigma)
                I_data_dict['octahedral_factors'].append(octahedral_factor)
                I_data_dict['octahedral_mismatch'].append(__octahedral_mismatch)
                I_data_dict['tolerance_factors'].append(generalised_tolerance_factor)
                I_data_dict["A_site_cation"].append(chemistry['A_cation'])

            if formation_energy < min_energy:
                min_energy = formation_energy
            if formation_energy > max_energy:
                max_energy = formation_energy
            if octahedral_mismatch < min_oct_mismatch:
                min_oct_mismatch = octahedral_mismatch
            if octahedral_mismatch > max_oct_mismatch:
                max_oct_mismatch = octahedral_mismatch
            if sigma > max_sigma:
                max_sigma = sigma
            if sigma < min_sigma:
                sigma = min_sigma

    #for i, x in enumerate(X_site):
    #    if i == 0:
    #        marker = '^'
    #    if i == 1:
    #        marker = 's'
    #    if i == 2:
    #        marker = 'd'
    #    if i == 3:
    #        marker = 'p'
    if switch == 'formation_energy':
        plt.scatter(F_data_dict['octahedral_factors'], F_data_dict['tolerance_factors'], marker='^',
                    norm=mpl.colors.Normalize(vmin=min_energy * 1.1, vmax=max_energy * 1.1),
                    c=F_data_dict['formation_energies'], edgecolor=None, alpha=0.45, s=45,
                    cmap=plt.get_cmap('RdYlGn'), label='X=F')
        plt.scatter(Cl_data_dict['octahedral_factors'], Cl_data_dict['tolerance_factors'], marker='s',
                    norm=mpl.colors.Normalize(vmin=min_energy * 1.1, vmax=max_energy * 1.1),
                    c=Cl_data_dict['formation_energies'], edgecolor=None, alpha=0.45, s=45,
                    cmap=plt.get_cmap('RdYlGn'), label='X=Cl')
        plt.scatter(Br_data_dict['octahedral_factors'], Br_data_dict['tolerance_factors'], marker='d',
                    norm=mpl.colors.Normalize(vmin=min_energy * 1.1, vmax=max_energy * 1.1),
                    c=Br_data_dict['formation_energies'], edgecolor=None, alpha=0.45, s=45,
                    cmap=plt.get_cmap('RdYlGn'), label='X=Br')
        plt.scatter(I_data_dict['octahedral_factors'], I_data_dict['tolerance_factors'], marker='p',
                    norm=mpl.colors.Normalize(vmin=min_energy * 1.1, vmax=max_energy * 1.1),
                    c=I_data_dict['formation_energies'], edgecolor=None, alpha=0.45, s=45,
                    cmap=plt.get_cmap('RdYlGn'), label='X=I')
    elif switch == 'octahedral_mismatch':
        plt.scatter(F_data_dict['octahedral_factors'], F_data_dict['tolerance_factors'], marker='^',
                    norm=mpl.colors.Normalize(vmin=0.0, vmax=0.5),
                    c=F_data_dict['octahedral_mismatch'], edgecolor=None, alpha=0.45, s=45,
                    cmap=plt.get_cmap('RdYlGn'), label='X=F')
        plt.scatter(Cl_data_dict['octahedral_factors'], Cl_data_dict['tolerance_factors'], marker='s',
                    norm=mpl.colors.Normalize(vmin=0.0, vmax=0.5),
                    c=Cl_data_dict['octahedral_mismatch'], edgecolor=None, alpha=0.45, s=45,
                    cmap=plt.get_cmap('RdYlGn'), label='X=Cl')
        plt.scatter(Br_data_dict['octahedral_factors'], Br_data_dict['tolerance_factors'], marker='d',
                    norm=mpl.colors.Normalize(vmin=0.0, vmax=0.5),
                    c=Br_data_dict['octahedral_mismatch'], edgecolor=None, alpha=0.45, s=45,
                    cmap=plt.get_cmap('RdYlGn'), label='X=Br')
        plt.scatter(I_data_dict['octahedral_factors'], I_data_dict['tolerance_factors'], marker='p',
                    norm=mpl.colors.Normalize(vmin=0.0, vmax=0.5),
                    c=I_data_dict['octahedral_mismatch'], edgecolor=None, alpha=0.45, s=45,
                    cmap=plt.get_cmap('RdYlGn'), label='X=I')
    elif switch == 'sigma':
        if (plot_anion is not None) and (plot_anion=="F"):
            plt.scatter(F_data_dict['octahedral_factors'], F_data_dict['tolerance_factors'], marker='^',
                        norm=mpl.colors.Normalize(vmin=0.0, vmax=2.1),
                        c=F_data_dict['sigma'], edgecolor=None, alpha=0.75, s=45,
                        cmap=plt.get_cmap('Blues_r'), label='X=F')
        elif (plot_anion is None):
            plt.scatter(F_data_dict['octahedral_factors'], F_data_dict['tolerance_factors'], marker='^',
                        norm=mpl.colors.Normalize(vmin=0.0, vmax=2.1),
                        c=F_data_dict['sigma'], edgecolor=None, alpha=0.75, s=45,
                        cmap=plt.get_cmap('RdBu'), label='X=F')

        if (plot_anion is not None) and (plot_anion == "Cl"):
            plt.scatter(Cl_data_dict['octahedral_factors'], Cl_data_dict['tolerance_factors'], marker='s',
                        norm=mpl.colors.Normalize(vmin=0.0, vmax=2.1),
                        c=Cl_data_dict['sigma'], edgecolor=None, alpha=0.75, s=45,
                        cmap=plt.get_cmap('Reds_r'), label='X=Cl')
        elif (plot_anion is None):
            plt.scatter(Cl_data_dict['octahedral_factors'], Cl_data_dict['tolerance_factors'], marker='s',
                        norm=mpl.colors.Normalize(vmin=0.0, vmax=2.1),
                        c=Cl_data_dict['sigma'], edgecolor=None, alpha=0.75, s=45,
                        cmap=plt.get_cmap('RdBu'), label='X=Cl')

        if (plot_anion is not None) and (plot_anion == "Br"):
            plt.scatter(Br_data_dict['octahedral_factors'], Br_data_dict['tolerance_factors'], marker='d',
                        norm=mpl.colors.Normalize(vmin=0.0, vmax=2.1),
                        c=Br_data_dict['sigma'], edgecolor=None, alpha=0.75, s=45,
                        cmap=plt.get_cmap('YlOrBr_r'), label='X=Br')
        elif (plot_anion is None):
            plt.scatter(Br_data_dict['octahedral_factors'], Br_data_dict['tolerance_factors'], marker='d',
                        norm=mpl.colors.Normalize(vmin=0.0, vmax=2.1),
                        c=Br_data_dict['sigma'], edgecolor=None, alpha=0.75, s=45,
                        cmap=plt.get_cmap('RdBu'), label='X=Br')

        if (plot_anion is not None) and (plot_anion == "I"):
            plt.scatter(I_data_dict['octahedral_factors'], I_data_dict['tolerance_factors'], marker='p',
                        norm=mpl.colors.Normalize(vmin=0.0, vmax=2.1),
                        c=I_data_dict['sigma'], edgecolor=None, alpha=0.75, s=45,
                        cmap=plt.get_cmap('Greens_r'), label='X=I')
        elif (plot_anion is None):
            plt.scatter(I_data_dict['octahedral_factors'], I_data_dict['tolerance_factors'], marker='p',
                        norm=mpl.colors.Normalize(vmin=0.0, vmax=2.1),
                        c=I_data_dict['sigma'], edgecolor=None, alpha=0.75, s=45,
                        cmap=plt.get_cmap('RdBu'), label='X=I')
    elif switch == 'A-site':
            colors = []
            for a in all_data_dict['A_site_cation']:
                if a == 'Li':
                    colors.append('#344d90')
                elif a == 'Na':
                    colors.append('#5cc5ef')
                elif a == 'K':
                    colors.append("#ffb745")
                elif a == 'Rb':
                    colors.append("#ffbebd")
                elif a == 'Cs':
                    colors.append("#CB0000")
            plt.scatter(all_data_dict['octahedral_factors'], all_data_dict['tolerance_factors'], marker='s',
                        c=colors, edgecolor=None, alpha=0.25, s=25)


    def f1(x):
        return (x + 1) - x  # stretch limit

    def f2(x):
        return (0.44 * x + 1.37) / (math.sqrt(2) * (x + 1))

    def f3(x):
        return (0.73 * x + 1.13) / (math.sqrt(2) * (x + 1))

    def f4(x):
        return 2.46 / np.sqrt(2 * (x + 1) ** 2)

    t = np.arange(0.1, 1.3, 0.05)

    y1 = f1(np.arange(math.sqrt(2) - 1, 0.77, 0.01))
    y2 = f2(np.arange(math.sqrt(2) - 1, 0.8, 0.01))
    y3 = f3(np.arange(0.8, 1.14, 0.01))
    y4 = f4(np.arange(0.73, 1.14, 0.01))
    plt.plot(np.arange(math.sqrt(2) - 1, 0.77, 0.01), y1, 'k--')
    plt.plot(np.arange(math.sqrt(2) - 1, 0.8, 0.01), y2, 'k--')
    plt.plot(np.arange(0.8, 1.14, 0.01), y3, 'k--')
    plt.plot(np.arange(0.73, 1.14, 0.01), y4, 'k--')
    plt.vlines(x=math.sqrt(2) - 1, ymin=0.78, ymax=1, color='k', linestyles='--')
    plt.vlines(x=1.14, ymin=0.65, ymax=0.83, color='k', linestyles='--')

    plt.xlabel('Octahedral factors $(\\bar{\\mu})$')
    plt.ylabel('Tolerance factors $(t)$')

    if switch == 'formation_energy':
        plt.legend()
        plt.colorbar(label='$E_{f}$ (eV/atom)')
    elif switch == 'octahedral_mismatch':
        plt.legend()
        cbar = plt.colorbar(label='Octahedral mismatch $(\\Delta\\mu)$', extend='min')
    elif switch == 'sigma':
        plt.legend()
        cbar = plt.colorbar(label='$\\sigma^{(2)}$(300 K)')

    elif switch == 'A-site':
        legend_elements = [Patch(facecolor='#344d90', edgecolor='k', label='A=Li'),
                           Patch(facecolor='#5cc5ef', edgecolor='k', label='A=Na'),
                           Patch(facecolor="#ffb745", edgecolor='k', label='A=K'),
                           Patch(facecolor="#ffbebd", edgecolor='k', label='A=Rb'),
                           Patch(facecolor="#CB0000", edgecolor='k', label='A=Cs'),
                           Line2D([0], [0], marker='+', color='w', label='ABX$_3$', markersize=5, markeredgecolor='b',
                                  alpha=0.4)
                           ]
        plt.legend(handles=legend_elements, loc=1, fontsize=12, ncol=1)


    plt.tight_layout()
    if switch == 'A-site':
        name = "formation_energy_landscape_dpv_A_new.pdf"
    elif switch == 'formation_energy':
        name = "formation_energy_landscape_dpv_new.pdf"
    elif switch == 'octahedral_mismatch':
        name = "formation_energy_landscape_dpv_delta_mu_new.pdf"
    elif switch == 'sigma':
        name = "sigma_geometry_landscape_new.pdf"
    plt.savefig(name)


def get_band_gap_energy_dict():
    band_gap_dict={}
    from pymatgen import MPRester
    from default_settings import MPRest_key
    mpr = MPRester(MPRest_key)
    response = mpr.session.get("https://materialsproject.org/materials/10.17188/1476059")
    text = response.text
    for t in text.split():
        if ('mp-' in t):
            mp_id = t.split('"')[1].split("/")[-1]
            try:
                data = mpr.get_data(mp_id)
            except:
                pass

            system_name = 'dpv_' + data[0]['pretty_formula']
            try:
                bandgap = data[0]['band_gap']
                print(system_name, mp_id,bandgap)
                band_gap_dict[system_name] = bandgap
            except:
                pass
            #print(data)
    import pickle
    pickle.dump(band_gap_dict,open('bandgap.p','wb'))

from scipy.stats import gaussian_kde
def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    # Kernel Density Estimation with Scipy
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)

def sigma_landscape(db, uids, x='frequency',y=None):
    formation_energy_dict = {'F': [], 'Cl': [], 'Br': [], 'I': []}
    sigma_dict = {'F': [], 'Cl': [], 'Br': [], 'I': []}
    freq_dict = {'F': [], 'Cl': [], 'Br': [], 'I': []}
    color_dict = {'F': '#061283', 'Cl': '#FD3C3C', 'Br': '#FFB74C', 'I': '#138D90'}

    if x=='band_gap':
        band_gap_data = pickle.load(open('bandgap.p','rb'))
        band_gap_dict = {'F': [], 'Cl': [], 'Br': [], 'I': []}
        promising_sigma=[]
        promising_Eg=[]

    for uid in uids:
        row = None
        formation_energy = None
        sigma = None
        frequency = None
        band_gap = None

        try:
            row = db.get(selection=[('uid', '=', uid)])
        except:
            pass
        if row is not None:
            atoms = row.toatoms()
            crystal = map_ase_atoms_to_crystal(atoms)
            chemistry = chemical_classifier(crystal)

            if x == 'band_gap':
                try:
                    band_gap = band_gap_data[uid]
                except KeyError:
                    pass

            try:
                formation_energy = row.key_value_pairs['formation_energy']
            except KeyError:
                pass

            try:
                sigma = row.key_value_pairs['sigma_300K_single']
            except KeyError:
                pass

            try:
                frequency = row.key_value_pairs['sigma_mode_averaged_300K']
            except KeyError:
                pass

            #print('system ' + uid + ' Formation Energy ' + str(formation_energy) + ' eV/atom; Sigma ' + str(sigma))
            if x=='formation_energies':
                if (formation_energy is not None) and (sigma is not None) and (str(sigma) != 'nan'):
                    X = chemistry['X_anion']
                    formation_energy_dict[X].append(formation_energy)
                    sigma_dict[X].append(sigma)
            elif x=='frequency':
                if (frequency is not None) and (sigma is not None) and (str(frequency) != 'nan') and (formation_energy is not None):
                    X = chemistry['X_anion']
                    freq_dict[X].append(frequency)
                    sigma_dict[X].append(sigma)
                    formation_energy_dict[X].append(formation_energy)
            elif x=='band_gap':
                if (sigma is not None) and (band_gap is not None):
                    X = chemistry['X_anion']
                    band_gap_dict[X].append(band_gap)
                    sigma_dict[X].append(sigma)

                    if uid in ['dpv_'+l for l in promising_pvs]:
                        promising_sigma.append(sigma)
                        promising_Eg.append(band_gap)
                        print(uid, sigma)

    if x == 'formation_energies':
        for k in formation_energy_dict.keys():
            plt.scatter(formation_energy_dict[k], sigma_dict[k], alpha=0.6, marker='o', s=25, edgecolor=None,
                        c=color_dict[k])

        legend_elements = [Patch(facecolor=color_dict['F'], edgecolor='k', label='X=F'),
                           Patch(facecolor=color_dict['Cl'], edgecolor='k', label='X=Cl'),
                           Patch(facecolor=color_dict['Br'], edgecolor='k', label='X=Br'),
                           Patch(facecolor=color_dict['I'], edgecolor='k', label='X=I')]
        plt.legend(handles=legend_elements, loc=1, fontsize=12, ncol=1)
        plt.axhline(y=1, color='k', linestyle='--')
        plt.ylim([0, 2])
        plt.xlabel('$\\Delta E_{f}$ (eV/atom)')
        plt.ylabel('$\\sigma^{(2)}$ (300 K)')
        plt.tight_layout()
        plt.savefig('sigma_Ef_landscape.pdf')

    elif  x=='frequency':

        fig = plt.figure(figsize=(6, 7))

        gs = fig.add_gridspec(2, 1, height_ratios=(1,3),
                              left=0.2, right=0.95, bottom=0.1, top=0.95,
                              wspace=0.05, hspace=0.1)

        ax = fig.add_subplot(gs[1])
        ax1 = fig.add_subplot(gs[0])

        if y is None:
            for k in freq_dict.keys():
                ax.scatter(freq_dict[k], sigma_dict[k], alpha=0.6, marker='o', s=25, edgecolor=None,
                            c=color_dict[k])
        if y == 'formation_energy':
            for k in freq_dict.keys():
                ax.scatter(freq_dict[k], formation_energy_dict[k], alpha=0.6, marker='o', s=25, edgecolor=None,
                            c=color_dict[k])

        legend_elements = [Patch(facecolor=color_dict['F'], edgecolor='k', label='X=F'),
                           Patch(facecolor=color_dict['Cl'], edgecolor='k', label='X=Cl'),
                           Patch(facecolor=color_dict['Br'], edgecolor='k', label='X=Br'),
                           Patch(facecolor=color_dict['I'], edgecolor='k', label='X=I')]
        ax.legend(handles=legend_elements, loc=1, fontsize=12, ncol=1)
        ax.set_xlabel('$\\langle\\omega\\rangle_{\\sigma}$ (THz)')
        ax.set_xlim([-1, 8])
        if y is None:
            ax.axhline(y=1, color='k', linestyle='--')
            ax.set_ylim([0, 2])

            ax.set_ylabel('$\\sigma^{(2)}$ (300 K)')
        if y == 'formation_energy':
            ax.set_ylabel('$E_f$ (eV/atom)')


        for c in ['F','Cl','Br','I']:
            x_grid = np.linspace(-1,8, 500)
            pdf = kde_scipy(np.array(freq_dict[c]), x_grid, bandwidth=0.1)
            ax1.plot( x_grid, pdf / sum(pdf), '-', lw=2, c=color_dict[c])
        ax1.set_xlim([-1,8])
        labels = [item.get_text() for item in ax.get_xticklabels()]
        empty_string_labels = [''] * len(labels)
        ax1.set_xticklabels(empty_string_labels)
        ax1.set_ylabel('$\\mathcal{P}(\\langle\\omega\\rangle_{\\sigma})$')

        plt.savefig('sigma_fe_landscape.pdf')
    elif x == 'band_gap':

        for k in formation_energy_dict.keys():
            plt.scatter(band_gap_dict[k], sigma_dict[k], alpha=0.5, marker='o', s=25, edgecolor=None,
                        c=color_dict[k])
        print(len(promising_Eg))
        plt.scatter(promising_Eg, promising_sigma, marker='s', s=30, edgecolor='k', c='k')
        legend_elements = [Patch(facecolor=color_dict['F'], edgecolor='k', label='X=F'),
                           Patch(facecolor=color_dict['Cl'], edgecolor='k', label='X=Cl'),
                           Patch(facecolor=color_dict['Br'], edgecolor='k', label='X=Br'),
                           Patch(facecolor=color_dict['I'], edgecolor='k', label='X=I')]
        plt.legend(handles=legend_elements, loc=1, fontsize=12, ncol=1)
        plt.axhline(y=1, color='k', linestyle='--')
        plt.ylim([0, 2])
        plt.xlabel('$E_{g}$ (eV)')
        plt.ylabel('$\\sigma^{(2)}$ (300 K)')
        plt.tight_layout()
        plt.savefig('sigma_Eg_landscape.pdf')


if __name__ == "__main__":
    dbname = os.path.join(os.getcwd(), 'double_halide_pv.db')

    # ====================================================================
    # this is a hack to get all the uids from the database
    all_uids = []
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
    # ====================================================================

    # use the ASE db interface
    db = connect(dbname)

    """
    for uid in all_uids:
        row = None
        sigma = None
        try:
            row = db.get(selection=[('uid', '=', uid)])
        except:
            pass
        if row is not None:
            try:
                sigma = row.key_value_pairs['sigma_300K_single']
                print('system ' + uid + ' sigma ' + str(sigma))
            except KeyError:
                pass
    """

    formation_energy_landscape(db, all_uids, switch='sigma',plot_anion='I')

    #sigma_landscape(db, all_uids, x='band_gap')

    #get_band_gap_energy_dict()

    #sigma_frequency_plot(db,all_uids)

