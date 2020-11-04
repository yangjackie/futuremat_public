import argparse
import os
import sqlite3
import json
import math
import pickle

from ase.db import connect
from itertools import permutations
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import stats

rc('text', usetex=True)

import matplotlib.pylab as pylab

params = {'legend.fontsize': '10',
          'figure.figsize': (5, 4),
          'axes.labelsize': 20,
          'axes.titlesize': 28,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)

from core.internal.builders.crystal import map_ase_atoms_to_crystal

color_dictionary = {0: '#81715E', 2: '#FAAE3D', 3: '#E38533', 1: '#E4535E', 4: '#138D90', 5: '#061283'}
y_label_dict = {'demixing': "$\\Delta H_{\\mbox{demix}}$ (eV/atom)",
                'binary-decomp': "$\\Delta H_{\\mbox{binary-decomp}}$ (eV/atom)",
                'binary-ox': "$\\Delta H_{\\mbox{binary-ox}}$ (eV/atom)",
                'sn-ox': "$\\Delta H_{\\mbox{Sn-ox}}$ (eV/atom)",
                'snpb-rhm-ox': "$\\Delta H_{\\mbox{SnPb-rhm-ox}}$ (eV/atom)",
                'sn-pb-rhm-ox': "$\\Delta H_{\\mbox{Sn/Pb-rhm-ox}}$ (eV/atom)"}

label_dict = {0: "Cs(Pb$_{x}$Sn$_{1-x})$Cl$_{3}$",
              1: "Cs(Pb$_{x}$Sn$_{1-x})$Br$_{3}$",
              2: "Cs(Pb$_{x}$Sn$_{1-x})$I$_{3}$"}

label_dict = {0: "Cl",
              1: "Br",
              2: "I"}

def plot_all_reactions_for_one_system(db, output=None, all_keys=None, X='Cl3'):
    funcs = [composition_dependent_demixing_energies,
             composition_dependent_binary_halide_decomposition_energies,
             composition_dependent_binary_oxidation_energies,
             composition_dependent_sn_only_oxidation_energies,
             composition_dependent_sn_pb_rhm_oxidiation_energies]
    # composition_dependent_snpb_rhm_solution_oxidation_energies]

    reaction_label_dict = {0: "$\\Delta H_{\\mbox{demix}}$ ",
                           1: "$\\Delta H_{\\mbox{binary-decomp}}$ ",
                           2: "$\\Delta H_{\\mbox{binary-ox}}$ ",
                           3: "$\\Delta H_{\\mbox{Sn-ox}}$ ",
                           4: "$\\Delta H_{\\mbox{Sn/Pb-rhm-ox}}$ ",
                           5: "$\\Delta H_{\\mbox{SnPb-rhm-ox}}$ "}

    reaction_color_dictionary = {0: '#53900f', 1: '#a4a71e', 2: '#d6ce15', 3: '#f58231', 4: '#1F2605', 5: '#1F6521'}

    for counter, func in enumerate(funcs):
        for energy_dict in [func(a=['Cs'], b=['Pb', 'Sn'], c=[X], all_keys=all_keys, db=db)]:
            if func == composition_dependent_demixing_energies:
                reaction_energies = [0, 0]
                averaged_energies = [0, 0]
                max_energies = [0, 0]
                min_energies = [0, 0]
                compositions = [0, 1]
                av_compositions = [0, 1]
            else:
                reaction_energies = []
                averaged_energies = []
                max_energies = []
                min_energies = []
                compositions = []
                av_compositions = []

            for k in energy_dict.keys():
                for e in energy_dict[k]:
                    compositions.append(k)
                    reaction_energies.append(e)
                averaged_energies.append(np.average(energy_dict[k]))  # -np.average(energy_dict[zero]))
                av_compositions.append(k)
                max_energies.append(max(energy_dict[k]))  # -max(energy_dict[zero]))
                min_energies.append(min(energy_dict[k]))  # -min(energy_dict[zero]

            x = sorted(av_compositions)
            max_energies = [n for _, n in sorted(zip(av_compositions, max_energies))]
            min_energies = [n for _, n in sorted(zip(av_compositions, min_energies))]
            averaged_energies = [n for _, n in sorted(zip(av_compositions, averaged_energies))]

            if func == composition_dependent_demixing_energies:
                popt, pcov = curve_fit(bowing_curve, x, averaged_energies)
                averaged_energies = bowing_curve(np.array(x), *popt)

            plt.plot(x, averaged_energies, 'o-', c=reaction_color_dictionary[counter], lw=2.5,
                     label=reaction_label_dict[counter])

            plt.plot(x, [0 for _ in x], 'k:')
    if 'Cl' in X:
        plt.xlabel('$x$ in Cs(Pb$_{x}$Sn$_{1-x})$Cl$_{3}$')
    if 'Br' in X:
        plt.xlabel('$x$ in Cs(Pb$_{x}$Sn$_{1-x})$Br$_{3}$')
    if 'I' in X:
        plt.xlabel('$x$ in Cs(Pb$_{x}$Sn$_{1-x})$I$_{3}$')
    plt.legend(loc=4)
    plt.ylabel("$\\Delta H_{\\mbox{reaction}}$ (eV/atom)")
    plt.ylim([-0.6, 0.5])
    # plt.yticks([-0.6,-0.3,0,0.25,0.5],[-0.6,-0.3,0,0.25,0.5])
    # plt.yscale('symlog')
    plt.tight_layout()
    plt.savefig(output)


def plot_all_reaction_energies_for_system(db, reaction='demixing', output=None, all_keys=None):
    if reaction == 'demixing':
        func = composition_dependent_demixing_energies
    if reaction == 'binary-decomp':
        func = composition_dependent_binary_halide_decomposition_energies
    if reaction == 'binary-ox':
        func = composition_dependent_binary_oxidation_energies
    if reaction == 'sn-ox':
        func = composition_dependent_sn_only_oxidation_energies
    if reaction == 'sn-pb-rhm-ox':
        func = composition_dependent_sn_pb_rhm_oxidiation_energies
    if reaction == 'snpb-rhm-ox':
        func = composition_dependent_snpb_rhm_solution_oxidation_energies

    energy_dicts = [func(a=['Cs'], b=['Pb', 'Sn'], c=['Cl3'], all_keys=all_keys, db=db),
                    func(a=['Cs'], b=['Pb', 'Sn'], c=['Br3'], all_keys=all_keys, db=db),
                    func(a=['Cs'], b=['Pb', 'Sn'], c=['I3'], all_keys=all_keys, db=db)]

    for counter, energy_dict in enumerate(energy_dicts):
        if reaction == 'demixing':
            reaction_energies = [0, 0]
            averaged_energies = [0, 0]
            max_energies = [0, 0]
            min_energies = [0, 0]
            compositions = [0, 1]
            av_compositions = [0, 1]
        else:
            reaction_energies = []
            averaged_energies = []
            max_energies = []
            min_energies = []
            compositions = []
            av_compositions = []
        zero = list(sorted(energy_dict.keys()))[0]
        for k in energy_dict.keys():
            for e in energy_dict[k]:
                compositions.append(k)
                reaction_energies.append(e)
            averaged_energies.append(np.average(energy_dict[k]))  # -np.average(energy_dict[zero]))
            av_compositions.append(k)
            max_energies.append(max(energy_dict[k]))  # -max(energy_dict[zero]))
            min_energies.append(min(energy_dict[k]))  # -min(energy_dict[zero]))

        x = sorted(av_compositions)
        max_energies = [n for _, n in sorted(zip(av_compositions, max_energies))]
        min_energies = [n for _, n in sorted(zip(av_compositions, min_energies))]
        averaged_energies = [n for _, n in sorted(zip(av_compositions, averaged_energies))]
        # if reaction == 'demixing':
        #    popt, pcov = curve_fit(bowing_curve, x, max_energies)
        #    max_energies = bowing_curve(np.array(x), *popt)
        # else:
        #    _maxe = max_energies
        #    slope, intercept, r_value, p_value, std_err = stats.linregress([x[0],x[-1]], [_maxe[0],_maxe[-1]])
        #    popt, pcov = curve_fit(bowing_curve, x, [_maxe[m] - (slope * x[m] + intercept) for m in range(len(x))])
        #    max_energies = bowing_curve(np.array(x), *popt)
        plt.plot(x, max_energies, '--', c=color_dictionary[counter])

        # if reaction == 'demixing':
        # popt, pcov = curve_fit(bowing_curve, x, min_energies)
        # min_energies = bowing_curve(np.array(x), *popt)
        # else:
        #    _mine = min_energies
        #    slope, intercept, r_value, p_value, std_err = stats.linregress([x[0],x[-1]], [_mine[0], _mine[-1]])
        #    popt, pcov = curve_fit(bowing_curve, x, [_mine[m] - (slope * x[m] + intercept) for m in range(len(x))])
        #    min_energies = bowing_curve(np.array(x), *popt)
        plt.plot(x, min_energies, '--', c=color_dictionary[counter])

        plt.fill_between(x, min_energies, max_energies, color=color_dictionary[counter], alpha=.25)

        # if reaction == 'demixing':
        # popt, pcov = curve_fit(bowing_curve, x, averaged_energies)
        # averaged_energies = bowing_curve(np.array(x), *popt)
        # else:
        #    _ave = averaged_energies
        #    slope, intercept, r_value, p_value, std_err = stats.linregress([x[0],x[-1]], [_ave[0],_ave[-1]])
        #    popt, pcov = curve_fit(bowing_curve, x, [_ave[m] - (slope * x[m] + intercept) for m in range(len(x))])
        #    averaged_energies = bowing_curve(np.array(x), *popt)

        plt.plot(x, averaged_energies, 'o-', c=color_dictionary[counter], lw=2.5,
                 label=label_dict[counter])
        # plt.plot(x,[0 for _ in x],'k:')

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.xlabel('$x$ in Cs(Pb$_{x}$Sn$_{1-x}$)X$_{3}$')
    plt.ylabel(y_label_dict[reaction])
    plt.legend(loc=1)
    # plt.yscale('symlog')
    plt.tight_layout()
    plt.savefig(output)


def plot_mixing_energy_for_single_system(db, a=None, b=None, c=None, output=None, all_keys=None):
    # figure out data from the end members
    _mixing_energies = composition_dependent_demixing_energies(a, all_keys, b, c, db)
    mixing_energies = [0, 0]
    averaged_energies = [0, 0]
    compositions = [0, 1]
    av_compositions = [0, 1]

    for k in _mixing_energies.keys():
        for e in _mixing_energies[k]:
            compositions.append(k)
            mixing_energies.append(e)
        averaged_energies.append(np.average(_mixing_energies[k]))
        av_compositions.append(k)

    gap = max(mixing_energies) - min(mixing_energies)

    plt.scatter(compositions, mixing_energies, marker='o', facecolor='#EFA747', edgecolor='k', alpha=0.5, s=80)

    averaged_energies = [x for _, x in sorted(zip(av_compositions, averaged_energies))]
    x = list(sorted(av_compositions))
    popt, pcov = curve_fit(bowing_curve, x, averaged_energies)
    # plt.plot(x, bowing_curve(np.array(x), *popt), '#F22F08', label='$b=%5.3f$' % tuple(popt))

    x_label = '$x$ in '
    if len(a) == 1:
        x_label += fix_string(a[0])
    else:
        raise NotImplementedError()
    if len(b) == 1:
        x_label += fix_string(b[0])
    else:
        mixed_site = [''.join([_i for _i in k if not _i.isdigit()]) for k in b]
        x_label += '('
        x_label += str(mixed_site[0]) + '$_{x}$'
        x_label += str(mixed_site[1]) + '$_{1-x}$'
        x_label += ')'
    if len(c) == 1:
        x_label += fix_string(c[0])
    else:
        raise NotImplementedError()

    plt.xlabel(x_label)
    plt.ylabel('Mixing enthalpy $\Delta H_{mix}(x)$ (eV/atom)')

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.ylim([min(mixing_energies) - 0.08 * gap, max(mixing_energies) + 0.08 * gap])
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output)


def bowing_curve(x, b):
    return b * x * (x - 1)


def fix_string(s):
    number = None
    for c in s:
        if c.isdigit():
            number = c
    if number is not None:
        print(number)
        replaced_s = s.replace(number, '$_{' + str(number) + '}$')
        return replaced_s
    else:
        return s


def remove_digit(s):
    return ''.join([i for i in s if not i.isdigit()])


def composition_dependent_snpb_rhm_solution_oxidation_energies(a, b, c, all_keys, db):
    _a = [remove_digit(s) for s in a]
    _b = [remove_digit(s) for s in b]
    _c = [remove_digit(s) for s in c]
    _a2 = [remove_digit(s) + '2' for s in a]
    _c6 = [remove_digit(s) + '6' for s in c]

    _sno = ["SnO2"]
    _sno_energies = []
    _pbo = ['PbO2']
    _pbo_energies = []

    for k in all_keys:
        for key in _sno:
            if (key in k) and ('binar' in k):
                row = db.get(selection=[('uid', '=', k)])
                total_energy = row.key_value_pairs['total_energy']
                structure = map_ase_atoms_to_crystal(row.toatoms())
                total_energy = total_energy / structure.all_atoms_count_dictionaries()['Sn']
                _sno_energies.append(total_energy)
        for key in _pbo:
            if (key in k) and ('binar' in k):
                row = db.get(selection=[('uid', '=', k)])
                total_energy = row.key_value_pairs['total_energy']
                structure = map_ase_atoms_to_crystal(row.toatoms())
                total_energy = total_energy / structure.all_atoms_count_dictionaries()['Pb']
                _pbo_energies.append(total_energy)
        if ('O2' in k) and ('Cs' not in k) and ('Pb' not in k) and ('Sn' not in k):
            row = db.get(selection=[('uid', '=', k)])
            total_energy = row.key_value_pairs['total_energy']
            structure = map_ase_atoms_to_crystal(row.toatoms())
            O_energy = total_energy / structure.all_atoms_count_dictionaries()['O']

    mixed_site = [site for site in [a, b, c] if len(site) == 2][-1]
    mixed_site = [''.join([_i for _i in k if not _i.isdigit()]) for k in mixed_site]

    system_contents = [_a2 + [b[0]] + _c6, _a2 + [b[1]] + _c6, _a2 + b + _c6]
    print(system_contents)
    a2bc6_energies = {}
    for system_content in system_contents:
        for k in all_keys:
            k_contains_all_elements = all([(content in k) for content in system_content])
            if k_contains_all_elements:
                row = db.get(selection=[('uid', '=', k)])
                total_energy = row.key_value_pairs['total_energy']
                structure = map_ase_atoms_to_crystal(row.toatoms())
                e_1 = ''.join([_i for _i in mixed_site[0] if not _i.isdigit()])
                e_2 = ''.join([_i for _i in mixed_site[1] if not _i.isdigit()])
                _d = structure.all_atoms_count_dictionaries()
                if e_1 not in _d.keys(): _d[e_1] = 0
                if e_2 not in _d.keys(): _d[e_2] = 0
                composition = _d[e_1] / (_d[e_1] + _d[e_2])
                if composition not in a2bc6_energies.keys():
                    a2bc6_energies[composition] = []
                a2bc6_energies[composition].append(total_energy / _d['Cs'])
    print(a2bc6_energies.keys())

    system_contents = [a + [b[0]] + c, a + [b[1]] + c, a + b + c]
    reaction_energies = {}
    for system_content in system_contents:
        for k in all_keys:
            k_contains_all_elements = all([(content in k) for content in system_content])
            if k_contains_all_elements:
                row = db.get(selection=[('uid', '=', k)])
                total_energy = row.key_value_pairs['total_energy']

                structure = map_ase_atoms_to_crystal(row.toatoms())
                e_1 = ''.join([_i for _i in mixed_site[0] if not _i.isdigit()])
                e_2 = ''.join([_i for _i in mixed_site[1] if not _i.isdigit()])
                _d = structure.all_atoms_count_dictionaries()

                if e_1 not in _d.keys(): _d[e_1] = 0
                if e_2 not in _d.keys(): _d[e_2] = 0

                composition = _d[e_1] / (_d[e_1] + _d[e_2])

                if composition not in reaction_energies.keys():
                    reaction_energies[composition] = []
                print(k)

                reaction_energy = -1.0 * total_energy \
                                  - (_d[e_1] + _d[e_2]) * O_energy \
                                  + min(a2bc6_energies[composition]) / 2.0 \
                                  + sum([_d['Pb'] / 2 * i for i in _pbo_energies]) \
                                  + sum([_d['Sn'] / 2 * i for i in _pbo_energies])
                reaction_energy = reaction_energy / structure.total_num_atoms()
                reaction_energies[composition].append(reaction_energy)
    return reaction_energies


def composition_dependent_sn_pb_rhm_oxidiation_energies(a, b, c, all_keys, db):
    _a = [remove_digit(s) for s in a]
    _b = [remove_digit(s) for s in b]
    _c = [remove_digit(s) for s in c]

    _sno = ["SnO2"]
    _sno_energies = []
    _pbo = ['PbO2']
    _pbo_energies = []
    _cs_sn_x = ['Cs2Sn' + cc + '6' for cc in _c]
    _cs_sn_x_energies = []
    _cs_pb_x = ['Cs2Pb' + cc + '6' for cc in _c]
    _cs_pb_x_energies = []

    for k in all_keys:
        for key in _sno:
            if (key in k) and ('binar' in k):
                row = db.get(selection=[('uid', '=', k)])
                total_energy = row.key_value_pairs['total_energy']
                structure = map_ase_atoms_to_crystal(row.toatoms())
                total_energy = total_energy / structure.all_atoms_count_dictionaries()['Sn']
                _sno_energies.append(total_energy)
        for key in _pbo:
            if (key in k) and ('binar' in k):
                row = db.get(selection=[('uid', '=', k)])
                total_energy = row.key_value_pairs['total_energy']
                structure = map_ase_atoms_to_crystal(row.toatoms())
                total_energy = total_energy / structure.all_atoms_count_dictionaries()['Pb']
                _pbo_energies.append(total_energy)
        for key in _cs_sn_x:
            if (key in k) and ('pure' in k):
                row = db.get(selection=[('uid', '=', k)])
                total_energy = row.key_value_pairs['total_energy']
                structure = map_ase_atoms_to_crystal(row.toatoms())
                total_energy = total_energy / structure.all_atoms_count_dictionaries()['Sn']
                _cs_sn_x_energies.append(total_energy)
        for key in _cs_pb_x:
            if (key in k) and ('pure' in k):
                row = db.get(selection=[('uid', '=', k)])
                total_energy = row.key_value_pairs['total_energy']
                structure = map_ase_atoms_to_crystal(row.toatoms())
                total_energy = total_energy / structure.all_atoms_count_dictionaries()['Pb']
                _cs_pb_x_energies.append(total_energy)
        if ('O2' in k) and ('Cs' not in k) and ('Pb' not in k) and ('Sn' not in k):
            row = db.get(selection=[('uid', '=', k)])
            total_energy = row.key_value_pairs['total_energy']
            structure = map_ase_atoms_to_crystal(row.toatoms())
            O_energy = total_energy / structure.all_atoms_count_dictionaries()['O']

    mixed_site = [site for site in [a, b, c] if len(site) == 2][-1]
    mixed_site = [''.join([_i for _i in k if not _i.isdigit()]) for k in mixed_site]

    system_contents = [a + [b[0]] + c, a + [b[1]] + c, a + b + c]
    reaction_energies = {}
    for system_content in system_contents:
        for k in all_keys:
            k_contains_all_elements = all([(content in k) for content in system_content])
            if k_contains_all_elements:
                row = db.get(selection=[('uid', '=', k)])
                total_energy = row.key_value_pairs['total_energy']

                structure = map_ase_atoms_to_crystal(row.toatoms())
                e_1 = ''.join([_i for _i in mixed_site[0] if not _i.isdigit()])
                e_2 = ''.join([_i for _i in mixed_site[1] if not _i.isdigit()])
                _d = structure.all_atoms_count_dictionaries()

                if e_1 not in _d.keys(): _d[e_1] = 0
                if e_2 not in _d.keys(): _d[e_2] = 0

                composition = _d[e_1] / (_d[e_1] + _d[e_2])

                if composition not in reaction_energies.keys():
                    reaction_energies[composition] = []

                reaction_energy = -1.0 * total_energy \
                                  - (_d[e_1] + _d[e_2]) * O_energy \
                                  + sum([_d['Pb'] / 2 * i for i in _cs_pb_x_energies]) \
                                  + sum([_d['Sn'] / 2 * i for i in _cs_sn_x_energies]) \
                                  + sum([_d['Pb'] / 2 * i for i in _pbo_energies]) \
                                  + sum([_d['Sn'] / 2 * i for i in _pbo_energies])
                reaction_energy = reaction_energy / structure.total_num_atoms()
                reaction_energies[composition].append(reaction_energy)
    return reaction_energies


def composition_dependent_sn_only_oxidation_energies(a, b, c, all_keys, db):
    _a = [remove_digit(s) for s in a]
    _b = [remove_digit(s) for s in b]
    _c = [remove_digit(s) for s in c]

    _sno = ["SnO2"]
    _sno_energies = []
    _cs_sn_x = ['Cs2Sn' + cc + '6' for cc in _c]
    _cs_sn_x_energies = []
    _cs_pb_x = ['CsPb' + cc + '3' for cc in _c]
    _cs_pb_x_energies = []

    for k in all_keys:
        for key in _sno:
            if (key in k) and ('binar' in k):
                row = db.get(selection=[('uid', '=', k)])
                total_energy = row.key_value_pairs['total_energy']
                structure = map_ase_atoms_to_crystal(row.toatoms())
                total_energy = total_energy / structure.all_atoms_count_dictionaries()['Sn']
                _sno_energies.append(total_energy)
        for key in _cs_sn_x:
            if (key in k) and ('pure' in k):
                row = db.get(selection=[('uid', '=', k)])
                total_energy = row.key_value_pairs['total_energy']
                structure = map_ase_atoms_to_crystal(row.toatoms())
                total_energy = total_energy / structure.all_atoms_count_dictionaries()['Sn']
                _cs_sn_x_energies.append(total_energy)
        for key in _cs_pb_x:
            if (key in k) and ('pure' in k):
                row = db.get(selection=[('uid', '=', k)])
                total_energy = row.key_value_pairs['total_energy']
                structure = map_ase_atoms_to_crystal(row.toatoms())
                total_energy = total_energy / structure.all_atoms_count_dictionaries()['Pb']
                _cs_pb_x_energies.append(total_energy)
        if ('O2' in k) and ('Cs' not in k) and ('Pb' not in k) and ('Sn' not in k):
            row = db.get(selection=[('uid', '=', k)])
            total_energy = row.key_value_pairs['total_energy']
            structure = map_ase_atoms_to_crystal(row.toatoms())
            O_energy = total_energy / structure.all_atoms_count_dictionaries()['O']

    mixed_site = [site for site in [a, b, c] if len(site) == 2][-1]
    mixed_site = [''.join([_i for _i in k if not _i.isdigit()]) for k in mixed_site]

    system_contents = [a + [b[0]] + c, a + [b[1]] + c, a + b + c]
    reaction_energies = {}
    for system_content in system_contents:
        for k in all_keys:
            k_contains_all_elements = all([(content in k) for content in system_content])
            if k_contains_all_elements:
                row = db.get(selection=[('uid', '=', k)])
                total_energy = row.key_value_pairs['total_energy']

                structure = map_ase_atoms_to_crystal(row.toatoms())
                e_1 = ''.join([_i for _i in mixed_site[0] if not _i.isdigit()])
                e_2 = ''.join([_i for _i in mixed_site[1] if not _i.isdigit()])
                _d = structure.all_atoms_count_dictionaries()

                if e_1 not in _d.keys(): _d[e_1] = 0
                if e_2 not in _d.keys(): _d[e_2] = 0

                composition = _d[e_1] / (_d[e_1] + _d[e_2])

                reaction_energy = -1.0 * total_energy \
                                  - _d['Sn'] * O_energy \
                                  + sum([_d['Sn'] * e for e in _cs_sn_x_energies]) / 2 \
                                  + sum([_d['Sn'] * e for e in _sno_energies]) / 2 \
                                  + sum([_d['Pb'] * e for e in _cs_pb_x_energies])
                reaction_energy = reaction_energy / structure.total_num_atoms()
                if composition not in reaction_energies.keys():
                    reaction_energies[composition] = []

                reaction_energies[composition].append(reaction_energy)
    return reaction_energies


def composition_dependent_binary_oxidation_energies(a, b, c, all_keys, db):
    _a = [remove_digit(s) for s in a]
    _b = [remove_digit(s) for s in b]
    _c = [remove_digit(s) for s in c]
    _ax = [aa + cc for aa in _a for cc in _c]
    _ax_energies = []
    _bx = [bb + cc + "2" for bb in _b for cc in _c]
    _bx_energies = []
    _bo = [bb + "O2" for bb in _b]
    _bo_energies = []

    for k in all_keys:
        for a_count, ax_key in enumerate(_ax):
            if (ax_key in k) and ('binar' in k):
                row = db.get(selection=[('uid', '=', k)])
                total_energy = row.key_value_pairs['total_energy']
                structure = map_ase_atoms_to_crystal(row.toatoms())
                total_energy = total_energy / structure.all_atoms_count_dictionaries()[_a[a_count]]
                _ax_energies.append(total_energy)
        for b_count, bx_key in enumerate(_bx):
            if (bx_key in k) and ('binar' in k):
                row = db.get(selection=[('uid', '=', k)])
                total_energy = row.key_value_pairs['total_energy']
                structure = map_ase_atoms_to_crystal(row.toatoms())
                total_energy = total_energy / structure.all_atoms_count_dictionaries()[_b[b_count]]
                _bx_energies.append(total_energy)
        for bo_counter, bo_key in enumerate(_bo):
            if (bo_key in k) and (('binar' in k) or ('pure' in k)):
                row = db.get(selection=[('uid', '=', k)])
                total_energy = row.key_value_pairs['total_energy']
                structure = map_ase_atoms_to_crystal(row.toatoms())
                total_energy = total_energy / structure.all_atoms_count_dictionaries()[_b[bo_counter]]
                _bo_energies.append(total_energy)
        if ('O2' in k) and ('Cs' not in k) and ('Pb' not in k) and ('Sn' not in k):
            row = db.get(selection=[('uid', '=', k)])
            total_energy = row.key_value_pairs['total_energy']
            structure = map_ase_atoms_to_crystal(row.toatoms())
            O_energy = total_energy / structure.all_atoms_count_dictionaries()['O']

    mixed_site = [site for site in [a, b, c] if len(site) == 2][-1]
    mixed_site = [''.join([_i for _i in k if not _i.isdigit()]) for k in mixed_site]

    system_contents = [a + [b[0]] + c, a + [b[1]] + c, a + b + c]
    reaction_energies = {}
    for system_content in system_contents:
        for k in all_keys:
            k_contains_all_elements = all([(content in k) for content in system_content])
            if k_contains_all_elements:
                row = db.get(selection=[('uid', '=', k)])
                total_energy = row.key_value_pairs['total_energy']

                structure = map_ase_atoms_to_crystal(row.toatoms())
                e_1 = ''.join([_i for _i in mixed_site[0] if not _i.isdigit()])
                e_2 = ''.join([_i for _i in mixed_site[1] if not _i.isdigit()])
                _d = structure.all_atoms_count_dictionaries()

                if e_1 not in _d.keys(): _d[e_1] = 0
                if e_2 not in _d.keys(): _d[e_2] = 0

                composition = _d[e_1] / (_d[e_1] + _d[e_2])

                _a_composition = [_d[m] for m in _a]
                _b_composition = [_d[m] for m in _b]
                all_b_count = sum(_b_composition)

                _ax_energies_local = [_a_composition[m] * _ax_energies[m] for m in range(len(_a_composition))]
                _bx_energies_local = [_b_composition[m] * _bx_energies[m] for m in range(len(_b_composition))]
                _bo_energies_local = [_b_composition[m] * _bo_energies[m] for m in range(len(_b_composition))]

                reaction_energy = -1 * total_energy - all_b_count * O_energy + \
                                  sum(_ax_energies_local) + \
                                  sum(_bx_energies_local) / 2 + \
                                  sum(_bo_energies_local) / 2
                reaction_energy = reaction_energy / structure.total_num_atoms()
                if composition not in reaction_energies.keys():
                    reaction_energies[composition] = []

                reaction_energies[composition].append(reaction_energy)
    return reaction_energies


def composition_dependent_binary_halide_decomposition_energies(a, b, c, all_keys, db):
    _a = [remove_digit(s) for s in a]
    _b = [remove_digit(s) for s in b]
    _c = [remove_digit(s) for s in c]
    _ax = [aa + cc for aa in _a for cc in _c]
    _ax_energies = []
    _bx = [bb + cc + "2" for bb in _b for cc in _c]
    _bx_energies = []

    for k in all_keys:
        for a_count, ax_key in enumerate(_ax):
            if (ax_key in k) and ('binar' in k):
                row = db.get(selection=[('uid', '=', k)])
                total_energy = row.key_value_pairs['total_energy']
                structure = map_ase_atoms_to_crystal(row.toatoms())
                total_energy = total_energy / structure.all_atoms_count_dictionaries()[_a[a_count]]
                _ax_energies.append(total_energy)
        for b_count, bx_key in enumerate(_bx):
            if (bx_key in k) and ('binar' in k):
                row = db.get(selection=[('uid', '=', k)])
                total_energy = row.key_value_pairs['total_energy']
                structure = map_ase_atoms_to_crystal(row.toatoms())
                total_energy = total_energy / structure.all_atoms_count_dictionaries()[_b[b_count]]
                _bx_energies.append(total_energy)

    print(_ax, _ax_energies)
    print(_bx, _bx_energies)

    reaction_energies = {}

    mixed_site = [site for site in [a, b, c] if len(site) == 2][-1]
    mixed_site = [''.join([_i for _i in k if not _i.isdigit()]) for k in mixed_site]

    system_contents = [a + [b[0]] + c, a + [b[1]] + c, a + b + c]

    for system_content in system_contents:
        for k in all_keys:
            k_contains_all_elements = all([(content in k) for content in system_content])
            if k_contains_all_elements:
                row = db.get(selection=[('uid', '=', k)])
                total_energy = row.key_value_pairs['total_energy']

                structure = map_ase_atoms_to_crystal(row.toatoms())
                e_1 = ''.join([_i for _i in mixed_site[0] if not _i.isdigit()])
                e_2 = ''.join([_i for _i in mixed_site[1] if not _i.isdigit()])
                _d = structure.all_atoms_count_dictionaries()

                if e_1 not in _d.keys(): _d[e_1] = 0
                if e_2 not in _d.keys(): _d[e_2] = 0

                _a_composition = [_d[m] for m in _a]
                _b_composition = [_d[m] for m in _b]
                _a_energies = [_a_composition[m] * _ax_energies[m] for m in range(len(_a_composition))]
                _b_energies = [_b_composition[m] * _bx_energies[m] for m in range(len(_b_composition))]
                composition = _d[e_1] / (_d[e_1] + _d[e_2])

                reaction_energy = -1.0 * total_energy + sum(_a_energies) + sum(_b_energies)
                reaction_energy = reaction_energy / structure.total_num_atoms()

                if composition not in reaction_energies.keys():
                    reaction_energies[composition] = []

                reaction_energies[composition].append(reaction_energy)
    return reaction_energies


def composition_dependent_demixing_energies(a, b, c, all_keys, db, demixing=True):
    end_members = [_a + _b + _c for _a in a for _b in b for _c in c]
    print(end_members)
    assert (len(end_members) == 2)
    mixed_site = [site for site in [a, b, c] if len(site) == 2][-1]
    mixed_site = [''.join([_i for _i in k if not _i.isdigit()]) for k in mixed_site]
    end_member_total_energies = {k: 0 for k in mixed_site}

    mixing_energies = {}

    # get the total energies of the two end members
    for m in mixed_site:
        for em in end_members:
            if m in em:
                matched_key = [k for k in all_keys if em in k][-1]
                row = db.get(selection=[('uid', '=', matched_key)])
                total_energy = row.key_value_pairs['total_energy']
                end_member_total_energies[m] = total_energy

                structure = map_ase_atoms_to_crystal(row.toatoms())
                element_1 = ''.join([_i for _i in mixed_site[0] if not _i.isdigit()])

                if element_1 in structure.all_atoms_count_dictionaries().keys():
                    composition = 1.0
                else:
                    composition = 0.0

                print(m, total_energy)
                if not demixing:
                    mixing_energies[composition] = [total_energy / structure.total_num_atoms()]
    # figure out which site has been mixed with two chemical elements, then we can decide
    #   the chemical compositions should be measured against which element

    system_content = a + b + c
    for k in all_keys:
        k_contains_all_elements = all([(content in k) for content in system_content])
        if k_contains_all_elements:
            row = db.get(selection=[('uid', '=', k)])
            total_energy = row.key_value_pairs['total_energy']

            structure = map_ase_atoms_to_crystal(row.toatoms())
            element_1 = ''.join([_i for _i in mixed_site[0] if not _i.isdigit()])
            element_2 = ''.join([_i for _i in mixed_site[1] if not _i.isdigit()])
            print(element_1)
            composition = structure.all_atoms_count_dictionaries()[element_1] / (
                    structure.all_atoms_count_dictionaries()[element_1] + structure.all_atoms_count_dictionaries()[
                element_2])

            mixing_energy = - total_energy + composition * end_member_total_energies[element_1] + (1.0 - composition) * \
                            end_member_total_energies[element_2]
            mixing_energy = mixing_energy / structure.total_num_atoms()

            print(k, '\t', element_1, '\t', structure.all_atoms_count_dictionaries()[element_1], '\t', composition,
                  '\t', mixing_energy)

            if composition not in mixing_energies.keys():
                mixing_energies[composition] = []
            if demixing:
                mixing_energies[composition].append(mixing_energy)
            else:
                mixing_energies[composition].append(total_energy / structure.total_num_atoms())
    return mixing_energies


def demixing_free_energies_with_configurational_entropy(a, b, c, all_keys, db, temperature, total_energy_dict):
    demixing_free_energy_dict = {}
    kb = 8.617e-5  # eV/K

    comp = 0.0
    free_en_mix_0 = [math.exp(-1.0 * e / (kb * temperature)) for e in total_energy_dict[comp]]
    free_en_mix_0 = -kb * temperature * math.log(sum(free_en_mix_0))

    comp = 1.0
    free_en_mix_1 = [math.exp(-1.0 * e / (kb * temperature)) for e in total_energy_dict[comp]]
    free_en_mix_1 = -kb * temperature * math.log(sum(free_en_mix_1))

    for comp in list(sorted(total_energy_dict.keys())):
        free_en_mix = [math.exp(-1.0 * e / (kb * temperature)) for e in total_energy_dict[comp]]
        l = len(free_en_mix)
        free_en_mix = -kb * temperature * math.log(sum(free_en_mix))
        free_en_mix = free_en_mix - ((1.0 - comp) * free_en_mix_0 + comp * free_en_mix_1)
        demixing_free_energy_dict[comp] = free_en_mix
        # print(comp, l, free_en_mix)
    # print('/n')
    return demixing_free_energy_dict


def get_helmhotz_fe(c, comp):
    # This is hard coded for the time-being
    helmholtz_data = pickle.load(open('/scratch/dy3/jy8620/Sn_halide_perovskite/equ_MD/free_energies.bp', 'rb'))
    for k in helmholtz_data.keys():
        if (c[-1] == helmholtz_data[k]['X']):
            if (comp == helmholtz_data[k]['composition']):
                return helmholtz_data[k]['free_energy']


def demixing_free_energies_with_configurational_entropy_from_helmholtz(a, b, c, all_keys, db, temperature,
                                                                       total_energy_dict):
    demixing_free_energy_dict = {}
    kb = 8.617e-5  # eV/K

    comp = 0.0
    hel_fe = get_helmhotz_fe(c, comp)
    energies = [e + hel_fe[temperature] * 1.0362E-2 for e in total_energy_dict[comp]]
    free_en_mix_0 = [math.exp(-1.0 * e / (kb * temperature)) for e in energies]
    free_en_mix_0 = -kb * temperature * math.log(sum(free_en_mix_0))

    comp = 1.0
    hel_fe = get_helmhotz_fe(c, comp)
    energies = [e + hel_fe[temperature] * 1.0362E-2 for e in total_energy_dict[comp]]
    free_en_mix_1 = [math.exp(-1.0 * e / (kb * temperature)) for e in energies]
    free_en_mix_1 = -kb * temperature * math.log(sum(free_en_mix_1))

    for comp in list(sorted(total_energy_dict.keys())):
        hel_fe = get_helmhotz_fe(c, comp)
        energies = [e + hel_fe[temperature] * 1.0362E-2 for e in total_energy_dict[comp]]
        free_en_mix = [math.exp(-1.0 * e / (kb * temperature)) for e in energies]
        free_en_mix = -kb * temperature * math.log(sum(free_en_mix))
        free_en_mix = free_en_mix - ((1.0 - comp) * free_en_mix_0 + comp * free_en_mix_1)
        demixing_free_energy_dict[comp] = free_en_mix
    return demixing_free_energy_dict


def plot_demixing_free_energies_with_configurational_entropy(db, a=None, b=None, c=None, output=None, all_keys=None,
                                                             phonon=False):
    from cycler import cycler
    from scipy.signal import savgol_filter
    from scipy import interpolate

    color = plt.cm.coolwarm(np.linspace(0, 1, 8))
    pylab.rcParams['axes.prop_cycle'] = cycler('color', color)
    total_energy_dict = composition_dependent_demixing_energies(a, b, c, all_keys, db, demixing=False)
    for temp in [100, 200, 300, 400, 500, 600, 700, 800]:
        demixing_free_energy_dict = demixing_free_energies_with_configurational_entropy_from_helmholtz(a, b, c,
                                                                                                       all_keys, db,
                                                                                                       temp,
                                                                                                       total_energy_dict)
        compositions = list(sorted(demixing_free_energy_dict.keys()))
        plt.plot(compositions, [demixing_free_energy_dict[k] for k in compositions], 'o-',
                 label=str(temp) + ' K')

    demixing_free_energy_dict = demixing_free_energies_with_configurational_entropy_from_helmholtz(a, b, c,
                                                                                                   all_keys, db,
                                                                                                   800,
                                                                                                   total_energy_dict)
    plt.plot(compositions, [demixing_free_energy_dict[k] for k in compositions], 'o-',
             label='With $F_{vib}$',c=color[-1])

    new_color = plt.cm.coolwarm(np.linspace(0, 1, 8))
    pylab.rcParams['axes.prop_cycle'] = cycler('color', new_color)
    total_energy_dict = composition_dependent_demixing_energies(a, b, c, all_keys, db, demixing=False)
    for temp in [100, 200, 300, 400, 500, 600, 700, 800]:
        demixing_free_energy_dict = demixing_free_energies_with_configurational_entropy(a, b, c, all_keys, db,
                                                                                        temp, total_energy_dict)

        compositions = list(sorted(demixing_free_energy_dict.keys()))
        plt.plot(compositions, [demixing_free_energy_dict[k] for k in compositions], '--')

    demixing_free_energy_dict = demixing_free_energies_with_configurational_entropy(a, b, c,
                                                                                    all_keys, db,
                                                                                    800,
                                                                                    total_energy_dict)
    plt.plot(compositions, [demixing_free_energy_dict[k] for k in compositions], '--',
             label='Without $F_{vib}$',c=new_color[-1])

    x_label = '$x$ in '
    if len(a) == 1:
        x_label += fix_string(a[0])
    else:
        raise NotImplementedError()
    if len(b) == 1:
        x_label += fix_string(b[0])
    else:
        mixed_site = [''.join([_i for _i in k if not _i.isdigit()]) for k in b]
        x_label += '('
        x_label += str(mixed_site[0]) + '$_{x}$'
        x_label += str(mixed_site[1]) + '$_{1-x}$'
        x_label += ')'
    if len(c) == 1:
        x_label += fix_string(c[0])+'$_{3}$'
    else:
        raise NotImplementedError()

    plt.xlabel(x_label)
    plt.ylabel('$\Delta G_{mix}(x)$ (eV/atom)')
    plt.ylim([-0.3,0.025])
    #plt.legend(loc=4)
    plt.tight_layout()
    plt.savefig(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Switches for analyzing the energy landscapes of doped perovskites',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--system", type=str, help="Name of the system to analyze")

    # Probably more flexible and easier to handle if the chemistries of A, B and C sites are explicitly given?
    parser.add_argument('-a', '--a_site', nargs='+',
                        help='A site, in AxByCz, attach a number to specify stoichiometry if >1 , such as Cr2 ')
    parser.add_argument('-b', '--b_site', nargs='+',
                        help='B site, in AxByCz, attach a number to specify stoichiometry if >1')
    parser.add_argument('-c', '--c_site', nargs='+',
                        help='C site, in AxByCz, attach a number to specify stoichiometry if >1, such as Cl3')

    parser.add_argument("--db", type=str, default=os.getcwd() + '/doping.db',
                        help="Name of the database that contains the results of the screenings.")
    parser.add_argument("--output", type=str, help='Name of the output file')

    parser.add_argument("--mixing_energy", action='store_true',
                        help='Plot the mixing enthalpy with respect to the two end members')

    parser.add_argument("--reaction", type=str, default=None,
                        help='Plot reaction energy landscape across all compositions')
    parser.add_argument("--summary", action='store_true', help="Plot a summary for multiple systems")

    parser.add_argument("--free_energies", action='store_true', help="Plot demixing free energies")
    parser.add_argument("--phonon", action='store_true',
                        help='whether vibrational contribution to free energies should be included.')
    args = parser.parse_args()

    if os.path.exists(args.db):
        db = connect(args.db)
    else:
        raise Exception("Database " + args.db + " does not exists, cannot proceed!")

    # ====================================================================
    # this is a hack to get all the uids from the database
    all_uids = []
    _db = sqlite3.connect(args.db)
    cur = _db.cursor()
    cur.execute("SELECT * FROM systems")
    rows = cur.fetchall()

    for row in rows:
        for i in row:
            if 'uid' in str(i):
                this_dict = json.loads(str(i))
                all_uids.append(this_dict['uid'])
    # ====================================================================

    if args.mixing_energy:
        if not args.summary:
            if (len(args.a_site), len(args.b_site), len(args.c_site)) not in list(set(permutations([1, 1, 2]))):
                raise Exception("Must and can only have one of the site with two different chemical elements!")
            plot_mixing_energy_for_single_system(db, args.a_site, args.b_site, args.c_site, args.output,
                                                 all_keys=all_uids)
    if args.reaction:
        plot_all_reaction_energies_for_system(db, args.reaction, args.output, all_keys=all_uids)

    if args.free_energies:
        plot_demixing_free_energies_with_configurational_entropy(db, a=args.a_site, b=args.b_site, c=args.c_site,
                                                                 output=args.output,
                                                                 all_keys=all_uids,
                                                                 phonon=args.phonon)
