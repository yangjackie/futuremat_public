"""
This module contains codes that generate a series of distorted crystal structures by displacing a B site atom in
a perovskite structure. This is for performing the energy scan to assess the degree of anhormonicity in the structure.
"""

import argparse
import copy
import os
import glob
import pickle

import matplotlib
from matplotlib import gridspec
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
import numpy as np
import math

from core.utils.loggings import setup_logger
from core.calculators.vasp import Vasp
from core.internal.builders.crystal import *
from core.dao.vasp import VaspReader, VaspWriter


def pes(x, a, b):
    import math
    return a * x * x + b * x * x * x * x


def square(x, a, b):
    return a * x * x + b


def composition_eq_point_curve():
    cwd_1 = os.getcwd()
    labels = {0: 'Cs(Pb$_{x}$Sn$_{1-x})$Cl$_3$',
              1: 'Cs(Pb$_{x}$Sn$_{1-x})$Br$_3$',
              2: 'Cs(Pb$_{x}$Sn$_{1-x})$I$_3$'}
    colors = {0: '#000000',  # hex code for black
              1: '#dd0000',  # hex code for electric red
              2: '#ffce00'  # hex code for tangerine yellow
              }
    for counter, X in enumerate(['Cl','Br','I']):
        data_dict={}
        for dir in [r for r in glob.glob(os.getcwd() + "/*" + str(X) + "*") if '.pdf' not in r]:
            os.chdir(dir)
            print(dir)
            cwd_2 = os.getcwd()
            data = []
            directories = glob.glob(os.getcwd() + "/disp_111_*_0_05")
            if 'CsSnBr3_cubic' in dir: directories = glob.glob(os.getcwd() + "/disp_Sn_111_*_0_05")
            for sub_dir in directories:
                os.chdir(sub_dir)
                print(sub_dir)
                energy = VaspReader(input_location='./OSZICAR').get_free_energies_from_oszicar()[-1]
                crystal = VaspReader(input_location='./POSCAR').read_POSCAR()
                energy = energy / crystal.total_num_atoms()
                info = pickle.load(open('info.p', 'rb'))
                info['energy'] = energy
                data.append(info)
                os.chdir(cwd_2)
            os.chdir(cwd_1)
            displacements = list(sorted([d['displacement'] for d in data]))

            energy = []
            for _dis in displacements:
                for d in data:
                    if d['displacement'] == _dis:
                        energy.append(d['energy'])
            energy = [e - energy[0] for e in energy]

            x = np.array(displacements)
            y = np.array(energy)


            from scipy.optimize import curve_fit
            popt, pcov = curve_fit(pes, x, y)
            x = [0 + (0.6 / 100) * i for i in range(100)]
            y = pes(np.array(x), *popt)
            a = popt[0]
            b = popt[1]
            if 2 * a / (4 * b) > 0:
                min_x = 0
            else:
                min_x = math.sqrt(-2 * a / (4 * b))
            concentration = data[0]['concentration']
            data_dict[1.0 - concentration] = min_x
        os.chdir(cwd_1)
        x=list(sorted(data_dict.keys()))
        plt.plot(x,[data_dict[_x] for _x in x],'o-', c=colors[counter], label=labels[counter], ms=12, lw=3)
    plt.legend()
    plt.xlabel('$x$ in Cs(Pb$_{x}$Sn$_{1-x}$)X$_{3}$')
    plt.ylabel('Minima on PES ($x_{\min}$, \AA)')
    plt.tight_layout()
    plt.savefig('xeq_comp_summary.pdf')

def get_pes_across_composition(X='Cl'):
    cmap = matplotlib.cm.get_cmap('coolwarm')
    cwd_1 = os.getcwd()
    data_dict = {}
    min_x = []
    min_y = []
    for dir in [r for r in glob.glob(os.getcwd() + "/*" + str(X) + "*") if '.pdf' not in r]:
        os.chdir(dir)
        cwd_2 = os.getcwd()
        data = []
        for sub_dir in glob.glob(os.getcwd() + "/disp_111_*_0_05"):
            os.chdir(sub_dir)
            print(sub_dir)
            energy = VaspReader(input_location='./OSZICAR').get_free_energies_from_oszicar()[-1]
            crystal = VaspReader(input_location='./POSCAR').read_POSCAR()
            energy = energy / crystal.total_num_atoms()
            info = pickle.load(open('info.p', 'rb'))
            info['energy'] = energy
            data.append(info)
            os.chdir(cwd_2)
        os.chdir(cwd_1)

        displacements = list(sorted([d['displacement'] for d in data]))
        energy = []
        for _dis in displacements:
            for d in data:
                if d['displacement'] == _dis:
                    energy.append(d['energy'])
        energy = [e - energy[0] for e in energy]

        x = np.array(displacements)
        y = np.array(energy)
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(pes, x, y)
        x = [0 + (0.6 / 100) * i for i in range(100)]
        y = pes(np.array(x), *popt)

        a = popt[0]
        b = popt[1]

        if 2 * a / (4 * b) > 0:
            min_x.append(0)
            min_y.append(0)
        else:
            min_x.append(math.sqrt(-2 * a / (4 * b)))
            min_y.append(pes(math.sqrt(-2 * a / (4 * b)), *popt))

        data_dict[1.0 - data[0]['concentration']] = {'x': x, 'y': y}

    for k in list(sorted(data_dict.keys())):
        plt.plot(data_dict[k]['x'], data_dict[k]['y'], '-', c=cmap(k), label=str(k))
    # plt.legend(loc=2)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.plot(min_x, min_y, 'o', c='#00743F')

    max_displacemnt = 0.6
    plt.xlim([0, max_displacemnt])
    plt.plot([0, max_displacemnt], [0, 0], 'k--')

    plt.xlabel('B-Cation Displacement $\Delta d$ (\AA)')
    plt.ylabel('$E-E_{\Delta d=0}$ (eV/atom)')
    plt.tight_layout()
    plt.savefig(X + "_B_111_landscape.pdf")


def pressure_eq_point_curve():
    cwd = os.getcwd()
    labels = {0: 'CsPbBr$_3$',
              1: 'Cs(Pb$_{0.5}$Sn$_{0.5})$Br$_3$',
              2: 'CsSnBr$_3$'}
    colors = {0: '#000000',  # hex code for black
              1: '#dd0000',  # hex code for electric red
              2: '#ffce00'  # hex code for tangerine yellow
              }
    for counter, sys in enumerate(['CsPbBr3_cubic', 'mixed_CsPbSnBr3_SC_1_1_1_CsPbSnBr3_5_str_17', 'CsSnBr3_cubic']):
        os.chdir(sys)
        this_dir = os.getcwd()
        data = []
        for dir in glob.glob("./disp_*"):

            os.chdir(dir)

            try:
                energy = VaspReader(input_location='./OSZICAR').get_free_energies_from_oszicar()[-1]
                crystal = VaspReader(input_location='./POSCAR').read_POSCAR()
                energy = energy / crystal.total_num_atoms()
                info = pickle.load(open('info.p', 'rb'))
                info['energy'] = energy
                data.append(info)
            except:
                pass
            os.chdir(this_dir)
        deformations = set([d['deformation'] for d in data])
        deformations = list(sorted(deformations))

        min_x = []

        for id, _def in enumerate(deformations):
            displacement = [d['displacement'] for d in data if d['deformation'] == _def]
            displacement = list(sorted(displacement))

            energy = []
            for dis in displacement:
                for d in data:
                    if (d['deformation'] == _def) and (d['displacement'] == dis):
                        energy.append(d['energy'])
            energy = [e - energy[0] for e in energy]

            x = np.array(displacement)
            y = np.array(energy)
            from scipy.optimize import curve_fit
            popt, pcov = curve_fit(pes, x, y)

            a = popt[0]
            b = popt[1]

            if 2 * a / (4 * b) > 0:
                min_x.append(0)
            else:
                min_x.append(math.sqrt(-2 * a / (4 * b)))

        plt.plot(deformations, min_x, 'o--', label=labels[counter], ms=10, c=colors[counter])

        _deformations = []
        _min_x = []
        for t in range(len(min_x)):
            if (min_x[t] > 0):
                _deformations.append(deformations[t])
                _min_x.append(min_x[t])
        x = np.array(_deformations)
        y = np.array(_min_x)

        from scipy import interpolate
        f = interpolate.interp1d(x, y)
        x = np.array([min(_deformations) + k * (max(_deformations) - min(_deformations)) / 100 for k in range(100)])
        y = f(x)

        from scipy.optimize import curve_fit

        popt, pcov = curve_fit(square, y, x, maxfev=2000)
        a = popt[0]
        b = popt[1]
        _y = np.array([(max(y) + 0.1 * max(y)) * k / 100 for k in range(100)])
        _x = square(_y, *popt)
        print(_x[0])
        plt.plot(_x, _y, '-', lw=2,c=colors[counter])

        os.chdir(cwd)
    plt.xlim([0.015, 0.052])
    plt.xlabel('Lattice Deformation $\delta $')
    plt.ylabel('Minima on PES ($x_{\min}$, \AA)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("xeq_pressure_summary_Br.pdf")




def get_this_pes(directory=os.getcwd()):
    cmap = matplotlib.cm.get_cmap('YlOrRd')

    cwd = os.getcwd()
    data = []
    for dir in glob.glob(directory + "/disp_*"):
        os.chdir(dir)
        try:
            energy = VaspReader(input_location='./OSZICAR').get_free_energies_from_oszicar()[-1]
            crystal = VaspReader(input_location='./POSCAR').read_POSCAR()
            energy = energy / crystal.total_num_atoms()
            info = pickle.load(open('info.p', 'rb'))
            info['energy'] = energy
            data.append(info)
        except:
            pass
        os.chdir(cwd)

    deformations = set([d['deformation'] for d in data])
    deformations = list(sorted(deformations))
    max_def = max(deformations) + 0.0001
    max_displacement = 0

    min_x = []
    min_y = []

    for id, _def in enumerate(deformations):
        displacement = [d['displacement'] for d in data if d['deformation'] == _def]
        displacement = list(sorted(displacement))

        energy = []
        for dis in displacement:
            for d in data:
                if (d['deformation'] == _def) and (d['displacement'] == dis):
                    energy.append(d['energy'])
        energy = [e - energy[0] for e in energy]

        x = np.array(displacement)
        y = np.array(energy)
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(pes, x, y)

        x = [0 + (0.6 / 100) * i for i in range(100)]
        y = pes(np.array(x), *popt)

        a = popt[0]
        b = popt[1]

        if 2 * a / (4 * b) > 0:
            min_x.append(0)
            min_y.append(0)
        else:
            min_x.append(math.sqrt(-2 * a / (4 * b)))
            min_y.append(pes(math.sqrt(-2 * a / (4 * b)), *popt))

        if id % 2 == 0:
            plt.plot(x, y, '-', c=cmap((0.05 + _def) / (2 * max_def)), label=str(_def * 100) + '\%', lw=2)
        else:
            plt.plot(x, y, '-', c=cmap((0.05 + _def) / (2 * max_def)), lw=2)

        max_displacement = max(x)

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.plot([0, max_displacement], [0, 0], 'k--')
    plt.plot(min_x, min_y, 'o-', c='k', lw=3.5)
    plt.legend()
    plt.ylim([-0.01, 0.0150])
    plt.xlim([0, max_displacement])
    plt.xlabel('B-Cation Displacement $\Delta d$ (\AA)')
    plt.ylabel('$E-E_{\Delta d=0}$ (eV/atom)')
    plt.tight_layout()
    plt.savefig('pes.pdf')


def make_distorted_structure(poscar, direction='111', max_displacement=0.6, number_of_points=15,
                             lattice_deformation=0, atom_to_displace=['Sn', 'Pb']):
    for counter, delta_d in enumerate([0 + max_displacement / number_of_points * k for k in range(number_of_points)]):
        crystal = VaspReader(input_location=poscar).read_POSCAR()
        # expand or shrink the lattice parameters
        crystal = deform_crystal_by_lattice_expansion_coefficients(crystal,
                                                                   def_fraction=[lattice_deformation for _ in range(3)])

        d = [int(_d) for _d in list(direction)]
        displacement_vector = cVector3D(d[0], d[1], d[2]).normalise()

        # just displace the first atom that is the element that we want to displace
        _displace_vec = displacement_vector.vec_scale(delta_d)

        # _next = True

        for mol in crystal.asymmetric_unit:
            for atom in mol.atoms:
                if (atom.label in atom_to_displace):  # and (_next is True):
                    print(atom.label)
                    atom.position = atom.position + _displace_vec
                    # _next = False

        try:
            n = crystal.all_atoms_count_dictionaries()['Sn']
        except KeyError:
            n = 0

        folder_name = 'disp' + '_' + str(direction) + '_' + str(n) + '_str_' + str(
            counter) + '_a_def_' + str(lattice_deformation).replace('.', '_')
        cwd = os.getcwd()
        try:
            os.makedirs(cwd + '/' + folder_name)
        except:
            pass

        os.chdir(cwd + '/' + folder_name)

        _dict = crystal.all_atoms_count_dictionaries()

        if 'Sn' not in _dict.keys():
            _dict['Sn'] = 0
        if 'Pb' not in _dict.keys():
            _dict['Pb'] = 0

        system_dict = {'atom_to_displace': atom_to_displace,
                       'concentration': _dict['Sn'] / (_dict['Sn'] + _dict['Pb']),
                       'displacement': delta_d,
                       'id': counter,
                       'direction': direction,
                       'deformation': lattice_deformation}
        pickle.dump(system_dict, open('info.p', 'wb'))

        VaspWriter().write_structure(crystal, 'POSCAR')
        os.chdir(cwd)


single_point_pbe = {'PREC': 'HIGH',
                    'ISMEAR': 0,
                    'SIGMA': 0.01,
                    'EDIFF': 1e-05,
                    'IALGO': 48,
                    'ISPIN': 1,
                    'NELM': 500,
                    'AMIN': 0.01,
                    'ISYM': 0,
                    'PREC': 'HIGH',
                    'ENCUT': 350,
                    'NSW': 0,
                    'LWAVE': False,
                    'clean_after_success': True,
                    'use_gw': False,
                    'Gamma_centered': True,
                    'NPAR': 16}


def run_single_point():
    logger = setup_logger(output_filename="dielectrics.log")
    structure = VaspReader(input_location='./POSCAR').read_POSCAR()
    vasp = Vasp(**single_point_pbe)
    vasp.set_crystal(structure)
    vasp.execute()
    if vasp.completed:
        logger.info("PBE self-consistent run completed properly.")
    else:
        raise Exception("PBE self-consistent run failed to converge, will stop proceeding")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generating customized distorted ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--poscar", type=str, help="location of the poscar")
    parser.add_argument("--lattice_deformation", type=float, help='percentage deformation of the lattice parameters',
                        default=0.0)
    parser.add_argument("--direction", type=str, help="direction along which the atom will be displaced along",
                        default='111')
    parser.add_argument("--max_displacement", type=float, help='maximum distance of atomic displacement', default=0.6)
    parser.add_argument("--number_of_points", type=int, help="number of points to scan on the PES", default=20)
    parser.add_argument("--atom_to_displace", nargs='+', help="which atom to displace", default='Sn Pb')
    parser.add_argument("--gen_conf", action='store_true', help='switch to choose generating conformations')
    parser.add_argument("--collect_data", action='store_true', help='switch to choose generating conformations')
    parser.add_argument("--X", type=str, help='the halogen atom you want to look at', default=None)
    parser.add_argument("--pressure_eq_curve", action='store_true',
                        help='Plot the equilibrium displacement as function of lattice deformation')
    parser.add_argument("--composition_eq_curve", action='store_true',
                        help='Plot the equilibrium displacement as function of Sn compositions')
    args = parser.parse_args()

    if args.gen_conf:
        make_distorted_structure(args.poscar,
                                 direction=args.direction,
                                 max_displacement=args.max_displacement,
                                 number_of_points=args.number_of_points,
                                 atom_to_displace=args.atom_to_displace,
                                 lattice_deformation=args.lattice_deformation)
    elif args.collect_data:
        if args.X is None:
            get_this_pes()
        else:
            get_pes_across_composition(args.X)
    elif args.pressure_eq_curve:
        pressure_eq_point_curve()
    elif args.composition_eq_curve:
        composition_eq_point_curve()
