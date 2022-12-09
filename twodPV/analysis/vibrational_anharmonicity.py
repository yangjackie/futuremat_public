import os
import argparse
import numpy as np
import pickle
from core.external.vasp.anharmonic_score import *
from core.calculators.vasp import *

import matplotlib.pylab as pylab

params = {'legend.fontsize': '8',
          'axes.labelsize': 36,
          'axes.titlesize': 24,
          'xtick.labelsize': 24,
          'ytick.labelsize': 28}
pylab.rcParams.update(params)

import matplotlib.pyplot as plt

def collect_data():
    data_dictionary = {}
    system_dir = '.'
    system_dir = [os.path.join(system_dir, o) for o in os.listdir(system_dir) if
                  os.path.isdir(os.path.join(system_dir, o))]

    for sd in system_dir:
        os.chdir(sd)

        compound_dir = '.'
        compound_dir = [os.path.join(compound_dir, o) for o in os.listdir(compound_dir) if
                        os.path.isdir(os.path.join(compound_dir, o))]

        for cd in compound_dir:
            os.chdir(cd)

            #first find the finite displacement calculations that are all converged.
            os.chdir('thermal_displacements')

            print('\n')
            print(os.getcwd())
            print('\n')

            random_dir = '.'
            random_dir = [os.path.join(random_dir, o) for o in os.listdir(random_dir) if
                            os.path.isdir(os.path.join(random_dir, o))]
            include_dir = []
            for c,d in enumerate(random_dir):
                os.chdir(d)
                vasp = Vasp()
                vasp.check_convergence()

                if vasp.completed:
                    #print(str(c+1)+'/'+str(len(random_dir))+' Random displacement calculation completed, to be included ...')
                    include_dir.append(d.replace('.','./thermal_displacements')+'/vasprun.xml')
                os.chdir('..')

            os.chdir('..')

            try:
                scorer = AnharmonicScore(ref_frame='./POSCAR_super',unit_cell_frame='./POSCAR_super',md_frames=include_dir,force_constants='./force_constants.hdf5')
            except:
                scorer = AnharmonicScore(ref_frame='./POSCAR_super', unit_cell_frame='./POSCAR_super',
                                         md_frames=include_dir)
            sigmas, _ = scorer.structural_sigma(return_trajectory=True)

            sys = sd+cd
            sys = sys.replace('./','')
            sys = sys.replace('ll','ll_')
            print(sys, np.average(sigmas))

            data_dictionary[sys] = sigmas

            os.chdir('..')
        os.chdir('..')

    pickle.dump(data_dictionary,open('anharmonic_data.p','wb'))

def plot_data():

    data = pickle.load(open('anharmonic_data.p','rb'))

    color_dict = {0: '#A3586D', 1: '#5C4A72', 2: '#F3B05A', 3: '#F4874B'}
    colors = [[color_dict[i] for _ in range(6)] for i in range(4)]
    colors = [item for sublist in colors for item in sublist]

    colors_2 = [[color_dict[i] for _ in range(12)] for i in range(4)]
    colors_2 = [item for sublist in colors_2 for item in sublist]

    group_dict = {0: '$A^{I}B^{II}_{M}X_{3}$',
                  1: '$A^{I}B^{II}_{TM}X_{3}$',
                  2: '$A^{II}B^{IV}C_{3}$',
                  3: '$A^{I}B^{X}C_{3}$'}
    system_dict = {0:['CsGeI_9','CsGeBr_9','CsGeF_9','CsSnF_3','CsGeCl_9','CsGeBr_9'],
                   1:['CsNbBr_9','CsNbI_9','CsNbI_9','CsNbI_9','CsNbBr_9','CsNbI_9'],
                   2:['BaZrO_9','BaZrO_9','BaZrO_9','BaZrSe_9','SrZrO_9','BaZrO_9'],
                   3:['CsNbO_9','KTaO_9','CsNbTe_9','CsTaO_9','CsNbTe_9','CsTaO_9']}
    term_dict = {0:'slab_100_AO_small',
                 1:'slab_100_BO2_small',
                 2:'slab_110_ABO_small',
                 3:'slab_110_O2_small',
                 4:'slab_111_AO3_small',
                 5:'slab_111_B_small'}

    fig = plt.figure(figsize=(26,12))
    xs=[]
    _data_to_plot=[]
    x_tick_labels=[]
    x_ticks=[]
    counter=0
    for group_id in group_dict.keys():
        for sys_count,system in enumerate(system_dict[group_id]):
            for k in data.keys():
                if (term_dict[sys_count] in k) and (system in k):
                    xs.append(counter)
                    _data_to_plot.append(data[k])
                    x_tick_labels.append(system.replace('_','$_{3}$-'))
                    counter+=1
                    x_ticks.append(counter)

    print(counter)
    ax = fig.add_subplot(111)
    bp = ax.boxplot(_data_to_plot,showfliers=False,patch_artist = True)
    ax.set_xlim([0.5,counter+0.5])
    ax.set_ylim([0.2,1.1])
    ax.set_ylabel('$\\sigma$(300 K)')

    for vx in [6.5,12.5,18.5]:
        ax.vlines(vx,ymin=0.2,ymax=1.1,linestyles='--',colors='k')
    ax.set_xticklabels(x_tick_labels)

    for tick in ax.get_xticklabels():
        tick.set_rotation(60)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(2)

    for whiskers, color in zip(bp['whiskers'], colors_2):
        whiskers.set(color=color,linewidth=2)

    for cap, color in zip(bp['caps'], colors_2):
        cap.set(color=color,linewidth=2)

    for median, color in zip(bp['medians'], colors):
        median.set(color=color,linewidth=2)

    plt.text(0.85, 0.22, '[100]//AO$_{2}$',rotation=90,fontsize=24)
    plt.text(1.85, 0.22, '[100]//BO', rotation=90, fontsize=24)
    plt.text(2.85, 0.22, '[110]//ABO', rotation=90, fontsize=24)
    plt.text(3.85, 0.22, '[110]//O$_{2}$', rotation=90, fontsize=24)
    plt.text(4.85, 0.22, '[111]//AO$_{3}$', rotation=90, fontsize=24)
    plt.text(5.85, 0.22, '[111]//B', rotation=90, fontsize=24)

    plt.text(2.5, 1.04, group_dict[0], color=color_dict[0], fontsize=34)
    plt.text(8.5, 1.04, group_dict[1], color=color_dict[1], fontsize=34)
    plt.text(14.5, 1.04, group_dict[2], color=color_dict[2], fontsize=34)
    plt.text(20.5, 1.04, group_dict[3], color=color_dict[3], fontsize=34)

    plt.tight_layout()
    plt.savefig('2D_anharmonic_box.pdf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='analyze anharmonicity scores for 2D perovskites',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--collect_data", action='store_true', help='calculate and collect the anharmonicty score')
    parser.add_argument("--plot_data", action='store_true', help='plot the anharmonicty scores for 2D perovskites')
    args = parser.parse_args()

    if args.collect_data:
        collect_data()
    elif args.plot_data:
        plot_data()