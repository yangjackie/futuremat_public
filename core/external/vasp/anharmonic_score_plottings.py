import glob
import os
import matplotlib.pyplot as plt
from core.external.vasp.anharmonic_score import *
from core.dao.vasp import VaspReader
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

cwd = os.getcwd()
halo = 'I'
folders = glob.glob('*' + halo + '*')

_sigma_100K = []
_sigma_300K = []

_compositions = []

_sigma_ref_300K =[]

for folder in ['../CsSnI_Pnma','../CsPbI_Pnma']:
    print(folder)
    #if not os.path.isdir(folder): continue
    os.chdir(folder)

    # get the compositions
    crystal = VaspReader(input_location="./phonon/POSCAR").read_POSCAR()
    _d = crystal.all_atoms_count_dictionaries()
    scorer = AnharmonicScore(md_frames='./vasprun_md.xml', ref_frame='./phonon/POSCAR', atoms=None, unit_cell_frame='./phonon/POSCAR')
    __sigmas, _ = scorer.structural_sigma(return_trajectory=True)
    _sigma_ref_300K.append(__sigmas[2000:])

    os.chdir(cwd)

for folder in folders:
    if not os.path.isdir(folder): continue
    os.chdir(folder)

    # get the compositions
    crystal = VaspReader(input_location="./POSCAR_equ").read_POSCAR()
    _d = crystal.all_atoms_count_dictionaries()
    if 'Pb' not in _d.keys(): _d['Pb'] = 0
    if 'Sn' not in _d.keys(): _d['Sn'] = 0
    composition = _d['Pb'] / (_d['Sn'] + _d['Pb'])
    _compositions.append(composition)
    scorer = AnharmonicScore(md_frames='./vasprun_100K_correct.xml', ref_frame='./POSCAR_equ', atoms=None, unit_cell_frame='./POSCAR_equ')
    __sigmas, _ = scorer.structural_sigma(return_trajectory=True)
    _sigma_100K.append(__sigmas[2000:])

    scorer = AnharmonicScore(md_frames='./vasprun_300K.xml', ref_frame='./POSCAR_equ', atoms=None, unit_cell_frame='./POSCAR_equ')
    __sigmas, _ = scorer.structural_sigma(return_trajectory=True)
    _sigma_300K.append(__sigmas[2000:])

    os.chdir(cwd)

fig, ax = plt.subplots(figsize=(6,5))

compositions = list(sorted(_compositions))
sigma_100K = []
sigma_300K = []

for i in range(len(compositions)):
    for j in range(len(_compositions)):
        if _compositions[j] == compositions[i]:
            sigma_100K.append(_sigma_100K[j])
            sigma_300K.append(_sigma_300K[j])

c = '#1fbfb8'
bp1 = plt.boxplot(sigma_100K, positions=[_x for _x in compositions],
                  widths=[0.45 / len(compositions) for _ in compositions], patch_artist=True,
                  boxprops=dict(facecolor=c, color=c, alpha=0.7),
                  capprops=dict(color=c),
                  whiskerprops=dict(color=c),
                  flierprops=dict(color=c, markeredgecolor=c),
                  medianprops=dict(color=c),
                  showfliers=False)

c = '#031163'
bp2 = plt.boxplot(sigma_300K, positions=[_x for _x in compositions],
                  widths=[0.45 / len(compositions) for _ in compositions], patch_artist=True,
                  boxprops=dict(facecolor=c, color=c, alpha=0.7),
                  capprops=dict(color=c),
                  whiskerprops=dict(color=c),
                  flierprops=dict(color=c, markeredgecolor=c),
                  medianprops=dict(color=c),
                  showfliers=False)

c = 'r'
bp3 = plt.boxplot(_sigma_ref_300K, positions=[0.0, 1.0],
                  widths=[0.45 / len(compositions) for _ in [0,1]], patch_artist=True,
                  boxprops=dict(facecolor=c, color=c, alpha=0.7),
                  capprops=dict(color=c),
                  whiskerprops=dict(color=c),
                  flierprops=dict(color=c, markeredgecolor=c),
                  medianprops=dict(color=c),
                  showfliers=False)

ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], ['100 K', '300 K', '$\\gamma$-phase 300 K'], loc='upper right')
ax.set_ylabel('$\\sigma(x,T)$', fontsize=26)
ax.set_xlabel('Cs(Pb$_{x}$Sn$_{1-x}$)'+str(halo)+'$_{3}$', fontsize=16)
xticklabels=['%.2f' % i for i in compositions]
ax.set_xticklabels(xticklabels, rotation = 45)
#start, end = ax.get_xlim()
#ax.set_xticks(ax.get_xticks()[::2])
ax.set_xlim([-0.06, 1.06])
ax.set_ylim([0.5, 4.5])
plt.tight_layout()
plt.savefig("sigma_box_" + str(halo) + ".pdf")
