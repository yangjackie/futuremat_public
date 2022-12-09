import itertools
from ase.db import connect
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import rc

rc('text', usetex=True)
import matplotlib.pylab as pylab

params = {'legend.fontsize': '11',
          'figure.figsize': (6, 5),
          'axes.labelsize': 20,
          'axes.titlesize': 16,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)
from twodPV.bulk_library import A_site_list, B_site_list, C_site_list
from core.models.element import *
from core.internal.builders.crystal import map_ase_atoms_to_crystal

termination_types = {'100': ['AO', 'BO2'], '110': ['O2', 'ABO'], '111': ['AO3', 'B']}
unique_element_list = list(itertools.chain.from_iterable(A_site_list + B_site_list + C_site_list))
unique_element_list = list(set(unique_element_list))


def descriptor_entry_dict():
    # arrange the list in the order of their atomic numbers
    rev_atom_numbers = {atomic_numbers[k]: k for k in atomic_numbers.keys()}
    sorted_atomic_numbers = list(sorted([atomic_numbers[e] for e in unique_element_list]))
    # entry index in the chemical descriptor for every structure
    output = {}
    for id, atomic_number in enumerate(sorted_atomic_numbers):
        output[rev_atom_numbers[atomic_number]] = id
    return output


db = connect("./2dpv.db")

design_matrix = []

composition_face_colors = []
composition_edge_colors = []
thickness_colors = []
formation_energy_color = []
vibrational_freq_color = []
sizes = []
for thick in [3, 5, 7, 9]:
    for orientation in ['100', '110', '111']:
        for term_type_id, term_type in enumerate(termination_types[orientation]):
            for i in range(len(A_site_list)):
                for a in A_site_list[i]:
                    for b in B_site_list[i]:
                        for c in C_site_list[i]:

                            if i == 0:
                                c_color = '#A3586D'
                            if i == 1:
                                c_color = '#5C4A72'
                            if i == 2:
                                c_color = '#F3B05A'
                            if i == 3:
                                c_color = '#F4874B'

                            if thick == 3:
                                t_color = '#7AC7A9'
                            if thick == 5:
                                t_color = '#90CA57'
                            if thick == 7:
                                t_color = '#F1D628'
                            if thick == 9:
                                t_color = '#2B8283'

                            system_name = a + b + c

                            uid = system_name + '3_pm3m'
                            row = db.get(selection=[('uid', '=', uid)])
                            pm3m_formation_e = row.key_value_pairs['formation_energy']

                            uid = system_name + '3_' + str(orientation) + "_" + str(term_type) + "_" + str(thick)
                            row = db.get(selection=[('uid', '=', uid)])

                            # construct a simple chemical descriptor for this structure
                            # this is adpted from Chem. Mater. 2017, 29, 6220−6227.
                            crystal = map_ase_atoms_to_crystal(row.toatoms())
                            _ac_dict = crystal.all_atoms_count_dictionaries()
                            _ac_frac_dict = {k: _ac_dict[k] / crystal.total_num_atoms() for k in _ac_dict.keys()}
                            chemical_descriptor = np.array([0.0 for _ in range(len(unique_element_list) + 2)])
                            for k in _ac_frac_dict.keys():
                                chemical_descriptor[descriptor_entry_dict()[k]] = _ac_frac_dict[k]
                            chemical_descriptor[-1] = float(thick) / 9.0
                            chemical_descriptor[-2] = (float(term_type_id) + 1.0) / 2.0
                            add_to_design_matrix = False
                            try:
                                twod_formation_e = row.key_value_pairs['formation_energy']
                                relative_formation_e = twod_formation_e - pm3m_formation_e

                                gamma_point_freqs = None
                                min_gamma_freq = None
                                try:
                                    gamma_point_freqs = row.data['gamma_phonon_freq']
                                except:
                                    pass

                                if gamma_point_freqs is not None:
                                    min_gamma_freq = min([(f ** 2).real for f in gamma_point_freqs])
                                else:
                                    min_gamma_freq = 0.0
                                if min_gamma_freq <= -6:
                                    min_gamma_freq = -6

                                add_to_design_matrix = True
                            except:
                                pass

                            if add_to_design_matrix:
                                design_matrix.append(chemical_descriptor)
                                thickness_colors.append(t_color)
                                if relative_formation_e > 2:
                                    relative_formation_e = 2
                                formation_energy_color.append(relative_formation_e)

                                composition_edge_colors.append(c_color)
                                if term_type_id == 0:
                                    composition_face_colors.append(c_color)
                                    sizes.append(30)
                                elif term_type_id == 1:
                                    composition_face_colors.append('w')
                                    sizes.append(80)

                                vibrational_freq_color.append(min_gamma_freq)

print("=====Generating a low-dimensional embedding of the data=====")
design_matrix = np.array(design_matrix)
design_matrix_embedded = TSNE(n_components=2, perplexity=10).fit_transform(design_matrix)
from matplotlib.patches import Patch

# plt.figure(figsize=(21,14))
fig, ax = plt.subplots(2, 3, figsize=[21, 14])
ax[0, 0].scatter(design_matrix_embedded[:, 0], design_matrix_embedded[:, 1], marker='o',
                 facecolor=composition_face_colors,
                 edgecolors=composition_edge_colors, alpha=0.5, s=sizes)  # , cmap=plt.get_cmap('RdYlBu'))
legend_elements = [
    Patch(facecolor='#A3586D', edgecolor='none', label='$A^{I}B^{II}_{M}X_{3}$-(AX,X$_{2}$,AX$_{3}$)'),
    Patch(edgecolor='#A3586D', facecolor='none', label='$A^{I}B^{II}_{M}X_{3}$-(BX$_{2}$,ABX,B)'),
    Patch(facecolor='#5C4A72', edgecolor='none', label='$A^{I}B^{II}_{TM}X_{3}$-(AX,X$_{2}$,AX$_{3}$)'),
    Patch(edgecolor='#5C4A72', facecolor='none', label='$A^{I}B^{II}_{TM}X_{3}$-(BX$_{2}$,ABX,B)'),
    Patch(facecolor='#F3B05A', edgecolor='none', label='$A^{II}B^{IV}C_{3}$-(AC,C$_{2}$,AC$_{3}$)'),
    Patch(edgecolor='#F3B05A', facecolor='none', label='$A^{II}B^{IV}C_{3}$-(BC$_{2}$,ABC,B)'),
    Patch(facecolor='#F4874B', edgecolor='none', label='$A^{I}B^{X}C_{3}$-(AC,C$_{2}$,AC$_{3}$)'),
    Patch(edgecolor='#F4874B', facecolor='none', label='$A^{I}B^{X}C_{3}$-(BC$_{2}$,ABC,B)')
]
ax[0, 0].legend(handles=legend_elements)
ax[0, 0].axis('off')



ax[0, 1].scatter(design_matrix_embedded[:, 0], design_matrix_embedded[:, 1], marker='o', facecolors=thickness_colors,
                 edgecolors=None,
                 alpha=0.5)
legend_elements = [
    Patch(facecolor='#7AC7A9', edgecolor='none', label='$n=3$'),
    Patch(facecolor='#90CA57', edgecolor='none', label='$n=5$'),
    Patch(facecolor='#F1D628', edgecolor='none', label='$n=7$'),
    Patch(facecolor='#2B8283', edgecolor='none', label='$n=9$'),
]
ax[0, 1].legend(handles=legend_elements)
ax[0, 1].axis('off')



im = ax[0, 2].scatter(design_matrix_embedded[:, 0], design_matrix_embedded[:, 1], marker='o', c=formation_energy_color,
                      edgecolors=None,
                      alpha=0.5, cmap=plt.get_cmap('PuOr'))
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axins1 = inset_axes(ax[0, 2],
                    width="50%",  # width = 50% of parent_bbox width
                    height="5%",  # height : 5%
                    loc='upper right')
fig.colorbar(im, cax=axins1, orientation="horizontal", label="$E_f^{2D,n=3}-E_{f}^{Pm\\bar{3}m}$ (eV/atom)")
ax[0, 2].axis('off')


ax[1, 0].axis('off')
im = ax[1, 0].scatter(design_matrix_embedded[:, 0], design_matrix_embedded[:, 1], marker='o', c=vibrational_freq_color,
                      edgecolors=None, alpha=0.5, cmap=plt.get_cmap('coolwarm'))
axins1 = inset_axes(ax[1, 0],
                    width="50%",  # width = 50% of parent_bbox width
                    height="2.5%",  # height : 5%
                    loc='lower right')
fig.colorbar(im, cax=axins1, orientation="horizontal", label="$\\omega_{\\min}^2$")

ax[1, 1].axis('off')
ax[1, 2].axis('off')

plt.tight_layout()
plt.savefig('tsne.pdf')
