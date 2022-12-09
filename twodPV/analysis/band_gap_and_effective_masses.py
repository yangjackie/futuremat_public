from ase.db import connect

from matplotlib import rc
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
rc('text', usetex=True)

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

rc('text', usetex=True)
params = {'legend.fontsize': '12',
          'axes.labelsize': 20,
          'axes.titlesize': 13,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15}
pylab.rcParams.update(params)


from twodPV.bulk_library import A_site_list, B_site_list, C_site_list
termination_types = {'100': ['AO', 'BO2'], '110': ['ABO', 'O2'], '111': ['AO3', 'B']}

db = connect('./2dpv.db')


symbol_dict={0:'o',1:'s',2:'d',3:'^'}
chem_dict={0:'$A^{I}B^{II}_{M}X_{3}$',1:'$A^{I}B^{II}_{TM}X_{3}$',2:'$A^{II}B^{IV}C_{3}$',3:'$A^{I}B^{X}C_{3}$'}

fig, subplots = plt.subplots(1,2,figsize=(12,6))
for c_id,carrier in enumerate(['electrons','holes']):

    for i in range(len(A_site_list)):
        band_gaps = []
        color_list = []
        masses = []
        symbols = []
        for thick in 3, 5, 7, 9:
            for a in A_site_list[i]:
                for orient in ['100', '110', '111']:
                    for term in termination_types[orient]:
                        for b in B_site_list[i]:
                            for c in C_site_list[i]:
                                system_name = a + b + c
                                uid = system_name + '3_' + str(orient) + "_" + str(term) + "_" + str(thick)
                                row = None
                                row = db.get(selection=[('uid', '=', uid)])
                                if row is not None:
                                    try:
                                        if not row.data['band_structure']['is_metal']:
                                            if row.data['band_structure']['direct_band_gap']:
                                                band_gaps.append(
                                                    row.data['band_structure']['direct_band_gap_energy'])
                                                color_list.append('#00743F')
                                            else:
                                                band_gaps.append(
                                                    row.data['band_structure']['indirect_band_gap_energy'])
                                                color_list.append('#F2A104')

                                            if carrier=='electrons':
                                                __this_mass=[]
                                                for item in row.data['band_structure']['electron_eff_mass']:
                                                    if str(item['spin'])=='1':
                                                        __this_mass.append(item['eff_mass'])
                                                masses.append(np.average(__this_mass))
                                                symbols.append(symbol_dict[i])
                                            if carrier=='holes':
                                                __this_mass = []
                                                for item in row.data['band_structure']['hole_eff_mass']:
                                                    if str(item['spin'])=='1':
                                                        __this_mass.append(item['eff_mass'])
                                                masses.append(np.average(__this_mass))
                                    except KeyError:
                                        pass
        print(carrier,len(band_gaps),len(masses))
        subplots[c_id].scatter(band_gaps,[abs(i) for i in masses],marker=symbol_dict[i],fc=color_list,alpha=0.6)
    subplots[c_id].set_ylim([0.2,100])
    subplots[c_id].set_yscale('log')
    subplots[c_id].set_xscale('symlog')
    subplots[c_id].set_xlabel('$E_g^{PBE}$ (eV)')
    if c_id==0:
        subplots[c_id].set_ylabel('$\\left\\vert m_{e}^{*} \\right\\vert$')
    if c_id==1:
        subplots[c_id].set_ylabel('$\\left\\vert m_{h}^{*} \\right\\vert$')

        from matplotlib.patches import Patch

        legend_elements = [Patch(facecolor='#00743F', edgecolor='k', label='Direct'),
                           Patch(facecolor='#F2A104', edgecolor='k', label='Indirect')]

        for i in range(len(A_site_list)):
            legend_elements.append(Line2D([0],[0],marker=symbol_dict[i],markerfacecolor='#00743F',markeredgecolor='None',color='w',markersize=7,label=chem_dict[i]))
        subplots[c_id].legend(handles=legend_elements, loc=4, fontsize=14, ncol=1)

plt.tight_layout()
plt.savefig('bandgap_masses.png')