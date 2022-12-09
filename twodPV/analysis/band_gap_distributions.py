from ase.db import connect

from matplotlib import rc
rc('text', usetex=True)

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

params = {'legend.fontsize': '16',
          'figure.figsize': (6, 5),
          'axes.labelsize': 50,
          'axes.titlesize': 60,
          'xtick.labelsize': 45,
          'ytick.labelsize': 45}

pylab.rcParams.update(params)

from twodPV.bulk_library import A_site_list, B_site_list, C_site_list
termination_types = {'100': ['AO', 'BO2'], '110': ['ABO', 'O2'], '111': ['AO3', 'B']}

from scipy.stats import gaussian_kde

def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    # Kernel Density Estimation with Scipy
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)


db = connect('./2dpv.db')

plt.figure(figsize=(64,64))

for i in range(len(A_site_list)):
    for thick_id,thick in enumerate([3,5,7,9]):
        metal_counter=0
        semiconductor_counter=0

        direct_band_gaps = []
        indirect_band_gaps = []
        for a in A_site_list[i]:
            for orient in ['100','110','111']:
                for term in termination_types[orient]:

                    for b in B_site_list[i]:
                        for c in C_site_list[i]:
                            system_name = a + b + c
                            uid = system_name + '3_' + str(orient) + "_" + str(term) + "_" + str(thick)
                            row = None
                            row = db.get(selection=[('uid', '=', uid)])

                            if row is not None:
                                try:
                                    if row.data['band_structure']['is_metal']:
                                       metal_counter+=1
                                    else:
                                        semiconductor_counter+=1
                                        if row.data['band_structure']['direct_band_gap']:
                                            direct_band_gaps.append(row.data['band_structure']['direct_band_gap_energy'])
                                        else:
                                            indirect_band_gaps.append(row.data['band_structure']['indirect_band_gap_energy'])
                                except KeyError:
                                    pass
        print(i,metal_counter,semiconductor_counter,len(direct_band_gaps),len(indirect_band_gaps))

        ax=plt.subplot(4,4,i*4+thick_id+1)

        #y_grid = np.linspace(0,max(direct_band_gaps)+1,50)
        #total_pdf = kde_scipy(np.array(direct_band_gaps), y_grid, bandwidth=0.05)
        #plt.plot( y_grid, total_pdf,'-', lw=4)

        #y_grid = np.linspace(0, max(indirect_band_gaps) + 1, 50)
        #total_pdf = kde_scipy(np.array(indirect_band_gaps), y_grid, bandwidth=0.05)
        #plt.plot( y_grid, total_pdf,'-', lw=4)

        ax.hist(direct_band_gaps,alpha=0.5,bins=25,color='#00743F')
        ax.hist(indirect_band_gaps, alpha=0.5, bins=25,color='#F2A104')
        ax.set_xlabel("band gap (eV)")
        ax.set_ylabel("counts")
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax,width="45%", height="45%", loc=1)
        axins.pie([metal_counter,len(direct_band_gaps),len(indirect_band_gaps)],colors=['#80ADD7','#00743F','#F2A104'])

        if i*4+thick_id+1==1:
            from matplotlib.patches import Patch

            legend_elements = [Patch(facecolor='#80ADD7', edgecolor='k', label='Metal'),
                                Patch(facecolor='#00743F', edgecolor='k', label='Direct Gap'),
                                Patch(facecolor='#F2A104', edgecolor='k', label='Indirect Gap')]
            ax.legend(handles=legend_elements, loc=2, fontsize=50, ncol=1)

        #if i*4+thick_id+1==1:
        #    ax.set_title('$A^{I}B^{II}_{M}X_{3}$')
        #if i*4+thick_id+1==5:
        #    ax.set_title('$A^{I}B^{II}_{TM}X_{3}$')
        #if i*4+thick_id+1==9:
        #    ax.set_title('$A^{II}B^{IV}C_{3}$')
        #if i*4+thick_id+1==13:
        #    ax.set_title('$A^{I}B^{X}C_{3}$')

#plt.tight_layout()
plt.savefig('bandgap_stats.png',bbox_inches='tight')
