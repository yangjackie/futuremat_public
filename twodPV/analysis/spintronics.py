from ase.db import connect

from matplotlib import rc
rc('text', usetex=True)

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

params = {'legend.fontsize': 60,
          'figure.figsize': (6, 5),
          'axes.labelsize': 80,
          'axes.titlesize': 60,
          'xtick.labelsize': 70,
          'ytick.labelsize': 70}

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
plt.figure(figsize=(76,16))

colors=['#192E5B','#1D65A6','#00743F','#F2A104']

for i in range(len(A_site_list)):
    for thick_id,thick in enumerate([3,5,7,9]):
        polarisations = []
        for a in A_site_list[i]:
            for orient in ['100','110','111']:
                for term in termination_types[orient]:

                    for b in B_site_list[i]:
                        for c in C_site_list[i]:
                            system_name = a + b + c
                            uid = system_name + '3_' + str(orient) + "_" + str(term) + "_" + str(thick)
                            row = None
                            row = db.get(selection=[('uid', '=', uid)])
                            s = None
                            if row is not None:
                                try:
                                    up = abs(row.key_value_pairs['spin_up_dos_at_ef'])
                                    down = abs(row.key_value_pairs['spin_down_dos_at_ef'])
                                    s = (up-down)/(up+down)
                                    #if abs(s)<1.1:
                                    polarisations.append(abs(s))
                                    print(uid,s)
                                except KeyError:
                                    pass
        ax = plt.subplot(1,4, i+1)
        #ax.hist(polarisations,alpha=0.5,bins=25,density=True)

        y_grid = np.linspace(-0.2,1.2,100)
        pdf = kde_scipy(np.array(polarisations), y_grid, bandwidth=0.05)
        ax.plot(y_grid, pdf,'-', c=colors[thick_id], lw=8, label="$n="+str(thick)+"$")
        ax.set_xlabel('$s$')
        ax.set_ylabel('$\\mathcal{P}(s)$')
    if i==0:
        ax.legend()
        #ax.set_xlim([0,1])
        #ax.set_yscale('log')
    if i==0:
        ax.set_title('$A^{I}B^{II}_{M}X_{3}$')
    if i==1:
        ax.set_title('$A^{I}B^{II}_{TM}X_{3}$')
    if i==2:
        ax.set_title('$A^{II}B^{IV}C_{3}$')
    if i==3:
        ax.set_title('$A^{I}B^{X}C_{3}$')

plt.savefig('spin_polarisation_at_fermi.png',bbox_inches='tight')