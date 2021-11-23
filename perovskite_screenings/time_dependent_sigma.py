from ase.db import connect
import math
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
from ase.geometry import *
from ase.neighborlist import *
rc('text', usetex=True)
import matplotlib.pylab as pylab
params = {'legend.fontsize': '14',
          'figure.figsize': (6, 5),
          'axes.labelsize': 24,
          'axes.titlesize': 24,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)

db = connect('./perovskites.db')

systems = ['CsPbI','CsSnI','CsPbBr','CsSnBr']
#systems = ['SrTiO','SrSnO','BaTiO','BaSnO']
colors =  ['#085f63', '#49beb7','#fccf4d', '#ef255f']

xs=[]
ys=[]
for k,s in enumerate(systems):

    uid = s + '_Pm3m'
    row = db.get(selection=[('uid', '=', uid)])
    y=row.data['sigma_300K']
    x=[i*2 for i in range(len(y))]
    xs.append(x)
    ys.append(y)

    print(s,np.average(y),np.std(y),row.key_value_pairs['band_gap'])
    #plt.plot(x,y,'.-',label=s+'$_{3}$',c=colors[k])

gs = gridspec.GridSpec(1, 2, width_ratios=[3.5, 1])
gs.update(wspace=0.025, hspace=0.07,bottom=0.2)
fig = plt.subplots(figsize=(7, 5.5))

ax = plt.subplot(gs[0])
for i in range(len(xs)):
    ax.plot(xs[i], ys[i], '.-', label=systems[i] + '$_{3}$', c=colors[i])

ax.legend(ncol=2)
ylim=ax.get_ylim()
xlim=ax.get_xlim()
ax.set_ylabel("$\\sigma(t)$")
ax.set_xlabel('Time (ps)')

ax = plt.subplot(gs[1])
for i in range(len(xs)):
    from scipy.stats import gaussian_kde
    def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
        # Kernel Density Estimation with Scipy
        # Note that scipy weights its bandwidth by the covariance of the
        # input data.  To make the results comparable to the other methods,
        # we divide the bandwidth by the sample standard deviation here.
        kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
        return kde.evaluate(x_grid)

    x_grid = np.linspace(ylim[0], ylim[1], 1000)
    pdf = kde_scipy(ys[i], x_grid, bandwidth=0.025)
    ax.plot(pdf / sum(pdf), x_grid, '-', lw=2, c=colors[i])
ax.set_ylim(ylim)
labels = [item.get_text() for item in ax.get_yticklabels()]
empty_string_labels = [''] * len(labels)
ax.set_yticklabels(empty_string_labels)
ax.set_xlabel('$p(\\sigma)$')
plt.tight_layout()
plt.savefig('sigma_trajectory.pdf')
