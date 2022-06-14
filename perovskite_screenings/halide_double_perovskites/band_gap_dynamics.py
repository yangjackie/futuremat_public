import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.pylab as pylab

from core.external.vasp.anharmonic_score import AnharmonicScore

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
params = {'legend.fontsize': '10',
          #'figure.figsize': (8, 9),
          'axes.labelsize': 15,
          'axes.titlesize': 15,
          'xtick.labelsize': 12,
          'ytick.labelsize': 10}
#         'axes.prop_cycle':cycler.cycler('color',color)}
pylab.rcParams.update(params)


def gga_hybrid_benchmark(time_step=1):
    hybrid_gaps, pbe_gaps = get_band_gaps()

    times = [i * time_step for i in range(len(pbe_gaps))]
    differences = [hybrid_gaps[i] - pbe_gaps[i] for i in range(len(pbe_gaps))]

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
    ax1.plot(times, pbe_gaps, 'b-')
    ax2.plot(times, hybrid_gaps, 'r-')
    ax3.plot(times, differences, 'm-')

    ax3.set_xlabel('Time (fs)')
    ax1.set_ylabel("$E_{g}^{PBE}(t)$ (eV)")
    #ax1.set_ylim([min(pbe_gaps) - 0.02, min(pbe_gaps) + 0.2])
    ax2.set_ylabel("$E_{g}^{HSE}(t)$ (eV)")
    #ax2.set_ylim([min(hybrid_gaps) - 0.01, min(hybrid_gaps) + 0.2])
    ax3.set_ylabel("$E_{g}^{HSE}(t)-E_{g}^{PBE}(t)$ (eV)")
    #ax3.set_ylim([min(differences) - 0.02, min(differences) + 0.2])
    plt.tight_layout()

    plt.savefig('bandgap_dynamics_benchmark.pdf')


def get_band_gaps():
    all_data = glob.glob('gap_dynamics_300K*.dat')
    pbe_gaps = []
    hybrid_gaps = []
    for i in range(len(all_data)):
        data = open('gap_dynamics_300K_' + str(i) + '.dat', 'r')
        for l in data.readlines():
            pbe_gaps.append(float(l.split()[1]))
            hybrid_gaps.append(float(l.split()[2]))
    print("averaged gap is "+str(np.mean(hybrid_gaps)))
    return hybrid_gaps, pbe_gaps

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def gap_dephasing_function(gap_list, factor=1):
    max_lag = 100
    mean = np.average(gap_list[1])
    gap_corrs = []
    lags = []

    index_dict = {}
    for i, value in enumerate(gap_list[0]):
        index_dict[value] = i

    for lag in range(max_lag + 1):
        gap_corr = 0
        N_lag = 0
        for start_frame in gap_list[0]:
            if start_frame + lag in gap_list[0]:
                gap_corr += (gap_list[1][start_frame] - mean) * (gap_list[1][start_frame + lag] - mean)
                N_lag += 1

        if lag == 0:
            gap_corr_zero = gap_corr

        gap_corr = gap_corr / (max_lag * factor)
        gap_corrs.append(gap_corr)
        lags.append(lag * factor)

    from scipy.interpolate import interp1d
    f2 = interp1d(lags, gap_corrs, kind='cubic')
    xnew = np.linspace(0, lags[-1], num=300 * len(lags), endpoint=True)
    gap_corrs = f2(xnew)

    c_t_prime_primes = []
    factor = xnew[1] - xnew[0]
    for i in range(len(gap_corrs)):
        c_t_prime_prime = sum(gap_corrs[:i]) * factor
        c_t_prime_primes.append(c_t_prime_prime)

    import math
    c_t_primes = []
    # convert to ps, hbar given in eV
    hbar = 6.5821e-16 * 1e15
    for i in range(len(c_t_prime_primes)):
        c_t_prime = sum(c_t_prime_primes[:i]) * factor
        c_t_primes.append(math.exp(-c_t_prime / hbar ** 2))
    return xnew, c_t_primes

def band_gap_dynamics(time_step=1,max_lag=1800):
    hybrid_gaps, pbe_gaps = get_band_gaps()
    times = [i * time_step for i in range(len(pbe_gaps))]

    #get the band_gap_auto_correlation function
    mean = np.average(hybrid_gaps)
    Eg_corrs = []
    lags = []
    indicies = [i for i in range(len(hybrid_gaps))]
    for lag in range(max_lag+1):
        #print("This lag is "+str(lag))
        N_lag=0
        Eg_corr = 0
        for start_frame in indicies:
            if start_frame+lag in indicies:
                Eg_corr += (hybrid_gaps[start_frame] - mean) * (hybrid_gaps[start_frame+lag] - mean)
                N_lag +=1
        Eg_corr=Eg_corr/(max_lag)
        if lag==0:
            E_lag_zero=Eg_corr
            #acf_zero.append(E_lag_zero)
        if N_lag != 0:
            Eg_corrs.append(Eg_corr)
            lags.append(lag)

    from scipy.interpolate import interp1d
    f2 = interp1d(lags,Eg_corrs, kind='cubic')
    xnew = np.linspace(0, lags[-1], num=20*len(lags), endpoint=True)

    Eg_corrs=f2(xnew)
    #plt.figure(figsize=(5,16))

    # ============================================
    # Decoherence functions
    # ============================================
    lags, c_t_primes = gap_dephasing_function([range(len(hybrid_gaps)), hybrid_gaps])

    minus_lag = [-1 * l for l in lags[1:]]
    minus_c_t_primes = [c for c in c_t_primes[1:]]

    for j in range(len(lags)):
        minus_lag.append(lags[j])
        minus_c_t_primes.append(c_t_primes[j])
    x = np.array(minus_lag)
    y = np.array(minus_c_t_primes)
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(gaus, x, y, p0=[1, mean, sigma])


    print('Dephasing time: ', popt[-1])
    print('U_Eg(0): ',Eg_corrs[0])

    print("Get anharmonic scores")
    from phonopy.interface.calculator import read_crystal_structure, write_crystal_structure
    from phonopy import Phonopy
    unitcell, _ = read_crystal_structure('../CONTCAR', interface_mode='vasp')
    supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix)
    phonon.generate_displacements()
    write_crystal_structure('./SPOSCAR', phonon.supercell)

    scorer = AnharmonicScore(md_frames=glob.glob('./vasprun_prod*.xml'), ref_frame='./SPOSCAR',
                             force_constants='../force_constants.hdf5', unit_cell_frame='./SPOSCAR',
                             primitive_matrix='auto')
    # force_sets_filename='FORCE_SETS')
    sigma, _ = scorer.structural_sigma(return_trajectory=True)


    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,figsize=(16,5))
    ax1.plot(times,hybrid_gaps,'-',c='#258039')
    ax1.set_ylabel("$E_{g}^{HSE}(t)$ (eV)",color='#258039')
    ax1.tick_params(axis='y', labelcolor='#258039')

    ax12 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = '#FDD20EFF'
    ax12.set_ylabel('kernel similarity', color=color)  # we already handled the x-label with ax1
    ax12.plot(times, sigma[:2000], '-', color=color)
    ax12.set_ylabel("$\\sigma^{(2)}(t)$", color=color)
    ax12.tick_params(axis='y', labelcolor=color)

    ax1.set_xlabel("Time (fs)")
    ax1.set_xlim([0, 2000])

    ax2.plot(xnew,Eg_corrs,'-',lw=2, c='#CB0000')
    ax2.plot(xnew,[0 for _ in xnew],'k--')
    ax2.set_ylabel("$u_{E_{g}}(\\tau)$ (eV$^{2}$)")
    ax2.set_xlabel("$\\tau$ (fs)")
    ax2.set_xlim([0, 1750])

    ax3.plot(lags, c_t_primes, '-', c='#F0810F', lw=2)
    ax3.set_ylabel("$D_{E_{g}}(t)$ ")
    ax3.set_xlabel("$t$ (fs)")
    ax3.set_xlim([0,15])
    plt.tight_layout()
    plt.savefig('bandgap_dynamics.pdf')

def sigma_gap_correlation():
    import glob,os
    all_systems = glob.glob("dpv_*6")
    cwd=os.getcwd()

    all_hybrid_gaps = []
    all_sigmas=[]

    for i,system in enumerate(all_systems):

        print('System '+str(i)+'/'+str(len(all_systems)))
        os.chdir(system)
        try:
            os.chdir('MD')

            hybrid_gaps, pbe_gaps = get_band_gaps()
        except:
            os.chdir(cwd)
            continue

        print("Get anharmonic scores")
        from phonopy.interface.calculator import read_crystal_structure, write_crystal_structure
        from phonopy import Phonopy
        unitcell, _ = read_crystal_structure('../CONTCAR', interface_mode='vasp')
        supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix)
        phonon.generate_displacements()
        write_crystal_structure('./SPOSCAR', phonon.supercell)

        scorer = AnharmonicScore(md_frames=glob.glob('./vasprun_prod*.xml'), ref_frame='./SPOSCAR',
                                 force_constants='../force_constants.hdf5', unit_cell_frame='./SPOSCAR',
                                 primitive_matrix='auto')
        # force_sets_filename='FORCE_SETS')
        sigma, _ = scorer.structural_sigma(return_trajectory=True)

        hybrid_gaps = hybrid_gaps[:2000]
        #hybrid_gaps = [k-np.mean(hybrid_gaps) for k in hybrid_gaps]
        sigma = sigma[:2000]

        for k in range(len(hybrid_gaps)):
            all_hybrid_gaps.append(hybrid_gaps[k])
            all_sigmas.append(sigma[k])

        os.chdir(cwd)

    plt.scatter(all_sigmas,all_hybrid_gaps,marker='o',c='#CB0000',alpha=0.5)
    plt.xlabel('$\\sigma^{(2)}$')
    plt.ylabel('$E_{g}(t)-\\langle E_{g}(t)\\rangle$')
    plt.tight_layout()
    plt.savefig('sigma_gap_correlations.pdf')

if __name__ == "__main__":
    #gga_hybrid_benchmark()
    #band_gap_dynamics()
    #sigma_gap_correlation()
    get_band_gaps()