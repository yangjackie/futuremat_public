from scipy.stats import gaussian_kde
import os
import tarfile
import shutil

import matplotlib
cmap = matplotlib.cm.get_cmap('BuPu')
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pylab as pylab
params = {'legend.fontsize': '14',
          'figure.figsize': (6,5),
          'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)

root_dir = os.getcwd()

temperatures = [200,300,400,500,600,700]

for temp in temperatures:
    print(temp)
    folder_name = 'MD_'+str(temp)
    folder_path = os.path.join(root_dir, folder_name)
    all_sigmas = []

    # tar_files = [file_name for file_name in os.listdir(folder_path) if file_name.endswith(".tar.gz")][:40]
    # path_holder = []
    #
    # for tar_count, tar_file in enumerate(tar_files):
    #     tar_file_path = os.path.join(folder_path, tar_file)
    #
    #     print(f"{tar_count}/{len(tar_files)}, Processing {tar_file_path}... for temperature "+str(temp)+' K')
    #     with tarfile.open(tar_file_path, 'r') as tar:
    #         tar.extractall(path=folder_path)
    #         path_holder.append(os.path.join(folder_path, tar_file[:-7]))
    #
    # vasprun_xmls = [i+'/vasprun.xml' for i in path_holder if os.path.isfile(i+'/vasprun.xml')]

    vasprun_xmls = [folder_path+'/vasprun.xml']
    scorer = AnharmonicScore(md_frames=vasprun_xmls, unit_cell_frame=root_dir+'/CONTCAR', ref_frame=root_dir+'/CONTCAR',
                             atoms=None,potim=1.0, mode_resolved=False,include_third_order=False,
                             force_constants=root_dir+'/force_constants.hdf5')
    sigma, time_stps = scorer.structural_sigma(return_trajectory=True)
    all_sigmas.append(sigma)
    smoothed_sigmas = kernel = gaussian_kde(all_sigmas)
    x=np.arange(0.9,1.1,0.001)
    plt.plot(x, kernel(x), '-', color=cmap(temp / 800), label=str(temp) + 'K')

    #clear up the untarred directories
    # for _path in path_holder:
    #     try:
    #         shutil.rmtree(os.path.join(_path))
    #     except:
    #         pass

plt.xlabel('$\\sigma(T)$')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('MD_sigma_distributions.pdf')
