import os

from twodPV.collect_data import *
import argparse
from pymatgen.io.vasp.outputs import *

def collect_dielectric_bulk(db):
    cwd = os.getcwd()
    base_dir = cwd + '/relax_Pm3m/'
    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:
                    system_name = a + b + c
                    uid = system_name + '3_pm3m'

                    print("Working on "+uid)
                    kvp = {}
                    data = {}
                    kvp['uid'] = uid

                    dir = os.path.join(base_dir, system_name + "_Pm3m")
                    os.chdir(dir)

                    try:
                        with zipfile.ZipFile('./phonon_G.zip') as z:
                            with open("./OUTCAR_ph", 'wb') as f:
                                f.write(z.read("OUTCAR"))
                        f.close()
                        z.close()
                    except:
                        pass

                    if os.path.isfile("./OUTCAR_ph"):
                        outcar = Outcar('./OUTCAR_ph')
                        outcar.read_lepsilon_ionic()
                        print('dielectric ionic tensor')
                        print(outcar.dielectric_ionic_tensor)
                        data['dielectric_ionic_tensor'] = outcar.dielectric_ionic_tensor
                        print('Born Effective Charge')
                        print(outcar.born)
                        populate_db(db, None, kvp, data)
                        os.remove('./OUTCAR_ph')
                    os.chdir(cwd)



def collect_this(info_dict):
    import tarfile,shutil
    root_dir = os.getcwd()
    os.chdir(info_dict['base_dir'])

    to_continue=False
    system_name = info_dict['system_name']
    try:
        tf = tarfile.open(system_name + '.tar.gz', 'r')
        tf.extractall()
        to_continue = True
    except:
        logger.info(system_name + ' tar ball not working')
        return info_dict
    if to_continue:
        _this_dir = os.getcwd()
        collect_data = False
        try:
            os.chdir(system_name)
            collect_data = True
        except:
            pass

        if collect_data:
            try:
                with zipfile.ZipFile('./phonon_G.zip') as z:
                    with open("./OUTCAR_ph", 'wb') as f:
                        f.write(z.read("OUTCAR"))
                f.close()
                z.close()
            except:
                pass

            try:
                if os.path.isfile("./OUTCAR_ph"):
                    try:
                        outcar = Outcar('./OUTCAR_ph')
                        outcar.read_lepsilon_ionic()
                        #logger.info(uid + '\t' + str(outcar.dielectric_ionic_tensor))
                        #data['dielectric_ionic_tensor'] = outcar.dielectric_ionic_tensor
                        info_dict['dielectric_ionic_tensor'] = outcar.dielectric_ionic_tensor
                    except:
                        pass

                    try:
                        reader = VaspReader(input_location='./OUTCAR_ph')
                        freqs = reader.get_vibrational_eigenfrequencies_from_outcar()
                        #logger.info('phonon frequencies :' + str(freqs))
                        #data['gamma_phonon_freq'] = np.array(freqs)
                        info_dict['gamma_phonon_freq'] = np.array(freqs)
                    except:
                        pass
                    #populate_db(db, None, kvp, data)
                    os.remove('./OUTCAR_ph')
            except:
                pass

            os.chdir(_this_dir)

        try:
            shutil.rmtree(system_name)
        except:
            pass

        os.chdir(root_dir)

    return info_dict

def collect_dielectric_2D(db):
    import tarfile,shutil
    import multiprocessing
    cwd = os.getcwd()
    terminations = [['AO','BO2'],['ABO','O2'],['AO3','B']]

    to_process_raw=[]

    for orient_id, orient in enumerate(['100','110','111']):
        for term in terminations[orient_id]:
            base_dir = cwd + '/slab_'+orient+'_'+term+'_small'
            #os.chdir(base_dir)

            for i in range(len(A_site_list)):
                for a in A_site_list[i]:
                    for b in B_site_list[i]:
                        for c in C_site_list[i]:
                            for thick in [3,5,7,9]:

                                info_dict = {}
                                info_dict['base_dir'] = base_dir
                                system_name = a+b+c+'_'+str(thick)
                                info_dict['system_name'] = system_name
                                uid = a + b + c + '3_' + str(orient) + "_" + str(term) + "_" + str(thick)
                                info_dict['uid'] = uid
                                logger.info('gather '+uid)
                                to_process_raw.append(info_dict)

    to_process = []
    num_threads = 28
    for counter,item in enumerate(to_process_raw):
        to_process.append(item)
        if (len(to_process) == num_threads*2) or (counter == len(to_process_raw)-1):
            pool = multiprocessing.Pool(num_threads)
            p = pool.map_async(collect_this,to_process)
            results = p.get(999999)
            pool.terminate()

            for result in results:
                kvp = {}
                data = {}
                kvp['uid'] = result['uid']
                try:
                    data['dielectric_ionic_tensor'] = result['dielectric_ionic_tensor']
                    logger.info(result['uid'] + '\t' + str(result['dielectric_ionic_tensor']))
                except:
                    pass
                try:
                    data['gamma_phonon_freq'] = result['gamma_phonon_freq']
                    logger.info('phonon frequencies :'+str(result['gamma_phonon_freq']))
                except:
                    pass
                populate_db(db, None, kvp, data)

            to_process = []
                #                     to_continue=False
            #                     try:
            #                         tf = tarfile.open(system_name + '.tar.gz','r')
            #
            #                         if os.path.isdir('phonon_temp'):
            #                             shutil.rmtree('phonon_temp')
            #                         else:
            #                             os.mkdir('phonon_temp')
            #
            #                         for member in tf.getmembers():
            #                             if 'phonon_G.zip' in member.name:
            #                                 tf.extract(member,'phonon_temp')
            #
            #                         to_continue=True
            #
            #                     except:
            #                         logger.info(system_name + ' tar ball not working')
            #                         continue
            #
            #
            #                     if to_continue:
            #                         _this_dir = os.getcwd()
            #                         collect_data = False
            #                         try:
            #                            os.chdir('phonon_temp/'+system_name)
            #                            collect_data = True
            #                         except:
            #                             pass
            #
            #                         if collect_data:
            #                             data={}
            #                             kvp={}
            #                             uid = a+b+c + '3_' + str(orient) + "_" + str(term) + "_" + str(thick)
            #                             kvp['uid'] = uid
            #
            #                             try:
            #                                 with zipfile.ZipFile('./phonon_G.zip') as z:
            #                                     with open("./OUTCAR_ph", 'wb') as f:
            #                                         f.write(z.read("OUTCAR"))
            #                                 f.close()
            #                                 z.close()
            #                             except:
            #                                 pass
            #
            #                             try:
            #                                 if os.path.isfile("./OUTCAR_ph"):
            #                                     try:
            #                                         outcar = Outcar('./OUTCAR_ph')
            #                                         outcar.read_lepsilon_ionic()
            #                                         logger.info(uid+'\t'+str(outcar.dielectric_ionic_tensor))
            #                                         data['dielectric_ionic_tensor'] = outcar.dielectric_ionic_tensor
            #                                     except:
            #                                         pass
            #
            #                                     try:
            #                                         reader = VaspReader(input_location='./OUTCAR_ph')
            #                                         freqs = reader.get_vibrational_eigenfrequencies_from_outcar()
            #                                         logger.info('phonon frequencies :'+str(freqs))
            #                                         data['gamma_phonon_freq'] = np.array(freqs)
            #                                     except:
            #                                         pass
            #                                     populate_db(db, None, kvp, data)
            #                                     os.remove('./OUTCAR_ph')
            #                             except:
            #                                 pass
            #
            #                             os.chdir(_this_dir)
            #
            #                         try:
            #                             shutil.rmtree('phonon_temp')
            #                         except:
            #                             pass
            #                         try:
            #                             os.rmtree('phonon_temp')
            #                         except:
            #                             pass
            #
            # os.chdir(cwd)

if __name__=="__main__":
    dbname = os.path.join(os.getcwd(), '2dpv.db')
    db = connect(dbname)
    logger = setup_logger(output_filename='data_collector_static_dielectric.log')
    collect_dielectric_bulk(db)