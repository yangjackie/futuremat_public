from twodPV.collect_data import *
import argparse
from core.utils.loggings import setup_logger


def this_data_batch(db, orientation='100', termination='AO', thick=3):
    cwd = os.getcwd()
    base_dir = cwd + '/slab_' + str(orientation) + '_' + str(termination) + '_small/'

    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:
                    system_name = a + b + c

                    kvp = {}
                    data = {}
                    uid = system_name + '3_' + str(orientation) + "_" + str(termination) + "_" + str(thick)
                    kvp['uid'] = uid
                    logger.info("Getting data for system  " + str(
                        uid) + ' from directory ' + base_dir + '/' + system_name + '_' + str(thick))
                    os.chdir(base_dir + '/' + system_name + '_' + str(thick))

                    ## get the two-dimensional electronic permitivity
                    # try:
                    #    data['e_polarizability_freq'] = get_geometry_corrected_electronic_polarizability()
                    # except:
                    #    logger.info("Cannot get frequency-dependent electronic polarizability")

                    logger.info("\n" + "============= out_of_plane_charge_polarisations ==============")
                    kvp = get_out_of_plane_charge_polarisations(kvp)

                    logger.info("\n" + "============= dos_related_properties ==============")
                    kvp = get_dos_related_properties(kvp)

                    logger.info("\n" + "============= band_structures_properties =============")
                    data = get_band_structures_properties(data)

                    logger.info("\n" + "====Summary of band structure information====")

                    for k in data['band_structure'].keys():
                        if isinstance(data['band_structure'][k], list):
                            for kk in range(len(data['band_structure'][k])):
                                for kkk in data['band_structure'][k][kk].keys():
                                    logger.info(str(k) + ' ' + str(kkk) + ' ' + str(data['band_structure'][k][kk][kkk]))
                        else:
                            logger.info(str(k) + ' ' + str(data['band_structure'][k]))

                    logger.info("\n")
                    populate_db(db, None, kvp, data)
                    os.chdir(cwd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch electronic data collector',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--orientation", type=str, default='100')
    parser.add_argument("--termination", type=str, default='AO')
    parser.add_argument("--thick", type=int, default=3)
    parser.add_argument("--db", type=str, default=os.getcwd() + '/2dpv.db',
                        help="Name of the database that contains the results of the screenings.")
    args = parser.parse_args()
    # We use absolute path because of chdir below!
    dbname = os.path.join(os.getcwd(), '2dpv.db')
    db = connect(dbname)
    logger = setup_logger(
        output_filename='data_collector_' + str(args.orientation) + '_' + str(args.termination) + '_' + str(
            args.thick) + '.log')
    logger.info('Established a sqlite3 database object ' + str(db))
    this_data_batch(db, orientation=args.orientation, termination=args.termination, thick=args.thick)
