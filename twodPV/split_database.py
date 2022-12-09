from ase.db import connect
import os
import sqlite3,json
from collect_data import populate_db

dbname = os.path.join(os.getcwd(), '2dpv.db')
all_uids = []
_db = sqlite3.connect(dbname)
cur = _db.cursor()
cur.execute("SELECT * FROM systems")
rows = cur.fetchall()

for row in rows:
    for i in row:
        if 'uid' in str(i):
            this_dict = json.loads(str(i))
            this_uid = this_dict['uid']
            all_uids.append(this_uid)

bulk_set = [id for id in all_uids if '_pm3m' in id] + [id for id in all_uids if '3_random_str_' in id]
twodPV_100_AO_set = [id for id in all_uids if '100_AO' in id and 'cell' not in id]
twodPV_100_BO2_set = [id for id in all_uids if '100_BO2' in id and 'cell' not in id]
twodPV_110_ABO_set = [id for id in all_uids if '110_ABO' in id and 'cell' not in id]
twodPV_110_O2_set = [id for id in all_uids if '110_O2' in id and 'cell' not in id]
twodPV_111_B_set = [id for id in all_uids if '111_B' in id and 'cell' not in id]
twodPV_111_AO3_set = [id for id in all_uids if '111_AO3' in id and 'cell' not in id]

set_dict={'100_AO':twodPV_100_AO_set,'100_BO2':twodPV_100_BO2_set,'110_ABO':twodPV_110_ABO_set,'110_O2':twodPV_110_O2_set,'111_B':twodPV_111_B_set,'111_AO3':twodPV_111_AO3_set}
set_dict={'110_O2':twodPV_110_O2_set}
original_db = connect(dbname)

"""
this_db = connect(os.path.join(os.getcwd(), '2dpv_set_bulk.db'))
original_db = connect(dbname)

for id in bulk_set:
    row = original_db.get(selection=[('uid', '=', id)])
    print('Populating ' + id)
    populate_db(this_db, row.toatoms(), row.key_value_pairs, row.data)

original_db = None
this_db = None
"""

for set in set_dict.keys():
    this_db = connect(os.path.join(os.getcwd(), '2dpv_set_'+set+'.db'))
    original_db = connect(dbname)

    for id in set_dict[set]:
        row = original_db.get(selection=[('uid', '=', id)])
        print('Populating '+id)
        populate_db(this_db,row.toatoms(),row.key_value_pairs,row.data)

    original_db = None
    this_db = None

