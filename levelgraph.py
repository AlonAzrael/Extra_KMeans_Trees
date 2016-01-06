

import plyvel
import numpy as np
from numpy import random

from scipy.sparse import csr_matrix, lil_matrix, csc_matrix, find as find_nonzero_indices

UINT = np.uint

class LevelGraph(object):

    def __init__(self, name='./db/levelgraph-test', init=True):
        if init:
            try:
                plyvel.destroy_db(name)
            except:
                pass
        self.db = plyvel.DB(name, create_if_missing=True)

    def add_relationship_batch(self, relationship_list, weight_list=None):
        weight = "1"
        wb = self.db.write_batch()
        for row in np.asarray(relationship_list, dtype=UINT):
            key = row.tostring()
            value = weight
            wb.put(key, value)

        wb.write()
        
    def add_relationship(self, aid, bid, weight="1"):
        key = np.asarray([aid, bid], dtype=UINT).tostring()
        value = weight
        self.db.put(key, value)

    def load_data(self, n_user, n_item, n_relationship):
        
        row_index_arr = np.zeros(n_relationship, dtype=UINT)
        column_index_arr = np.zeros(n_relationship, dtype=UINT)
        data = np.ones(n_relationship, dtype=UINT)

        all_user_history_item_dict = [{}]*n_user

        i = 0
        for key in self.db.iterator(include_value=False):
            user_id, item_id = np.fromstring(key, dtype=UINT)
            # print user_id, item_id
            row_index_arr[i] = user_id
            column_index_arr[i] = item_id
            all_user_history_item_dict[user_id][item_id] = 1
            i += 1

        self.all_user_history_item_dict = all_user_history_item_dict

        smat = csr_matrix((data, (row_index_arr, column_index_arr)), shape=(n_user, n_item))
        self.data_smat = smat
        return smat

    def close_db(self):
        self.db.close()



"""
unittest
======================================================
"""

def gen_toy_dataset(n_user, n_item):
    mat = random.choice(2, n_user*n_item, p=[0.7,0.3]).reshape((n_user,n_item))
    # print mat
    smat = csr_matrix(mat)
    # print smat
    row_index_arr,column_index_arr,_ = find_nonzero_indices(smat)
    relationships = zip(row_index_arr, column_index_arr)
    return relationships

def gen_debug_dataset():
    data = [
        [1,1,1,1,0,0,0,0,0,0],
        [1,1,1,0,0,1,0,0,0,0],
        [1,0,0,1,0,1,1,0,0,0],
        [1,1,0,1,1,0,0,1,0,0],
        [0,0,1,1,1,1,0,0,0,0],
        [0,0,0,1,1,0,0,1,1,0],
        [0,0,0,1,0,1,1,1,0,1],
        [0,0,0,0,1,0,1,1,1,1],
        [0,0,1,0,0,0,1,0,1,1],
        [0,0,0,1,0,1,1,0,1,1],
    ]
    n_user = len(data)
    n_item = len(data[0])

    print n_user, n_item

    smat = csr_matrix(data, shape=(n_user, n_item), dtype=UINT)
    row_index_arr,column_index_arr,_ = find_nonzero_indices(smat)
    relationships = zip(row_index_arr, column_index_arr)
    return n_user, n_item, relationships

def test_levelgraph():
    mode = "init"
    if mode == "init":
        n_user = 10000
        n_item = 200
        relationships = gen_toy_dataset(n_user, n_item)
    elif mode == "debug":
        n_user, n_item, relationships = gen_debug_dataset()

    n_relationship = len(relationships)
    print "n_relationship", n_relationship

    lg = LevelGraph()
    lg.add_relationship_batch(relationships)
    smat = lg.load_data(n_user, n_item, n_relationship)
    print "reflect: ",smat

if __name__ == '__main__':
    test_levelgraph()


