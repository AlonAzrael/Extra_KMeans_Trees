
import plyvel

import numpy as np
from numpy import random

UINT = np.uint

from scipy.sparse import csr_matrix, lil_matrix, csc_matrix, find as find_nonzero_indices

class LevelRec():

    def __init__(self, name="./db/levelrec-test", init=True):
        if init:
            try:
                plyvel.destroy_db(name)
            except:
                pass
        self.db = plyvel.DB(name, create_if_missing=True)

    def set_data(self, data_smat):
        self.data_smat = data_smat

        # save all user's rec_item
        self.all_user_rec_item_dict = [{}]*data_smat.shape[0]

    def get_data(self):
        return self.data_smat

    def update_cluster_user_rec_item(self, cluster_user_id_arr):
        data_smat = get_data_smat()
        all_user_rec_item_dict = self.all_user_rec_item_dict

        cluster_user_data_smat = data_smat[cluster_user_id_arr]
        user_id_arr, item_id_arr, _ = find_nonzero_indices(cluster_user_data_smat)

        cluster_user_dict = {}
        for user_id in cluster_user_id_arr:
            cluster_user_dict[user_id] = set()
        
        item_set = set(item_id_arr)
        item_count_dict = {}
        for i,item_id in enumerate(item_id_arr):
            try:
                item_count_dict[item_id] += 1
            except KeyError:
                item_count_dict[item_id] = 1

            user_item_set = cluster_user_dict[user_id_arr[i]]
            user_item_set.add(item_id)
            

        for user_id in cluster_user_id_arr:
            user_item_set = cluster_user_dict[user_id]
            user_new_item_set = item_set - user_item_set

            user_rec_item_dict = all_user_rec_item_dict[user_id]

            for item_id in user_new_item_set:
                weight = item_count_dict[item_id]
                try:
                    user_rec_item_dict[item_id] += weight
                except KeyError:
                    user_rec_item_dict[item_id] = weight

    def write_all_user_rec_item(self):
        all_user_rec_item_dict = self.all_user_rec_item_dict
        wb = self.db.write_batch()

        temp_arr = np.asarray([0,0],dtype=UINT)
        for user_id, user_rec_item_dict in enumerate(all_user_rec_item_dict):

            for item_id,weight in user_rec_item_dict.items():
                temp_arr[0] = user_id
                temp_arr[1] = item_id
                key = temp_arr.tostring()
                value = str(weight)

                wb.put(key, value)

        wb.write()

"""
unittest
======================================================
"""





