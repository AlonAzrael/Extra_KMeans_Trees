

import time

import numpy as np
from numpy import random

from scipy.sparse import csr_matrix, lil_matrix, csc_matrix, find as find_nonzero_indices

UINT = np.uint

def gen_big_toy_dataset(n_user, n_item):
    mat = random.choice(2, n_user*n_item, p=[0.7,0.3]).reshape((n_user,n_item))
    # raw_input("pause")

    return mat

def bench_framework(data, callback=None):
    n_row = data.shape[0]
    timer = 0
    K = 5
    n_sample_row = n_row/2

    row_set_arr = np.asarray([set() for i in xrange(n_row)])
    user_k_arr = np.zeros(n_row, dtype=UINT)
    for i,row in enumerate(data):
        row_set = row_set_arr[i]
        for j,col in enumerate(row):
            if col:
                row_set.add(j)

    start_time = time.time()
    mode = "set"
    for i in xrange(100):
        k_index_arr = random.choice(np.arange(n_row), K, replace=False)
        sample_index_arr = random.choice(np.arange(n_row), n_sample_row, replace=False)

        # np op
        if mode == "np":
            k_data = data[k_index_arr]
            sample_data = data[sample_index_arr]

            result = np.dot(sample_data, k_data.T)
            best_k = np.argmax(result, axis=1)

            # print best_k

        elif mode == "set" :
            # set op
            for row_index in sample_index_arr:
                user_row_set = row_set_arr[row_index]
                temp_k = 0
                cur_k = 0
                cur_k_index = 0
                for k_index,k_row_index in enumerate(k_index_arr):
                    
                    row_set = row_set_arr[k_row_index]
                    for item_id in row_set:
                        if item_id in user_row_set:
                            temp_k += 1
                    
                    if temp_k > cur_k:
                        cur_k_index = k_index
                        cur_k = temp_k
                    temp_k = 0

                user_k_arr[row_index] = cur_k_index

        # print user_k_arr[sample_index_arr]

    elapsed_time = time.time() - start_time
    print elapsed_time

def bench_np_ndarray():
    random.int()

if __name__ == '__main__':
    data = gen_big_toy_dataset(10000,30)
    bench_framework(data)


