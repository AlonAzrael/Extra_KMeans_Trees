

# from libcpp cimport bool
# from libc.string cimport memset
from libc.stdlib cimport malloc, free

import plyvel

from collections import deque
from cymem.cymem cimport Pool

import numpy as np
cimport numpy as np
np.import_array()

ctypedef np.npy_float32 DTYPE_t
ctypedef np.npy_float64 DOUBLE_t
ctypedef np.uint_t UINT_t
ctypedef np.uint32_t UINT32_t
ctypedef np.npy_intp SIZE_t

from numpy import random

UINT = np.uint

# from scipy.sparse import csr_matrix, lil_matrix, csc_matrix, find as find_nonzero_indices

"""
ExtraKMeansTree Algorithm
======================================================
"""

cdef class ExtraKMeansTreeNode(object):

    cdef public UINT_t is_leaf
    cdef public list child_node_list
    cdef public UINT_t row_index
    cdef public ExtraKMeansTreeNode parent
    cdef public np.ndarray cluster_data_index_arr

    def __cinit__(self, UINT_t K, UINT_t row_index=0):
        self.is_leaf = 0
        self.child_node_list = [0]*K
        self.row_index = row_index
        self.cluster_data_index_arr = None

    cdef append_child(self, ExtraKMeansTreeNode node, UINT_t k_index):
        """
        Parameters
        ----------
        k_index int
            some k

        child_row_index: int
            index point to data_smat, as this row is for this k_index
        """
        self.child_node_list[k_index] = node

    cdef set_parent(self, ExtraKMeansTreeNode node):
        self.parent = node

    cdef to_leaf(self, np.ndarray[UINT_t, ndim=1] cluster_data_index_arr):
        self.cluster_data_index_arr = cluster_data_index_arr
        self.is_leaf = 1

"""
efficient random choice
======================================================
"""

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF

cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    return seed[0] % (<UINT32_t>RAND_R_MAX + 1)


cdef inline SIZE_t rand_int(SIZE_t low, SIZE_t high,
                            UINT32_t* random_state) nogil:
    """Generate a random integer in [0; end)."""
    return low + our_rand_r(random_state) % (high - low)

cdef np.ndarray rand_choice_index(UINT_t N, UINT_t K):
    cdef np.ndarray[UINT_t, ndim=1] k_index_arr = np.zeros(K, dtype=UINT)
    cdef UINT_t step = N/K
    cdef SIZE_t i = 0 
    cdef SIZE_t j = i + step
    cdef SIZE_t index = 0
    while j <= N:
        # k_index_arr[index] = rand_int(i, j, &random_state)
        k_index_arr[index] = random.randint(i,j)
        i = j
        j = i + step
        index += 1
        
    return k_index_arr

cdef struct CSRMatrix:
    UINT_t* values
    UINT_t* row_start
    UINT_t* col_idx
    UINT_t n_row
    UINT_t n_value

cdef print_csr_matrix(CSRMatrix* csr_matrix):
    for i in range(csr_matrix.n_row+1):
        print csr_matrix.row_start[i],
    print ""
    for i in range(csr_matrix.n_value):
        print csr_matrix.col_idx[i],

cdef CSRMatrix* load_user_csr(str name="./db/levelgraph-test", UINT_t n_user=0, UINT_t n_item=0, UINT_t n_relationship=0):
    
    # get db

    db = plyvel.DB(name, create_if_missing=True)

    # temp var declare
    cdef Pool mem = Pool()
    cdef UINT_t i = 0
    cdef str key
    cdef list temp_list
    cdef UINT_t temp_id
    cdef UINT_t new_user_id = 0
    cdef UINT_t new_item_id = 0
    cdef UINT_t cur_user_id = 100000 # it should be impossible!

    # read all relationship
    cdef UINT_t* row_start = <UINT_t*>malloc((n_user+1)*sizeof(UINT_t))
    cdef UINT_t* col_idx = <UINT_t*>malloc((n_relationship)* sizeof(UINT_t))
    
    i = 0
    for key in db.iterator(include_value=False):
        new_user_id, new_item_id = np.fromstring(key, dtype=UINT)
        # print new_user_id, new_item_id
        if new_user_id != cur_user_id:
            row_start[new_user_id] = i
            cur_user_id = new_user_id
        col_idx[i] = new_item_id
        i += 1
    row_start[n_user] = i

    del db

    cdef CSRMatrix* user_csr = <CSRMatrix*>malloc(1* sizeof(CSRMatrix))
    user_csr.row_start = row_start
    user_csr.col_idx = col_idx
    user_csr.n_value = n_relationship
    user_csr.n_row = n_user

    # debug 
    if 0:
        print_csr_matrix(user_csr)

    # convert list to big array

    # cdef UINT_t* user_ptr_arr = <UINT_t*>calloc(n_user+1, sizeof(UINT_t))
    # cdef UINT_t* user_item_id_arr = <UINT_t*>calloc(n_relationship, sizeof(UINT_t))

    # for new_user_id, temp_list in enumerate(user_list):
    #     user_ptr_arr[user_id] = i
    #     for new_item_id in temp_list:
    #         user_item_id_arr[i] = new_item_id
    #         i += 1
    # user_ptr_arr[n_user] = i

    # del user_list

    # cdef UINT_t* item_ptr_arr = <UINT_t*>calloc(n_item+1, sizeof(UINT_t))
    # cdef UINT_t* item_user_id_arr = <UINT_t*>calloc(n_relationship, sizeof(UINT_t))
    # i = 0
    # for new_item_id, temp_list in enumerate(item_list):
    #     item_ptr_arr[new_item_id] = i
    #     for new_user_id in temp_list:
    #         item_user_id_arr[i] = new_user_id
    #         i += 1
    # item_ptr_arr[n_item] = i

    # del item_list

    return user_csr


cdef ExtraKMeansTreeNode build_extrakmeanstree(CSRMatrix* user_csr, UINT_t K=2, UINT_t max_leaf_size=0):
    """
    Parameters
    ----------
    data: scipy.csr_matrix
        csr_matrix is efficient for matrix dot.

    Warning
    -------
    it would be a issue when a big K is set, that is, there could be some relatively small clusters which result in a bad cluster effect.
    So dont set K too big, number below 10 is suggested.
    And also better give a big max_leaf_size, like 500.
    """
    if max_leaf_size == 0:
        max_leaf_size = 100*K+1
    assert max_leaf_size > K

    # unpack user_item_sparse_matrix
    cdef UINT_t n_user = user_csr.n_row
    cdef UINT_t* user_row_start = user_csr.row_start
    cdef UINT_t* user_col_idx = user_csr.col_idx

    # cdef Pool mem = Pool()
    cdef np.ndarray[UINT_t, ndim=1] user_index_arr = np.arange(n_user, dtype=UINT)
    cdef ExtraKMeansTreeNode root_node = ExtraKMeansTreeNode(K, 0)

    user_index_arr_deque = deque()
    user_index_arr_deque.append(user_index_arr)

    node_queue = deque()
    node_queue.append(root_node)

    # type declare
    cdef ExtraKMeansTreeNode cur_node
    cdef UINT_t cur_n_user = 0
    cdef np.ndarray[UINT_t, ndim=1] cur_user_index_arr
    cdef np.ndarray[UINT_t, ndim=1] k_center_user_index_arr
    cdef list next_user_index_arr
    cdef np.ndarray[UINT_t, ndim=1] new_user_index_arr
    cdef ExtraKMeansTreeNode new_node

    # temp var 
    cdef UINT_t cur_user_index = 0
    cdef UINT_t k_row_index = 0
    cdef UINT_t k_index = 0
    cdef UINT_t k_user_index = 0
    cdef UINT_t cur_user_start = 0
    cdef UINT_t cur_user_end = 0
    cdef UINT_t kth_center_start = 0
    cdef UINT_t kth_center_end = 0
    cdef UINT_t best_k_val = 0
    cdef UINT_t best_k_index = 0
    cdef UINT_t cur_k_val = 0
    cdef UINT_t temp_user_cursor = 0
    cdef UINT_t move_flag = 0
    cdef UINT_t terminate_flag = 0

    cdef UINT_t counter = 0
    while len(node_queue)>0:
        cur_node = node_queue.popleft()
        cur_user_index_arr = user_index_arr_deque.popleft()
        cur_n_user = cur_user_index_arr.shape[0]

        # reach the leaf size threshold
        if cur_n_user <= max_leaf_size :
            cur_node.to_leaf(cur_user_index_arr)
            # if leaf_callback is not None:
            #     leaf_callback(cur_node)

            # print  "leaf:", cur_user_index_arr
            continue

        # print "parent:",cur_user_index_arr

        """
        k_center_smat like this:
        k_index: data_row
        k0: [1,0,1,...]
        k1: [1,0,0,...]
        k2: [0,1,1,...]
        ...
        """
        k_center_user_index_arr = random.choice(cur_user_index_arr, K, replace=False)
        # print "k center",k_center_user_index_arr

        # k_center_user_index_arr = rand_choice_index(cur_n_user, K)

        next_user_index_arr = [[] for _ in range(K)]

        for cur_row_index, cur_user_index in enumerate(cur_user_index_arr):
            cur_user_start = user_row_start[cur_user_index]
            cur_user_end = user_row_start[cur_user_index+1]
            best_k_val = 0
            best_k_index = 0

            # get which k cur user belong to
            for k_index, k_row_index in enumerate(k_center_user_index_arr):
                k_user_index = k_row_index
                kth_center_start = user_row_start[k_user_index]
                kth_center_end = user_row_start[k_user_index+1]

                cur_k_val = 0

                temp_user_cursor = cur_user_start
                move_flag = 0
                terminate_flag = 0
                
                while 1:
                # for i in range(100):

                    # terminate
                    if terminate_flag:
                        break
                    
                    # move user cursor
                    if move_flag == 0:
                        temp_kth_center_col_id = user_col_idx[kth_center_start]
                        while 1:
                            if temp_user_cursor >= cur_user_end :
                                terminate_flag = 1
                                break

                            temp_user_col_id = user_col_idx[temp_user_cursor]
                            if temp_user_col_id == temp_kth_center_col_id:
                                cur_k_val += 1
                                temp_user_cursor += 1
                            elif temp_user_col_id < temp_kth_center_col_id:
                                temp_user_cursor += 1
                            else:
                                kth_center_start += 1
                                move_flag = 1
                                break
                        
                    # move kth center
                    else:
                        temp_user_col_id = user_col_idx[temp_user_cursor]
                        while 1:
                            if kth_center_start >= kth_center_end:
                                terminate_flag = 1
                                break

                            temp_kth_center_col_id = user_col_idx[kth_center_start]
                            if temp_kth_center_col_id < temp_user_col_id:
                                kth_center_start += 1
                            elif temp_kth_center_col_id == temp_user_col_id:
                                cur_k_val += 1
                                kth_center_start += 1
                            else:
                                temp_user_cursor += 1
                                move_flag = 0
                                break
                else:
                    print "big while error"
                    raise Exception("big while error")

                # debug
                # print cur_user_index, k_row_index, cur_k_val

                if cur_k_val > best_k_val:
                    best_k_index = k_index
                    best_k_val = cur_k_val

            # assign cur user to best kth-center
            next_user_index_arr[best_k_index].append(cur_user_index)

        # print "next_user_index_arr", next_user_index_arr


        """
        cluster_mat like this:
        row_index: best_k_index
        r0: [0]
        r1: [1]
        r2: [3]
        ...
        """

        # finish each k
        for k_index,k_user_index in enumerate(k_center_user_index_arr):
            new_user_index_arr = np.asarray(next_user_index_arr[k_index], dtype=UINT)
            
            user_index_arr_deque.append(new_user_index_arr)
            # print "child: ",new_user_index_arr

            new_node = ExtraKMeansTreeNode(K, k_user_index)
            cur_node.append_child(new_node, k_index)
            new_node.set_parent(cur_node)
            node_queue.append(new_node)

        # if counter == 1:
        #     break
        # else:
        #     counter += 1

    return root_node

# def build_multi_extrakmeanstree(n_trees, data_smat, data_index_arr, k, max_leaf_size):
#     root_node_list = [0]*n_trees
#     for i in xrange(n_trees):
#         root_node = build_extrakmeanstree(data_smat, data_index_arr, k, max_leaf_size)
#         root_node_list.append(root_node)

#     return root_node_list

cpdef ExtraKMeansTreeNode load_and_build(str db_name, UINT_t n_user, UINT_t n_item, UINT_t n_relationship, UINT_t K=2, UINT_t max_leaf_size=20): 
    cdef CSRMatrix* user_csr = load_user_csr(db_name, n_user, n_item, n_relationship)
    # return None
    # print_csr_matrix(user_csr)

    cdef ExtraKMeansTreeNode root_node = build_extrakmeanstree(user_csr, K, max_leaf_size)

    return root_node


# if __name__ == '__main__':
#     test_build_tree()


