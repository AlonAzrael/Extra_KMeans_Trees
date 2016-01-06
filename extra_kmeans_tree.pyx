

# from libcpp cimport bool
# from libcpp.deque cimport deque as cdeque

from collections import deque

# from cymem.cymem cimport Pool

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

from scipy.sparse import csr_matrix, lil_matrix, csc_matrix, find as find_nonzero_indices

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


cpdef ExtraKMeansTreeNode build_extrakmeanstree(data_smat, np.ndarray data_index_arr=None, UINT_t K=2, UINT_t max_leaf_size=0, leaf_callback=None):
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

    # cdef np.ndarray[UINT_t, ndim=1] data_index_arr

    if data_index_arr is None:
        data_index_arr = np.arange(data_smat.shape[0], dtype=UINT)

    cdef ExtraKMeansTreeNode root_node = ExtraKMeansTreeNode(K, 0)

    data_smat_queue = deque()
    data_smat_queue.append(data_smat)

    data_index_arr_queue = deque()
    data_index_arr_queue.append(data_index_arr)

    node_queue = deque()
    node_queue.append(root_node)

    # type declare
    cdef ExtraKMeansTreeNode cur_node
    cdef UINT_t cur_n_user = 0
    cdef np.ndarray[UINT_t, ndim=1] cur_data_index_arr
    cdef np.ndarray[UINT_t, ndim=1] k_center_data_index_arr
    cdef np.ndarray[UINT_t, ndim=2] dist_mat
    cdef np.ndarray best_k_index_arr
    cdef UINT_t k_index = 0
    cdef UINT_t pointer_index = 0
    cdef np.ndarray next_pointer_index_arr
    cdef np.ndarray[UINT_t, ndim=1] next_data_index_arr
    cdef UINT_t spliter_row_index
    cdef ExtraKMeansTreeNode new_node

    while len(node_queue)>0:
        cur_node = node_queue.popleft()
        cur_data_smat = data_smat_queue.popleft()
        cur_data_index_arr = data_index_arr_queue.popleft()
        cur_n_user = len(cur_data_index_arr)

        # reach the leaf size threshold
        if cur_n_user <= max_leaf_size :
            cur_node.to_leaf(cur_data_index_arr)
            if leaf_callback is not None:
                leaf_callback(cur_node)

            # print  "leaf:",cur_data_index_arr
            continue

        # print "parent:",cur_data_index_arr

        """
        k_center_smat like this:
        k_index: data_row
        k0: [1,0,1,...]
        k1: [1,0,0,...]
        k2: [0,1,1,...]
        ...
        """
        # k_center_data_index_arr = random.choice(np.arange(cur_n_user, dtype=UINT), K, replace=False)
        k_center_data_index_arr = rand_choice_index(cur_n_user, K)
        k_center_data_smat = cur_data_smat[k_center_data_index_arr]

        """
        cluster_mat like this:
        row_index: best_k_index
        r0: [0]
        r1: [1]
        r2: [3]
        ...
        """
        # dist_mat = (cur_data_smat*k_center_data_smat.T).toarray()
        dist_mat = np.dot(cur_data_smat,k_center_data_smat.T)
        best_k_index_arr = np.argmax(dist_mat, axis=1)

        # finish each k
        for k_index,pointer_index in enumerate(k_center_data_index_arr):
            next_pointer_index_arr = np.where(best_k_index_arr == k_index)[0]
            
            next_data_smat = cur_data_smat[next_pointer_index_arr]
            data_smat_queue.append(next_data_smat)
            
            next_data_index_arr = cur_data_index_arr[next_pointer_index_arr]
            data_index_arr_queue.append(next_data_index_arr)
            # print "child: ",next_data_index_arr

            spliter_row_index = cur_data_index_arr[pointer_index]
            new_node = ExtraKMeansTreeNode(K, spliter_row_index)
            cur_node.append_child(new_node, k_index)
            new_node.set_parent(cur_node)
            node_queue.append(new_node)

    return root_node

# def build_multi_extrakmeanstree(n_trees, data_smat, data_index_arr, k, max_leaf_size):
#     root_node_list = [0]*n_trees
#     for i in xrange(n_trees):
#         root_node = build_extrakmeanstree(data_smat, data_index_arr, k, max_leaf_size)
#         root_node_list.append(root_node)

#     return root_node_list



# if __name__ == '__main__':
#     test_build_tree()


