


from collections import deque

import numpy as np
from numpy import random

UINT = np.uint

from scipy.sparse import csr_matrix, lil_matrix, csc_matrix, find as find_nonzero_indices

"""
ExtraKMeansTree Algorithm
======================================================
"""

class ExtraKMeansTreeNode(object):

    def __init__(self, K, row_index=-1):
        self.is_leaf = False
        self.child_node_list = [0]*K
        self.row_index = row_index

    def append_child(self, node, k_index):
        """
        Parameters
        ----------
        k_index int
            some k

        child_row_index: int
            index point to data_smat, as this row is for this k_index
        """
        self.child_node_list[k_index] = node

    def set_parent(self, node):
        self.parent = node

    def to_leaf(self, cluster_data_index_arr):
        self.cluster_data_index_arr = cluster_data_index_arr
        self.is_leaf = True


def build_extrakmeanstree(data_smat, data_index_arr=None, K=2, max_leaf_size=None, leaf_callback=None):
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
    if max_leaf_size is None:
        max_leaf_size = 100*K+1
    assert max_leaf_size > K

    if data_index_arr is None:
        data_index_arr = np.arange(data_smat.shape[0], dtype=UINT)

    root_node = ExtraKMeansTreeNode(K)

    data_smat_queue = deque()
    data_smat_queue.append(data_smat)

    data_index_arr_queue = deque()
    data_index_arr_queue.append(data_index_arr)

    node_queue = deque()
    node_queue.append(root_node)

    while len(node_queue)>0:
        cur_node = node_queue.popleft()
        cur_data_smat = data_smat_queue.popleft()
        cur_data_index_arr = data_index_arr_queue.popleft()
        cur_n_user = len(cur_data_index_arr)

        # reach the leaf size threshold
        if cur_n_user <= max_leaf_size:
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
        k_center_data_index_arr = random.choice(np.arange(cur_n_user, dtype=UINT), K, replace=False)
        k_center_data_smat = cur_data_smat[k_center_data_index_arr]

        """
        cluster_mat like this:
        row_index: best_k_index
        r0: [0]
        r1: [1]
        r2: [3]
        ...
        """
        dist_mat = (cur_data_smat*k_center_data_smat.T).toarray()
        # dist_mat = np.dot(cur_data_smat,k_center_data_smat.T)
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

def build_multi_extrakmeanstree(n_trees, data_smat, data_index_arr, k, max_leaf_size):
    root_node_list = [0]*n_trees
    for i in xrange(n_trees):
        root_node = build_extrakmeanstree(data_smat, data_index_arr, k, max_leaf_size)
        root_node_list.append(root_node)

    return root_node_list


"""
unittest
======================================================
"""

def gen_toy_dataset():
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

    smat = csr_matrix(data, shape=(n_user, n_item), dtype=UINT)
    # row_index_arr,column_index_arr,_ = find_nonzero_indices(smat)
    # relationships = zip(row_index_arr, column_index_arr)
    return smat, np.arange(n_user,dtype=UINT)

def print_extrakmeanstree(node, depth=0, indent=2):
    big_indent = "".join([" "]*indent)
    if node.is_leaf:
        print big_indent, "leaf: ", node.cluster_data_index_arr
    else:
        # if depth == 5:
        #     return
        print big_indent,"node(depth:{}): ".format(depth),node.row_index
        for child in node.child_node_list:
            print_extrakmeanstree(child, depth+1, indent+2)

def test_build_tree():

    def test_leaf_callback(leaf_node):
        # print leaf_node.cluster_data_index_arr
        pass

    smat, data_index_arr = gen_toy_dataset()
    root_node = build_extrakmeanstree(smat, data_index_arr, 2, leaf_callback=test_leaf_callback)
    print_extrakmeanstree(root_node)

if __name__ == '__main__':
    test_build_tree()


