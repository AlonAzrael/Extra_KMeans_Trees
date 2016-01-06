

import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs":np.get_include()})

import extra_kmeans_tree as EXT
import extra_kmeans_tree_proto as EXTpt
import extra_kmeans_tree_optimal as EXTopm

import time

from levelgraph import LevelGraph

from scipy.sparse import csr_matrix, lil_matrix, csc_matrix, find as find_nonzero_indices
UINT = np.uint

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

    n_user = 100000
    n_item = 20
    data = np.random.randint(0,2,size=(n_user,n_item))
    data = np.asarray(data, dtype=UINT)

    smat = csr_matrix(data, shape=(n_user, n_item), dtype=UINT)
    # row_index_arr,column_index_arr,_ = find_nonzero_indices(smat)
    # relationships = zip(row_index_arr, column_index_arr)
    return data, np.arange(n_user,dtype=UINT)

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

    # module = EXTopm
    module = EXTpt
    # module = EXT

    n_user=10000
    n_item=200
    n_relationship=599894
    

    start_time = time.time()

    LG = LevelGraph(init=False)
    smat = LG.load_data(n_user=n_user, n_item=n_item, n_relationship=n_relationship)
    # print smat.indptr
    # print smat.indices
    del LG
    print "load smat"

    root_node = module.build_extrakmeanstree(smat, None, K=5, max_leaf_size=200, leaf_callback=None)
    # root_node = module.load_and_build(db_name="./db/levelgraph-test", n_user=n_user, n_item=n_item, n_relationship=n_relationship, K=5, max_leaf_size=200)

    elapsed_time = time.time() - start_time
    print elapsed_time

    # print_extrakmeanstree(root_node)


def main():
    node = EXTopm.ExtraKMeansTreeNode(1,1)
    # print node
    test_build_tree()


if __name__ == '__main__':
    main()


