


from collections import deque

import numpy as np
from numpy import random

from scipy.sparse import csr_matrix, csc_matrix, find as find_nonzero_indices


def gen_toy_dataset():
    dataset = random.randint(0,2,size=(3,10))
    print dataset
    return dataset

def as_sparse_matrix(mat):
    smat = csr_matrix(mat)
    smat = csc_matrix(mat)
    return smat

def op_sparse_matrix():
    mat = random.randint(0,2,size=(10,10))
    print mat
    smat = csr_matrix(mat)
    indices = find_nonzero_indices(smat)[1]
    


def benchmark_distance(smat):
    # k_mat = random.randint(0,2,size=(2,10))
    k_mat = smat[0:2]
    print k_mat
    dist_mat = smat*k_mat.T
    print dist_mat
    cluster_mat = np.argmax(dist_mat.toarray(), axis=1)
    print cluster_mat
    index_arr = np.where(cluster_mat == 0)[0]
    print cluster_mat[index_arr]



def main():
    op_sparse_matrix()

if __name__ == '__main__':
    main()

