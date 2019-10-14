import numpy as np
import scipy.sparse as sp

#import tensorflow as tf
from input_data import load_data
from preprocessing import (construct_feed_dict, mask_test_edges,
                           preprocess_graph, sparse_to_tuple)

adj, features = load_data('cora')
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)

print(adj.nnz)
print(adj_train.nnz)
#print(features.shape)
'''
a = tf.constant([[1,2,2],[1,2,3]],tf.float32)
ses = tf.Session()
x = tf.transpose(a)
y = tf.matmul(a, x)
ys = tf.nn.sigmoid(y)
print(ses.run(a))
print(ses.run(x))
print(ses.run(y))
print(ses.run(ys))


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

adj = np.array([[1,2,3],[2,1,1],[3,1,1]])
print(adj)
adj = adj - \
    sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
#adj.eliminate_zeros()
#print(sp.dia_matrix(adj).toarray())
adj_triu = sp.triu(adj)
adj_tuple = sparse_to_tuple(adj_triu)
edges = adj_tuple[0]
edges_all = sparse_to_tuple(adj)[0]
print(edges)
print(edges_all)
'''
