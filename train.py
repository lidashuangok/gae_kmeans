from __future__ import division, print_function

import os
import time

import numpy
import scipy.sparse as sp
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from input_data import load_data
from model import GCNModelAE, GCNModelVAE
from optimizer import OptimizerAE, OptimizerVAE
from clustering_metric import clustering_metrics
from preprocessing import (construct_feed_dict, mask_test_edges,
                           preprocess_graph, sparse_to_tuple)
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""




# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model
dataset_str = FLAGS.dataset

# Load data
adj, features ,labels= load_data(dataset_str)

# Store original adjacency matrix (without diagonal entries) for later
# adj_orig = adj
# adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
# adj_orig.eliminate_zeros()

#adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
#adj = adj_train

if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless

# Some preprocessing
adj_norm = preprocess_graph(adj)

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    # 'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create model
model = None
if model_str == 'gcn_ae':
    model = GCNModelAE(placeholders, num_features, features_nonzero)
elif model_str == 'gcn_vae':
    model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

# Optimizer
with tf.name_scope('optimizer'):
    if model_str == 'gcn_ae':
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm)
    elif model_str == 'gcn_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# cost_val = []
# acc_val = []

def verify(trueLabels, kLabels):
    mapp = {k: k for k in numpy.unique(kLabels)}
    for k in numpy.unique(kLabels):
        k_mapping = numpy.argmax(numpy.bincount(kLabels[trueLabels==k]))
        mapp[k] = k_mapping
    predictions = [mapp[label] for label in trueLabels]
    print(mapp)
    return mapp, predictions





# cost_val = []
# acc_val = []
# val_roc_score = []
#
# adj_label = adj_train + sp.eye(adj_train.shape[0])
# adj_label = sparse_to_tuple(adj_label)

# Train model
feed_dict = construct_feed_dict(adj_norm,features, placeholders)
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    #feed_dict = construct_feed_dict(adj_norm, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Run single weight update
    outs = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)

    # Compute average loss
    avg_cost = outs[1]
    #avg_accuracy = outs[2]

    #roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
    #val_roc_score.append(roc_curr)
    if epoch%10==0:
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),)

print("Optimization Finished!")

clf = KMeans(n_clusters=7,random_state=42)

feed_dict.update({placeholders['dropout']: 0})
emb = sess.run(model.z_mean, feed_dict=feed_dict)
labels = numpy.argmax(labels,axis=1)
print(labels.shape)
print(labels[5])
clf.fit(emb)
y = clf.predict(emb)
#trainMapping, trainPredictions = verify(labels, y)
#print('Accuracy: {}'.format(accuracy_score(y, trainPredictions)))
cm = clustering_metrics(labels, y)
print(cm.evaluationClusterModelFromLabel())