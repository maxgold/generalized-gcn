## Script to run experiments
## Generate datasets, build a GCN, and evaluate the performance

from __future__ import print_function
import tensorflow as tf
import numpy as np
import json
import time
from graph_layer import *
from gen_tsp import gen_tsp_data, gen_tsp_data_cycle_linked
import pickle

if __name__ == '__main__':

    num_nodes = 6
    node_list = [num_nodes]
    edges = {}
    for node in node_list:
        edges[node] = 2*node
    num_train = 1000
    num_test  = 100
    field_size = [1, 1, 1]
    num_layers = len(field_size)
    sizes = [6, 6, 6, 6, 1]
    skip = True

    ## FULLY CONNECTED GRAPHS
    data_train = gen_tsp_data(node_list, num_train, field_size, num_layers)
    data_test  = gen_tsp_data(node_list, num_test, field_size, num_layers)

    ## CHORD GRAPHS
    #data_train = gen_tsp_data_cycle_linked(node_list, num_ex, field_size, num_layers, edges)
    #data_test  = gen_tsp_data_cycle_linked(node_list, num_ex, field_size, num_layers, edges)




    n_input = 6
    n_out  = 1

    if skip:
        weights, biases = construct_weights_biases_skip(sizes)
    else:
        weights, biases = construct_weights_biases(sizes)

    starter_learning_rate = .001
    decay_step = int((200*8)**2/3)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                decay_step, .997, staircase=True)

    params = {}
    params['global_step'] = global_step
    params['lr'] = learning_rate
    params['relu'] = {
        'h1': True,
        'out': False
    }
    params['weights'] = weights
    params['biases'] = biases
    params['weight_reg'] = .001
    params['skip'] = skip

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, 1])
    visited = tf.placeholder("float", [None, 1])

    P1_tf   = {}
    P2_tf   = {}
    Ann_tf  = {}
    F_tf    = {}
    for i in range(num_layers):
        P1_tf[i]   = tf.placeholder('float', [None, None], name='P1')
        P2_tf[i]   = tf.placeholder('float', [None, None], name='P2')
        Ann_tf[i]  = tf.placeholder('float', [None, None], name='Ann')
        F_tf[i]    = tf.placeholder('float', [None, None], name='F')

    placeholders = [x, y, P1_tf, P2_tf, Ann_tf, F_tf, global_step, visited]

    sess = tf.Session()
    model = graph_conv_net_label(params)
    model.create_layers(x,y,P1_tf,P2_tf,Ann_tf, F_tf)
    init = tf.global_variables_initializer()
    sess.run(init)

    # tf Graph input

    training_epochs = 30
    batch_size      = 1
    display_step    = 1


    train_inds = range(num_train)[:int(.9*num_train)]
    test_inds  = range(num_train)[int(.9*num_train):]

    train_model_class(sess, model, data_train, placeholders, field_size, train_inds, 
                     test_inds, node_list, training_epochs, batch_size, display_step)


    test_inds = range(num_test)
    res = score_model_class(sess, model, data_test, placeholders, field_size, node_list, test_inds)
    print(res)








