from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
from tsp_utils import gen_cycle_data_missing

#from tsp import gen_tsp_data



## TODO
# make it learn a cycle graph
# start using the graph_layer functions again instead of a dense layer
# try flattening the input and have it predict 1x3 instead of 3x1

def graph_layer(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    out_layer = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])
    out_layer = tf.nn.relu(out_layer)

    return out_layer

def graph_layer2(x, hidden1, hidden2):
    ## NOTE: THIS FLATTENS x...not really what I had in mind
    layer_1   = tf.layers.dense(x, hidden1, use_bias=True)
    out_layer = tf.layers.dense(layer_1, hidden2, use_bias=True)

    return out_layer


class graph_conv_net_r(object):
    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate

    def create_layers(self, x, y, adj):
        pred = x
        for hidden in self.layers:
            ## This flattens pred then does something...not really what I had in mind
            pred  = tf.layers.dense(pred, hidden)
            pred  = tf.einsum('ijk,ilj->ilk', pred, adj) # i verified in numpy that this works


        self.cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, pred))))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

class graph_conv_net(object):
    def __init__(self, params):
        self.weights = params['weights']
        self.biases  = params['biases']
        self.relu    = params['relu']
        self.learning_rate = params['lr']
        self.activation = params['activation']
        self.params = params

    def create_layers(self, x, y, adj):
        pred = x
        L    = len(self.weights.keys())
        i    = 0
        self.cost = 0
        for w, b in zip(self.weights.keys(), self.biases.keys()):
            pred  = tf.add(tf.matmul(pred, self.weights[w]), self.biases[b])
            self.cost += .001*tf.nn.l2_loss(self.weights[w])
            self.cost += .001*tf.nn.l2_loss(self.biases[b])
            if i < L - 1:
                pred  = tf.nn.relu(pred)
            i += 1
            #pred  = tf.einsum('ijk,ilj->ilk', pred, adj) # i verified in numpy that this works

        self.pred = pred
        self.y    = y
        self.softmax = tf.nn.softmax(pred, dim=1)
        self.cost1 = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y, dim=1)
        self.cost += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y, dim=1))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)


### IDEA: Try reshaping the last layer to be standard 1d fully connected

class graph_conv_net_new(object):
    def __init__(self, params):
        self.weights = params['weights']
        self.biases  = params['biases']
        self.weights_keys = list(np.sort(list(self.weights.keys())))
        self.bias_keys    = list(np.sort(list(self.biases.keys())))
        self.relu    = params['relu']        
        self.global_step = params['global_step']
        self.learning_rate = params['lr']
        self.activation = params['activation']
        self.params = params
        self.weight_reg = params['weight_reg']

    def create_layers(self, x, y, P1d, P2d, A_nnd, Fd):
        pred = x
        L    = len(self.weights.keys())
        i    = 0
        self.cost = 0
        for w, b in zip(self.weights_keys[:-1], self.bias_keys[:-1]):
            # could make a 2 layer net by adding another S
            P1   = P1d[i]
            P2   = P2d[i]
            A_nn = A_nnd[i]
            F    = Fd[i]
            I    = tf.concat((tf.matmul(P1, pred), tf.matmul(P2, pred), F), 1)
            S    = tf.add(tf.matmul(I, self.weights[w]), self.biases[b])
            S    = tf.nn.relu(S)
            pred = tf.matmul(A_nn, S)
            pred = tf.nn.relu(pred)
            self.cost += self.weight_reg*tf.nn.l2_loss(self.weights[w])
            self.cost += self.weight_reg*tf.nn.l2_loss(self.biases[b])
            i += 1

        
        w = self.weights_keys[-1]
        b = self.bias_keys[-1]
        pred = tf.add(tf.matmul(pred, self.weights[w]), self.biases[b])
        self.cost += self.weight_reg*tf.nn.l2_loss(self.weights[w])
        self.cost += self.weight_reg*tf.nn.l2_loss(self.biases[b])


        self.pred = pred
        self.y    = y
        self.softmax = tf.nn.softmax(pred, dim=0)
        self.cost1 = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y, dim=0)
        self.cost += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y, dim=0))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, global_step=self.global_step)



# Parameters

def test_learn_sum():
    learning_rate   = 0.001
    training_epochs = 100
    batch_size      = 100
    display_step    = 1
    num_nodes       = 15

    n_input = 6
    n_out   = 1
    # tf Graph input
    x = tf.placeholder("float", [None, num_nodes, n_input])
    y = tf.placeholder("float", [None, num_nodes, n_out])
    adj_mat = tf.placeholder("float", [None, num_nodes, num_nodes])

    X = np.random.rand(10000, num_nodes, 6)
    Y = X.sum(axis=2)[:, :,None]

    X_test = np.random.rand(100, num_nodes, 6)
    Y_test = X_test.sum(axis=2)[:, :,None]

    layers = [6, 1]

    with tf.Session() as sess:
        model = graph_conv_net_r(layers, learning_rate)
        model.create_layers(x, y, adj_mat)

        init = tf.global_variables_initializer()
        sess.run(init)
        i = 0
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(X)/batch_size)
            adj = np.tile(np.eye(num_nodes), (batch_size, 1,1))

            # Loop over all batches
            for i in range(total_batch):
                batch_x = X[i*batch_size:(i+1)*batch_size]
                batch_y = Y[i*batch_size:(i+1)*batch_size]

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([model.optimizer, model.cost], feed_dict={x: batch_x,
                                                              y: batch_y, adj_mat:adj})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
        print("Optimization Finished!")
        c = sess.run(model.cost, feed_dict={x: X_test, y: Y_test, adj_mat:adj})
        print('Final cost', c)

    return(model)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(2))
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    t = np.tile(np.eye(adj.shape[1]), (adj.shape[0],1,1))*d_inv_sqrt[:,:,None]
    temp = np.zeros(adj.shape)
    for z in range(adj.shape[0]):
        temp1 = adj[z, :, :]
        temp2 = t[z,:,:]
        temp[z,:,:] = (temp2.dot(temp1)).dot(temp2)

    return temp

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    N = adj.shape[2]
    eye = np.repeat(np.eye(N)[None, :,:],adj.shape[0],axis=0)
    adj_normalized = normalize_adj(adj + eye)
    return adj_normalized

def construct_weights_biases(size):
    weights = {}
    biases  = {}
    for i in range(len(size)-2):
        name_w = 'h' + str(i)
        name_b = 'b' + str(i)
        weights[name_w] = tf.Variable(tf.random_normal([2*size[i] + 2, size[i+1]]))
        biases[name_b]  = tf.Variable(tf.random_normal([size[i+1]]))
    i += 1
    name_w = 'h' + str(i)
    name_b = 'b' + str(i)
    weights[name_w] = tf.Variable(tf.random_normal([size[i], size[i+1]]))
    biases[name_b]  = tf.Variable(tf.random_normal([size[i+1]]))


    return(weights, biases)



features = np.load('data/tsp/features.npy').item()
routes   = np.load('data/tsp/routes.npy').item()
adj      = np.load('data/tsp/adj.npy').item()
P1       = np.load('data/tsp/p1.npy').item()
P2       = np.load('data/tsp/p2.npy').item()
A_nn     = np.load('data/tsp/Ann.npy').item()
F        = np.load('data/tsp/f.npy').item()
ws       = np.load('data/tsp/ws.npy').item()
node_list = np.load('data/tsp/node_list.npy')
field_size = np.load('data/tsp/field_size.npy')
node_list = np.load('data/tsp/node_list.npy')
optimal = np.load('data/tsp/optimal.npy').item()

tot_ex = len(features.keys())
num_layers = len(field_size)


# field_size = [1, 1, 1]
# num_layers = len(field_size)
# node_list  = [4,5,6]
# num_ex     = 100

# features, routes, P1, P2, A_nn, F, ws, optimal = gen_tsp_data(node_list, num_ex, field_size, num_layers)

keys = list(features.keys())

train_size = int(.5*tot_ex)

inds       = np.random.permutation(tot_ex)
train_inds = inds[:train_size]
test_inds  = inds[train_size:]

learning_rate   = 0.01
n_input = 6
n_out  = 1

sizes = [n_input, 12, 12, 12, n_out]
assert(len(sizes) == (num_layers+2))
weights, biases = construct_weights_biases(sizes)


decay_step = int(train_size*np.mean(np.array(node_list))**2/3)
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                            decay_step, 0.97, staircase=True)


params = {}
params['global_step'] = global_step
params['lr'] = learning_rate
params['activation'] = None
params['relu'] = {
    'h1': True,
    'out': False
}
params['weights'] = weights
params['biases'] = biases
params['weight_reg'] = .001


# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, 1])
P1_tf   = {}
P2_tf   = {}
Ann_tf  = {}
F_tf    = {}
for i in range(num_layers):
    P1_tf[i]   = tf.placeholder('float', [None, None], name='P1')
    P2_tf[i]   = tf.placeholder('float', [None, None], name='P2')
    Ann_tf[i]  = tf.placeholder('float', [None, None], name='Ann')
    F_tf[i]    = tf.placeholder('float', [None, None], name='F')


sess = tf.Session()

model = graph_conv_net_new(params)
model.create_layers(x,y,P1_tf,P2_tf,Ann_tf, F_tf)

init = tf.global_variables_initializer()
sess.run(init)
i = 0

training_epochs = 17
batch_size      = 1
display_step    = 1

# Training cycle


# RESET THE LEARNING RATE
# model.global_step = tf.Variable(0, trainable=False)
# model.learning_rate = tf.train.exponential_decay(starter_learning_rate, model.global_step,
#                                           decay_step, 0.97, staircase=True)

# init = tf.variables_initializer([model.global_step])
# sess.run(init)

for epoch in range(training_epochs):
    start = time.time()
    avg_cost = 0.
    total    = 0.

    # Loop over all batches
    np.random.shuffle(train_inds)
    for ind in train_inds:
        num_nodes = features[keys[ind]][0].shape[0]
        start1 = time.time()
        for in_ind in range(num_nodes):
            batch_x  = features[keys[ind]][in_ind]
            batch_y  = routes[keys[ind]][in_ind]
            P1_feed  = P1[keys[ind]][in_ind]
            P2_feed  = P2[keys[ind]][in_ind]
            Ann_feed = A_nn[keys[ind]][in_ind]
            F_feed   = F[keys[ind]][in_ind]

            for i in range(batch_x.shape[2]):
                x_feed = batch_x[:,:,i]
                x_feed = x_feed / x_feed.sum(0)
                #x_feed = x_feed/num_nodes
                y_feed = np.zeros([num_nodes, 1])
                y_feed[batch_y[i]] = 1

                feed_dict = {}
                for layer in range(num_layers):
                    feed_dict[P1_tf[layer]] = P1_feed[layer]
                    feed_dict[P2_tf[layer]] = P2_feed[layer]
                    feed_dict[Ann_tf[layer]] = Ann_feed[layer]
                    feed_dict[F_tf[layer]]   = F_feed[layer]

                feed_dict[x] = x_feed
                feed_dict[y] = y_feed

                _, pred, c, y_t, c1, softmax = sess.run([model.optimizer, model.pred, model.cost, 
                                                    model.y, model.cost1, model.softmax], 
                                                    feed_dict=feed_dict)
                avg_cost += c1[0]
                total    += 1
                # Compute average loss
        end1 = time.time()
        #print('inner took ' + str(end1-start1) + ' seconds')
            
    # Display logs per epoch step
    end = time.time()
    if epoch % display_step == 0:
        print("Cost:", '%04d' % (epoch+1), "cost=", \
            "{:.9f}".format(avg_cost/total))
        print('Epoch took ' + str(end-start) + ' seconds')
        print('Learning rate is ' + str(sess.run(model.learning_rate)))
        for w, b in zip(weights.keys(), biases.keys()):
            h0 = sess.run(weights[w])
            b0 = sess.run(biases[b])



test_inds = np.sort(test_inds)
solved = 0
unsolved = 0
unsolved_key = []
solved_key = []
solved_cost = {}
optimal_cost = {}

for node in node_list:
    solved_cost[node] = []
    optimal_cost[node] = []


for ind in test_inds:
    print(ind)
    key = keys[ind]
    num_nodes = key[1]
    for in_ind in range(num_nodes):
        batch_x  = features[keys[ind]][in_ind]
        batch_y  = routes[keys[ind]][in_ind]
        P1_feed  = P1[keys[ind]][in_ind]
        P2_feed  = P2[keys[ind]][in_ind]
        Ann_feed = A_nn[keys[ind]][in_ind]
        F_feed   = F[keys[ind]][in_ind]
        opt_cost = optimal[keys[ind]][in_ind]
        W        = ws[keys[ind]][in_ind]

        start_node = in_ind
        goal_node  = in_ind

        visited = np.zeros(num_nodes)
        visited[start_node] = 1

        x_feed = batch_x[:,:,0]
        x_feed = x_feed / x_feed.sum(0)
        #x_feed = x_feed/num_nodes
        goal0  = batch_x[:, 3:, 0]
        y_feed = np.zeros([num_nodes, 1])
        feed_dict = {}
        for layer in range(num_layers):
            feed_dict[P1_tf[layer]] = P1_feed[layer]
            feed_dict[P2_tf[layer]] = P2_feed[layer]
            feed_dict[Ann_tf[layer]] = Ann_feed[layer]
            feed_dict[F_tf[layer]]   = F_feed[layer]

        cost = 0
        steps = 0
        cur_node = start_node
        solved_bool = False
        while steps < 30:
            feed_dict[x] = x_feed
            feed_dict[y] = y_feed

            softmax = sess.run([model.softmax], feed_dict=feed_dict)
            steps += 1
            next_node = np.argmax(softmax)
            state0 = np.zeros([num_nodes, 3])
            state0[next_node, 0] = 1
            state0[goal_node, 1] = 1
            visited[next_node] = 1
            state0[:, 2] = visited
            cost += W[cur_node, next_node]
            cur_node = next_node

            if (next_node == start_node) & (np.sum(visited)==num_nodes):
                solved += 1
                solved_key.append(keys[ind])
                solved_cost[num_nodes].append(cost)
                optimal_cost[num_nodes].append(opt_cost)
                solved_bool = True
                break
            x_feed = np.c_[state0, goal0]
            x_feed = x_feed / x_feed.sum(0)
            #x_feed = x_feed/num_nodes

        if not solved_bool:
            print('Index ' + str(ind) + ' was not solved.')
            unsolved += 1
            unsolved_key.append(keys[ind])


for num_nodes in node_list:
    solved_cost1 = np.array(solved_cost[num_nodes])
    optimal_cost1 = np.array(optimal_cost[num_nodes])

    #print(np.mean(solved_cost1 - optimal_cost1))
    print(num_nodes)
    print(np.mean((solved_cost1-optimal_cost1)/optimal_cost1))


    sc1 = solved_cost1.reshape(solved_cost1.size/num_nodes,num_nodes).min(axis=1)
    oc1 = optimal_cost1.reshape(optimal_cost1.size/num_nodes,num_nodes).min(axis=1)

    print(np.mean((sc1-oc1)/oc1))












