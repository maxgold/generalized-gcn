######## TEST CYCLE MISSING ###################

num_nodes = 5
num_layers = 2
field_size = [1, 2]
features, adj, routes, P1, P2, A_nn, F = gen_cycle_data_missing(num_nodes, field_size, num_layers)

key1         = list(adj.keys())[10]
num_nodes    = adj[key1][0].shape[0]


train_size = int(.1*len(features.keys()))

inds       = np.random.permutation(len(features))
train_inds = inds[:train_size]
test_inds  = inds[train_size:]

learning_rate   = 0.01
n_input = features[key1].shape[1]
#n_out    = labels.shape[1]
n_out  = 1

# this should have num_layers + 2 elements
sizes = [n_input, 6, 12, n_out]
assert(len(sizes) == (num_layers+2))
weights, biases = construct_weights_biases(sizes)


#sizes = [n_input, 12, 24, 1]

decay_step = int(train_size*num_nodes/3)
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.01
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
adj_mat = tf.placeholder("float", [num_nodes, num_nodes])
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

training_epochs = 30
batch_size      = 1
display_step    = 1

# Training cycle

keys = list(features.keys())


for epoch in range(training_epochs):
    start = time.time()
    avg_cost = 0.
    total    = 0.

    # Loop over all batches
    np.random.shuffle(train_inds)
    for ind in train_inds:
        batch_x  = features[keys[ind]]
        batch_y  = routes[keys[ind]]
        P1_feed  = P1[keys[ind]]
        P2_feed  = P2[keys[ind]]
        Ann_feed = A_nn[keys[ind]]
        F_feed   = F[keys[ind]]

        for i in range(batch_x.shape[2]):
            x_feed = batch_x[:,:,i]
            x_feed = x_feed / x_feed.sum(0)
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


avg_cost = 0
total = 0
highs = []
lookback = []

keys = list(features.keys())
N = len(test_inds)
count = 0

for ind in test_inds:
    if count % 50 == 0:
        print(count/N)
    count += 1
    batch_x  = features[keys[ind]]
    batch_y  = routes[keys[ind]]
    P1_feed  = P1[keys[ind]]
    P2_feed  = P2[keys[ind]]
    Ann_feed = A_nn[keys[ind]]
    F_feed   = F[keys[ind]]

    for i in range(batch_x.shape[2]):            
        x_feed = batch_x[:,:,i]
        x_feed = x_feed / x_feed.sum(0)
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
        high = np.max(softmax)
        highs.append(high)
        lookback.append((ind, i))

        # Compute average loss
        avg_cost += c1[0]
        total    += 1

print(avg_cost/total)
highs = np.array(highs)

test_inds = np.sort(test_inds)
solved = 0
unsolved = 0
unsolved_key = []
solved_key = []
suboptimal = []
for ind in test_inds:
    waypoint1 = 0
    waypoint2 = 0

    batch_x  = features[keys[ind]]
    batch_y  = routes[keys[ind]]
    P1_feed  = P1[keys[ind]]
    P2_feed  = P2[keys[ind]]
    Ann_feed = A_nn[keys[ind]]
    F_feed   = F[keys[ind]]
    opt_steps = batch_x.shape[2]

    goal_node1 = keys[ind][0]
    goal_node2 = keys[ind][1]
    start_node = keys[ind][2]
    if start_node == goal_node1:
        waypoint1 += 1
    visited = np.zeros(num_nodes)
    visited[start_node] = 1

    i = 0

    x_feed = batch_x[:,:,0]
    x_feed = x_feed / x_feed.sum(0)
    goal0  = batch_x[:, 3:, i]
    y_feed = np.zeros([num_nodes, 1])
    feed_dict = {}
    for layer in range(num_layers):
        feed_dict[P1_tf[layer]] = P1_feed[layer]
        feed_dict[P2_tf[layer]] = P2_feed[layer]
        feed_dict[Ann_tf[layer]] = Ann_feed[layer]
        feed_dict[F_tf[layer]]   = F_feed[layer]

    steps = 0

    while steps < 30:
        feed_dict[x] = x_feed
        feed_dict[y] = y_feed

        softmax = sess.run([model.softmax], feed_dict=feed_dict)
        steps += 1
        next_node = np.argmax(softmax)
        state0 = np.zeros([num_nodes, 3])
        state0[next_node, 0] = 1
        state0[goal_node2, 1] = 1
        visited[next_node] = 1
        state0[:, 2] = visited
        if next_node == goal_node1:
            waypoint1 = 1
        if next_node == goal_node2:
            waypoint2 = 1
        if waypoint1 == 1 & waypoint2 == 1:
            #print('Index ' + str(ind) + ' took ' + str(steps) + ' steps.')
            #print('The optimal tour took ' + str(opt_steps) + ' steps.')
            solved += 1
            solved_key.append(keys[ind])
            suboptimal.append(steps - opt_steps)
            break
        x_feed = np.c_[state0, goal0]
        x_feed = x_feed / x_feed.sum(0)

    if waypoint1 == 0 or waypoint2 == 0:
        print('Index ' + str(ind) + ' was not solved.')
        unsolved += 1
        unsolved_key.append(keys[ind])

print('The total amount solved: ' + str(solved))
print('The total amount unsolved: ' + str(unsolved))










##### TEST REGULAR CYCLE ########

features = np.load('data/features.npy')
adj      = np.load('data/adj.npy')
routes   = np.load('data/routes.npy')
weights  = np.load('data/weights.npy')
P1       = np.load('data/P1.npy')
P2       = np.load('data/P2.npy')
A_nn     = np.load('data/A_nn.npy')

num_nodes    = adj.shape[0]
num_graphs   = features.shape[3]
num_examples = features.shape[2]*features.shape[3]
num_edges    = 2*num_nodes

## this gives it the right answer....there should be num_nodes - 1 examples for each solution
#features  = np.transpose(features, (0,1,3,2)).reshape((num_nodes,num_nodes+5,num_examples))
features  = np.transpose(features, (0,1,3,2)).reshape((num_nodes,6,num_examples))
features  = np.transpose(features, (2,0,1))
#features  = features.reshape(features.shape[0], features.shape[1]*features.shape[2])
features /= features.sum(1)[:,None]

adj       = np.tile(adj[:,:,:,None], (1,1,1,num_nodes)).reshape((num_nodes,num_nodes,num_examples))
adj       = np.transpose(adj, (2,0,1))
adj_n     = preprocess_adj(adj)

#adj_weights   = np.tile(weights[:,:,:,None], (1,1,1,num_nodes)).reshape((num_nodes,num_nodes,num_examples))
#adj_weights   = np.transpose(weights, (2,0,1))
#adj_weights_n = preprocess_adj(weights)

labels        = np.zeros((num_examples, num_nodes))
label_routes  = routes.T.flatten().astype(int)
labels[np.arange(num_examples), label_routes] = 1


#features = 1 - features didn't work
#features = np.concatenate((features[:, :, 0][:,:,None], features[:, :, 2][:,:,None]), axis=2)
#for i in range(len(features)):
#    features[i] /= features[i].sum(axis=0)

test_size = 4000

train_size = int(.75*len(features))

inds       = np.random.permutation(len(features))
train_inds = inds[:train_size]
test_inds  = inds[train_size:]


X = features[train_inds]
#X_adj = weights_n[:test_size]
X_adj = adj_n[train_inds]
Y = labels[train_inds]

X_test = features[test_inds]
#X_adj_test = weights_n[test_size:]
X_adj_test = adj_n[test_inds]
Y_test = labels[test_inds]



learning_rate   = 0.01
n_input = features.shape[2]
#n_out    = labels.shape[1]
n_out  = 1

#sizes = [n_input, 12, 24, 1]
sizes = [2*n_input, 6, n_out]
weights, biases = construct_weights_biases(sizes)


params = {}
params['lr'] = learning_rate
params['activation'] = None
params['relu'] = {
    'h1': True,
    'out': False
}
params['weights'] = weights
params['biases'] = biases


# tf Graph input
x = tf.placeholder("float", [num_nodes, n_input])
y = tf.placeholder("float", [num_nodes, 1])
adj_mat = tf.placeholder("float", [num_nodes, num_nodes])
P1_tf   = tf.placeholder('float', [num_edges, num_nodes], name='P1')
P2_tf   = tf.placeholder('float', [num_edges, num_nodes], name='P2')
Ann_tf  = tf.placeholder('float', [num_nodes, num_edges], name='Ann')

sess = tf.Session()

#model = graph_conv_net(layers, learning_rate, tf.nn.relu)
#model = graph_conv_net(params)
#model.create_layers(x, y, adj_mat)

model = graph_conv_net_new(params)
model.create_layers(x,y,P1_tf,P2_tf,Ann_tf)

init = tf.global_variables_initializer()
sess.run(init)
i = 0

training_epochs = 1000
batch_size      = 1
display_step    = 2


# Training cycle

Ann_feed = A_nn[:,:,0]
P1_feed  = P1[:,:,0]
P2_feed  = P2[:,:,0]

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = max(1, int(len(X)/batch_size))

    # Loop over all batches
    for i in range(total_batch):
        batch_x = X[i*batch_size:(i+1)*batch_size][0]
        batch_y = Y[i*batch_size:(i+1)*batch_size].T
        adj     = X_adj[i*batch_size:(i+1)*batch_size][0]

        #adj     = np.tile(np.eye(num_nodes), (batch_size, 1,1))


        # Run optimization op (backprop) and cost op (to get loss value)
        _, pred, c, y_t, c1, softmax = sess.run([model.optimizer, model.pred, model.cost, 
                                            model.y, model.cost1, model.softmax], 
                                            feed_dict={x: batch_x, y: batch_y, P1_tf:P1_feed, P2_tf:P2_feed,
                                            Ann_tf:Ann_feed})
        

        # Compute average loss
        avg_cost += c1[0] / total_batch
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost=", \
            "{:.9f}".format(avg_cost))
        h0 = sess.run(weights['h0'])
        h1 = sess.run(weights['h1'])
        b0 = sess.run(biases['b0'])
        b1 = sess.run(biases['b1'])
        print((np.mean(np.abs(h0)), np.mean(np.abs(h1)), np.mean(np.abs(b0)), np.mean(np.abs(b1))))



for i in range(total_batch):
    batch_x = X[i*batch_size:(i+1)*batch_size]
    batch_y = Y[i*batch_size:(i+1)*batch_size]
    adj     = X_adj[i*batch_size:(i+1)*batch_size][0]
    #adj     = np.tile(np.eye(num_nodes), (batch_size, 1,1))


    # Run optimization op (backprop) and cost op (to get loss value)
    _, pred, c, y_t, c1, softmax = sess.run([model.optimizer, model.pred, model.cost, model.y, model.cost1, model.softmax], 
                        feed_dict={x: batch_x, y: batch_y, P1_tf:P1_feed, P2_tf:P2_feed,
                                            Ann_tf:Ann_feed})

    if np.max(y_t + softmax) < 1.6:
        print(i)
    print(y_t + softmax)
        



Ann_feed = A_nn[:,:,0]
P1_feed  = P1[:,:,0]
P2_feed  = P2[:,:,0]

avg_cost = 0.
total_batch = max(1, int(len(X_test)/batch_size))

highs = np.zeros([0])
# Loop over all batches
for i in range(total_batch):
    batch_x = X_test[i*batch_size:(i+1)*batch_size][0]
    batch_y = Y_test[i*batch_size:(i+1)*batch_size].T

    # Run optimization op (backprop) and cost op (to get loss value)
    pred, c, y_t, c1, softmax = sess.run([model.pred, model.cost, 
                                        model.y, model.cost1, model.softmax], 
                                        feed_dict={x: batch_x, y: batch_y, 
                                        P1_tf:P1_feed, P2_tf:P2_feed,
                                        Ann_tf:Ann_feed})

    highs = np.r_[highs, (np.max(softmax))]

    avg_cost += c1[0] / total_batch








