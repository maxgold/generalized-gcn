from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import numpy as np
import pickle
from copy import copy
import numpy.linalg as nlg
from tsp_utils import *
import sys
import os






def gen_tsp_data(node_list, num_inds, field_size, num_layers):
  search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
  search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
  num_routes = 1

  features = {}
  routes   = {}
  P1       = {}
  P2       = {}
  A_nn     = {}
  F        = {}
  adj      = {}
  ws       = {}
  optimal  = {}

  for num_nodes in node_list:
    for ind in range(num_inds):
      key1 = (ind, num_nodes)
      features[key1] = {}
      routes[key1]   = {}
      adj[key1]      = {}
      P1[key1]       = {}
      P2[key1]       = {}
      A_nn[key1]     = {}
      F[key1]        = {}
      ws[key1]       = {}
      optimal[key1]  = {}
      adj_mat = create_adj(num_nodes)
      adj_fn = adj_mat.Distance

      A = (adj_mat.matrix > 0).astype(int)
      W1 = adj_mat.matrix/float(adj_mat.scale)
      #W = W1 / np.mean(W1.sum(0))
      W = W1 / num_nodes

      for start_node in range(num_nodes):
        routing = pywrapcp.RoutingModel(num_nodes, num_routes, start_node)
        routing.SetArcCostEvaluatorOfAllVehicles(adj_fn)
        assignment = routing.SolveWithParameters(search_parameters)
        feature0, route0 = feature_from_assignment(routing, assignment, num_nodes)
        adj0       = construct_adj_fieldsize(A, W, field_size, num_layers)
        P1_0, P2_0, A_nn0, F_0 = nn_mats_from_adj_fieldsize_beta(adj0, field_size, num_layers)
        features[key1][start_node] = feature0
        routes[key1][start_node] = route0
        adj[key1][start_node] = adj0
        P1[key1][start_node] = P1_0
        P2[key1][start_node] = P2_0
        A_nn[key1][start_node] = A_nn0
        F[key1][start_node]    = F_0
        ws[key1][start_node] = W1
        optimal[key1][start_node] = float(assignment.ObjectiveValue())/adj_mat.scale

  return(features, routes, P1, P2, A_nn, F, ws, optimal)


def main(node_list, num_ex, field_size, num_layers):
  features, routes, P1, P2, A_nn, F, ws, optimal = gen_tsp_data(node_list, num_ex, field_size, num_layers)

  np.save('data/tsp/features.npy', features, allow_pickle=True)
  np.save('data/tsp/routes.npy', routes, allow_pickle=True)
  np.save('data/tsp/p1.npy', P1, allow_pickle=True)
  np.save('data/tsp/p2.npy', P2, allow_pickle=True)
  np.save('data/tsp/Ann.npy', A_nn, allow_pickle=True)
  np.save('data/tsp/f.npy', F, allow_pickle=True)
  np.save('data/tsp/ws.npy', ws, allow_pickle=True)
  np.save('data/tsp/field_size.npy', field_size, allow_pickle=True)
  np.save('data/tsp/num_layers.npy', num_layers, allow_pickle=True)
  np.save('data/tsp/node_list.npy', node_list, allow_pickle=True)
  np.save('data/tsp/optimal.npy', optimal, allow_pickle=True)






if __name__ == "__main__":
  num_ex    = int(sys.argv[1])
  node_list = []
  for n in sys.argv[2:]:
    node_list.append(int(n))
  print(node_list)
  field_size = [1, 1, 1]
  num_layers = len(field_size)
  main(node_list, num_ex, field_size, num_layers)
  localfile = './data'
  remotehost = 'ec2-user@ec2-54-191-168-233.us-west-2.compute.amazonaws.com'
  remotefile = '~/generalized_gcn'
  os.system('scp -ri ~/ec2/planning.pem "%s" "%s:%s"' % (localfile, remotehost, remotefile) )
























