## Generates instances of the TSP problem
## and solves them using Google's solver

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import numpy as np
import pickle
from copy import copy
import numpy.linalg as nlg
from tsp_utils import *
import sys
import os


def google_or_cost(W0, start_node):
  num_nodes = W0.shape[0]
  search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
  search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
  num_routes = 1

  adj_mat = create_adj(num_nodes)
  adj_mat.matrix = adj_mat.scale * W0
  adj_fn = adj_mat.Distance

  adj_mat1 = adj_mat.matrix

  routing = pywrapcp.RoutingModel(num_nodes, num_routes, start_node)
  routing.SetArcCostEvaluatorOfAllVehicles(adj_fn)
  assignment = routing.SolveWithParameters(search_parameters)
  feature0, route0 = feature_from_assignment(routing, assignment, num_nodes)
  opt_val = float(assignment.ObjectiveValue())/adj_mat.scale
  return(opt_val)

def gen_tsp_data(node_list, num_inds, field_size, num_layers):
  search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
  search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
  num_routes = 1

  features  = {}
  routes    = {}
  P1        = {}
  P2        = {}
  A_nn      = {}
  F         = {}
  adj       = {}
  ws        = {}
  optimal   = {}
  dist_left = {}
  rankings  = {}

  for num_nodes in node_list:
    for ind in range(num_inds):
      if ind % 10 == 0:
        print(ind)
      key1 = (ind, num_nodes)
      features[key1]  = {}
      routes[key1]    = {}
      adj[key1]       = {}
      P1[key1]        = {}
      P2[key1]        = {}
      A_nn[key1]      = {}
      F[key1]         = {}
      ws[key1]        = {}
      optimal[key1]   = {}
      dist_left[key1] = {}
      rankings[key1]  = {}
      adj_mat = create_adj(num_nodes)
      adj_fn = adj_mat.Distance

      adj_mat1 = adj_mat.matrix
      A = (adj_mat1 > 0).astype(int)
      W1 = adj_mat1/float(adj_mat.scale)
      #W = W1 / np.mean(W1.sum(0))
      W = W1 / num_nodes

      for start_node in range(num_nodes):
        routing = pywrapcp.RoutingModel(num_nodes, num_routes, start_node)
        routing.SetArcCostEvaluatorOfAllVehicles(adj_fn)
        assignment = routing.SolveWithParameters(search_parameters)
        feature0, route0 = feature_from_assignment(routing, assignment, num_nodes)
        opt_val = float(assignment.ObjectiveValue())/adj_mat.scale
        if opt_val > 1000:
          route = np.arange(start_node, start_node+num_nodes, 1)%num_nodes
          feature0, route0 = feature_from_route(route, num_nodes)
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
        optimal[key1][start_node] = opt_val
        dist_left[key1][start_node] = get_dist_to_go(start_node, route0, opt_val, W1)
        rankings[key1][start_node]  = get_ranking_targets(start_node, route0, num_nodes)

  return(features, routes, P1, P2, A_nn, F, ws, optimal, dist_left, rankings)

def gen_tsp_data_cycle_linked(node_list, num_inds, field_size, num_layers, edges):
  search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
  search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
  num_routes = 1

  features  = {}
  routes    = {}
  P1        = {}
  P2        = {}
  A_nn      = {}
  F         = {}
  adj       = {}
  ws        = {}
  optimal   = {}
  dist_left = {}
  rankings  = {}

  for num_nodes in node_list:
    avg = 0
    total = 0
    num_edges = edges[num_nodes]

    for ind in range(num_inds):
      if ind % 10 == 0:
        None
        #print(ind)
      key1 = (ind, num_nodes)
      features[key1]  = {}
      routes[key1]    = {}
      adj[key1]       = {}
      P1[key1]        = {}
      P2[key1]        = {}
      A_nn[key1]      = {}
      F[key1]         = {}
      ws[key1]        = {}
      optimal[key1]   = {}
      dist_left[key1] = {}
      rankings[key1]  = {}
      adj_mat = create_adj_cycle(num_nodes, num_edges)
      adj_fn = adj_mat.Distance

      adj_mat1 = adj_mat.matrix * (adj_mat.matrix < 1e6)
      A = (adj_mat1 > 0).astype(int)
      W1 = adj_mat1/float(adj_mat.scale)
      #W = W1 / np.mean(W1.sum(0))
      W = W1 / num_nodes

      for start_node in range(num_nodes):
        routing = pywrapcp.RoutingModel(num_nodes, num_routes, start_node)
        routing.SetArcCostEvaluatorOfAllVehicles(adj_fn)
        assignment = routing.SolveWithParameters(search_parameters)
        feature0, route0 = feature_from_assignment(routing, assignment, num_nodes)
        opt_val = float(assignment.ObjectiveValue())/adj_mat.scale
        avg += (adj_mat.cycle_cost - opt_val)/opt_val
        total += 1
        if opt_val > 1000:
          #print('ugh')
          route = np.arange(start_node, start_node+num_nodes, 1)%num_nodes
          feature0, route0 = feature_from_route(route, num_nodes)
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
        optimal[key1][start_node] = opt_val
        dist_left[key1][start_node] = get_dist_to_go(start_node, route0, opt_val, W1)
        rankings[key1][start_node] = get_ranking_targets(start_node, route0, num_nodes)
    print(num_nodes)
    print(avg/total)

  return(features, routes, P1, P2, A_nn, F, ws, optimal, dist_left, rankings)

def get_dist_to_go(start_node, route, opt_val, W):
  d_to_go = []
  n_to_go = []
  d_from_c = [opt_val]
  to_go_dist = opt_val
  cur_node = start_node
  for next_node in route[:-1]:
      to_go_dist -= W[cur_node, next_node]
      d_to_go.append(to_go_dist)
      d_from_c.append(to_go_dist)
      n_to_go.append(next_node)
      cur_node = next_node
  n_to_go.append(route[-1])
  d_to_go.append(0)

  return(zip(n_to_go, d_to_go, d_from_c))

def get_ranking_targets(start_node, route, num_nodes):
  targets = np.zeros((num_nodes, num_nodes))
  nodes   = range(num_nodes)
  t = np.zeros(num_nodes)
  visited = [start_node]
  unvisited = [n for n in route if n not in visited]
  i = 0
  t[visited] = -10
  t[unvisited] = (np.arange(len(unvisited))+1).astype(float)/(len(unvisited))
  targets[i, :] = t

  for next_node in route[:-1]:
    i += 1
    t = np.zeros(num_nodes)
    visited.append(next_node)
    unvisited = [n for n in route if n not in visited]
    t[visited] = -10
    t[unvisited] = (np.arange(len(unvisited))+1).astype(float)/(len(unvisited))
    targets[i, :] = t
  targets[:, start_node] = 0
  return(targets)













