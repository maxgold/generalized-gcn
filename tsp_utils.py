## Utility functions to help generate TSP instances

import numpy as np
import pickle
from copy import copy
import numpy.linalg as nlg
import json
import random

#from tsp_utils import *

class create_adj(object):
  def __init__(self, tsp_size):
    self.scale  = 1000
    self.matrix = np.round(np.random.rand(tsp_size,tsp_size),3)*self.scale
    np.fill_diagonal(self.matrix, 0)
    self.matrix = (self.matrix + self.matrix.T)/2

  def Distance(self, from_node, to_node):
    return self.matrix[from_node][to_node]


def edges_from_mat(matrix):
  edges = zip(np.where(matrix>0)[0], np.where(matrix>0)[1])
  return(edges)

class create_adj_cycle(object):
  def __init__(self, tsp_size, num_new_edges = None):
    if num_new_edges == None:
      num_new_edges = tsp_size
    self.scale = 1000
    cycle_weights = np.random.rand(tsp_size)
    cycle_cost = np.sum(cycle_weights)
    self.matrix = np.zeros((tsp_size, tsp_size))
    for i in range(tsp_size):
      self.matrix[i, (i+1)%tsp_size] = cycle_weights[i]
      self.matrix[(i+1)%tsp_size, i] = cycle_weights[i]
    
    cycle_edges = edges_from_mat(self.matrix)
    t = np.ones((tsp_size, tsp_size))
    np.fill_diagonal(t, 0)
    all_edges = np.array(edges_from_mat(t))
    all_edges = all_edges[all_edges[:,0] < all_edges[:,1]]
    all_edges = list(map(tuple, all_edges))
    new_edges = [x for x in all_edges if x not in cycle_edges]
    random.shuffle(new_edges)
    new_edges = np.array(new_edges)
    num_new_edges = min(len(new_edges), num_new_edges)
    for i in range(num_new_edges):
      val = np.random.rand(1)
      self.matrix[new_edges[i,0],new_edges[i,1]] = val
      self.matrix[new_edges[i,1],new_edges[i,0]] = val

    self.matrix = np.round(self.matrix, 3) * self.scale
    self.matrix[self.matrix==0] = 1e6
    self.cycle_cost = cycle_cost
    np.fill_diagonal(self.matrix, 0)

  def Distance(self, from_node, to_node):
    return(self.matrix[from_node][to_node])



def distance(x1, y1, x2, y2):
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return dist


def nn_mats_from_adj_fieldsize_beta(adj, field_size, num_layers):
  P1d = {}
  P2d = {}
  A_nnd = {}
  Fd     = {}
  num_nodes = adj[0].shape[0]
  for layer in range(num_layers):
    dist = field_size[layer]
    A = adj[layer]
    num_edges = np.sum(A[:,:,0] > 0)
    num_nodes = A.shape[0]
    edges = np.zeros([0,2])
    R = A.shape[0]
    C = A.shape[1]
    for i in range(C):
        for j in range(R):
            if A[i,j, 0] > 0:
                edges = np.r_[edges, np.array([[i, j]])]

    edges = edges.astype(int)
    P1 = np.zeros([edges.shape[0], num_nodes])
    P2 = np.zeros([edges.shape[0], num_nodes])
    F  = np.zeros([edges.shape[0], 2])
    ## TODO try replacing this with the weight
    P1[np.arange(edges.shape[0]).astype(int), edges[:,0]] = 1
    P2[np.arange(edges.shape[0]).astype(int), edges[:,1]] = 1

    A_nn = np.zeros([num_nodes, num_edges])
    c = 0
    for edge in edges:
      F[c, 0] = A[edge[0], edge[1], 0]
      F[c, 1] = A[edge[0], edge[1], 1] #should be field_size[layer]
      #F[c, 2] = A[edge[0], edge[1], 2] #should be entry of the weight power matrix
      A_nn[edge[0], c] = 1 # this should be set to W_ij
      c += 1
    F = F/num_nodes
    P1d[layer] = P1
    P2d[layer] = P2
    A_nnd[layer] = A_nn
    Fd[layer]    = F

  return(P1d, P2d, A_nnd, Fd)

def construct_adj_fieldsize(A, W, field_size, num_layers):
  res = {}
  for i in range(num_layers):
    t1 = nlg.matrix_power(W, field_size[i])
    t2 = nlg.matrix_power(A, field_size[i])
    t = np.concatenate((t1[:,:,None],t2[:,:,None]), axis=2)
    res[i] = t

  return res

def feature_from_assignment(routing, assignment, num_nodes):
  route = []
  features = np.zeros([num_nodes,6,0])

  index = routing.Start(0)

  cur_node = index % num_nodes
  start_node = cur_node
  goal_node = cur_node
  visited = np.zeros(num_nodes)
  visited[start_node] = 1
  goal = np.zeros([num_nodes, 3])
  goal[goal_node, 0] = 1
  goal[goal_node, 1] = 1
  goal[:, 2] = 1
  state0 = np.zeros([num_nodes, 3])
  state0[cur_node, 0] = 1
  state0[goal_node, 1]  = 1
  visited[cur_node] = 1
  state0[:, 2] = visited
  feature0 = np.c_[state0, goal][:,:,None]
  features = np.concatenate((features,feature0), axis=2)

  while np.sum(visited)!=num_nodes:
    index = assignment.Value(routing.NextVar(cur_node))
    cur_node = index % num_nodes
    route.append(cur_node)
    state0 = np.zeros([num_nodes, 3])
    state0[cur_node, 0] = 1
    state0[goal_node, 1]  = 1
    visited[cur_node] = 1
    state0[:, 2] = visited
    feature0 = np.c_[state0, goal][:,:,None]
    features = np.concatenate((features,feature0), axis=2)
  route.append(start_node)
  return(features, route)

def feature_from_route(route, num_nodes):
  # this doesn't work for some unknown reason...w
  features = np.zeros([num_nodes,6,0])
  cur_node = route[0]
  i = 1
  start_node = cur_node
  goal_node = cur_node
  visited = np.zeros(num_nodes)
  visited[start_node] = 1
  goal = np.zeros([num_nodes, 3])
  goal[goal_node, 0] = 1
  goal[goal_node, 1] = 1
  goal[:, 2] = 1
  state0 = np.zeros([num_nodes, 3])
  state0[cur_node, 0] = 1
  state0[goal_node, 1]  = 1
  visited[cur_node] = 1
  state0[:, 2] = visited
  feature0 = np.c_[state0, goal][:,:,None]
  features = np.concatenate((features,feature0), axis=2)

  while np.sum(visited)!=num_nodes:
    cur_node = route[i]
    state0 = np.zeros([num_nodes, 3])
    state0[cur_node, 0] = 1
    state0[goal_node, 1]  = 1
    visited[cur_node] = 1
    state0[:, 2] = visited
    feature0 = np.c_[state0, goal][:,:,None]
    features = np.concatenate((features,feature0), axis=2)
    i += 1
  route = list(route)
  route.append(start_node)
  route = route[1:]
  return(features, route)









