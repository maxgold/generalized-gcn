from heapq import heappush, heappop
import heapq
from sys import maxint
import numpy as np
import networkx as nx



class State(object):
    def __init__(self, nodes, t_val, h_val):
        self.nodes = nodes
        self.t_val = t_val
        self.h_val = h_val

    def __cmp__(self, other):
        return cmp(self.h_val, other.h_val)

    def __eq__(self, other):
        # and self.nodes != REMOVED
        return self.nodes == other.nodes


def get_inds(open_list, new_state):
    indices = [i for i, x in enumerate(open_list) if x == new_state]
    return(indices)



def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]




class Graph(object):
    def __init__(self, n, start, W):
        self.nodes = range(n)
        self.W     = W
        self.num_nodes = n
        self.start = start
        self.nx_graph = nx.from_numpy_matrix(self.W)

    def neighbors(self, visited):
        res = []
        if len(visited) < self.num_nodes:
            unvisited = diff(self.nodes, visited)
            for y in unvisited:
                res.append(visited + [y])
        else:
            res.append(visited + [self.start])
        return(res)
    def heuristic(self, visited):
        if len(visited) < self.num_nodes:
            unvisited = np.array(diff(self.nodes, visited))
            new_W = self.W[unvisited[:,None], unvisited]
            nxg = nx.from_numpy_matrix(new_W)
            t = nx.minimum_spanning_tree(nxg)
            return(t.size(weight='weight'))
        else:
            return(0)



start = 1
num_nodes = 20

W = np.random.rand(num_nodes, num_nodes)
np.fill_diagonal(W, 0)
G = Graph(num_nodes, start, W)


def my_astar(G, start):
    G.start = start
    num_nodes = G.num_nodes
    cur_node = start
    # second node should be value of minimum spanning tree
    start_state = State([cur_node], 0, 0)
    open_list = []
    closed_list = []
    seen_states = {}
    seen_states[(cur_node)] = [0, 0]

    heapq.heappush(open_list, start_state)
    visited = [start]
    unfinished = True

    i = 0
    while unfinished:
        if i % 100 == 0:
            print(i)
        i += 1
        cur_state = heapq.heappop(open_list)
        visited   = cur_state.nodes
        cur_node  = cur_state.nodes[-1]
        if (len(cur_state.nodes) == (num_nodes + 1)) & (cur_state.nodes[-1] == start):
            unfinished = False
            break
        for new_state in G.neighbors(visited):
            if tuple(new_state) in seen_states.keys():
                prev_tour_cost, prev_h_cost = seen_states[new_state]
            else:
                prev_tour_cost, prev_h_cost = np.inf, np.inf

            successor_tour_cost = cur_state.t_val + G.W[cur_node, new_state[-1]]
            #h = .1*(num_nodes - len(visited))
            h = 2*G.heuristic(visited)
            #h = 0
            successor_h_cost    = successor_tour_cost + h
            new_state_cost = State(new_state, successor_tour_cost, successor_h_cost)
            if new_state_cost in open_list:
                if prev_tour_cost <= successor_tour_cost:
                    # this should break to neighbor loop
                    break
                else:
                    inds = get_inds(open_list, new_state_cost)
                    assert(len(inds) == 1)
                    open_list[inds[0]] = open_list[-1]
                    open_list.pop()
                    heapq.heappush(open_list, new_state_cost)
                    heapq.heapify(open_list)
                    seen_states[tuple(new_state)] = [successor_tour_cost, successor_h_cost]
            elif new_state_cost in closed_list:
                if prev_tour_cost <= successor_tour_cost:
                    # this should break to the neighbor loop    
                    break
                else:
                    inds = get_inds(closed_list, new_state_cost)
                    for ind in inds:
                        closed_list[ind] = closed_list[-1]
                    for ind in inds:
                        closed_list.pop()
                    heapq.heappush(closed_list, new_state_cost)
                    heapq.heapify(closed_list)
                    seen_states[tuple(new_state)] = [successor_tour_cost, successor_h_cost]                       
            else:
                seen_states[tuple(new_state)] = [successor_tour_cost, successor_h_cost]
                heapq.heappush(open_list, new_state_cost)       

        heappush(closed_list, cur_state)

    print(cur_state.nodes)
    print(cur_state.t_val)


















































