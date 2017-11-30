# This is an implementation of Chu-Liu-Edmonds algorithm
# For convenience, graphs will pe represented as dictionaries in the following form
#  {node_1 : {node_i : cost_edge_1_to_i, ...}, ...}
# Similarly, trees will be keept as ..

# Note: I use weight and cost with the same meaning

import numpy as np

def max_incoming_edge_selection(graph):
  '''Selects for each non-root node a maximum incoming edge and returns the 
  subgraph tree selection'''
  
  vertices = list(graph.keys())
  n = len(vertices)
  best_pred = dict(zip(vertices, [(0, 0)]*n)) # keeps maximum (predecessor, cost)
  
  for vtx, nbhs in graph.items():
    for nbh, cost in nbhs.items():
      if cost > best_pred[nbh][1]:
        best_pred[nbh] = (vtx, cost)
  
  return best_pred

def cycle_detection(best_pred):
  '''This function takes a best_pred_list (which is a list that has on the i^th
  position a pair (j, cost) which gives the maximum costpredecessor for i, i.e
  edge (j, i) is in the selection)
  and returns a cycle if one exists, and an empty list otherwise'''
  
  vertices = list(best_pred.keys())
  n = len(vertices)
  visited = dict(zip(vertices, [-1]*n)) # keeps maximum (predecessor, cost)
  
  # We do DFS: keep track of the currend cycle attempt in 'cycle' and of the visited vertices
  for i in vertices:
    cycle = [i]
    if visited[i] == -1:
      # find the list of predecessors from node i (i <- i_1 <- i_2 <- ...)
      pred = best_pred[i][0]
      
      while visited[pred] == -1:
        cycle.append(pred)
        visited[pred] = i
        pred = best_pred[pred][0]
      
      # find if the reason for stopping was because we went back to a 
      # visited brach while looking at another vertex (i.e. visited[pred] != i)
      # or because we found a cycle (i.e. visited[pred] == i)
      
      if visited[pred] == i and pred != i:
        return cycle
  
  return []

def contract(graph, cycle):
  '''This function contracts the graph as in CLE algorithm:
  it takes bestincoming and outcomming edge and comprimes the cycle as one node'''
  
  vertices = list(graph.keys())
  n = len(vertices)
  
  # First we find the weight of the cycle
  len_cycle = len(cycle)
  cycle_weight = 0
  for i in range(1, len_cycle):
    vtx = cycle[i]
    pred = cycle[i-1]
    cycle_weight += graph[pred][vtx]
  
  # Now we find the incomming edge to the cycle for each node in the graph
  incomming = dict(zip(vertices, [(0, -1)]*n)) # keeps for each non-in-cycle vertex (weight_to_cycle, vtx_to_cycle)
  for i in vertices:
    if i not in cycle:
      for j in range(1, len_cycle):
        vtx = cycle[j]
        pred = cycle[j-1]
        weight = cycle_weight - graph[pred][vtx] + graph[i][vtx]
        if weight > incomming[i][0]:
          incomming[i] = (weight, vtx)
  
  # Now we find the outgoing edge from the cycle (i.e. max of ougoing arcs)
  outgoing = dict(zip(vertices, [(0, -1)]*n)) # keeps for each non-in-cycle vertex (weight_from_cycle, vtx_from_cycle)
  for i in vertices:
    if i not in cycle and (i!=0):
      for vtx in cycle:
        if graph[vtx][i] > outgoing[i][0]:
          outgoing[i] = (graph[vtx][i], vtx)
  
  # Now we construct the new graph
  new_node = max(vertices) + 1
  new_graph = {}
  new_graph[new_node] = {}
  for vtx, nbhs in graph.items():
    if vtx not in cycle:
      new_graph[vtx] = {}
    
      # Add edges independent from cycle
      for nbh, cost in nbhs.items():
        if nbh not in cycle:
          new_graph[vtx][nbh] = cost
    
      # Add the edge from and to the cycle
      new_graph[vtx][new_node] = incomming[vtx][0]
      if vtx != 0:
        new_graph[new_node][vtx] = outgoing[vtx][0]
  
  # Now we take the second components of incoming and outgoing dictionaries to
  # have the links to the particular vertex in the cycle
  to_cycle_edges = {}
  for vtx, pair in incomming.items():
    if vtx not in cycle:
      to_cycle_edges[vtx] = pair[1]
      
  from_cycle_edges = {}
  for vtx, pair in incomming.items():
    if vtx not in cycle and vtx != 0:
      from_cycle_edges[vtx] = pair[1]
  
  return (new_graph, to_cycle_edges, from_cycle_edges)

def cle(graph):
  '''This is the CLE algorithm implementation. Should return the MST'''
  
  # Now we find G_M
  best_pred = max_incoming_edge_selection(graph)
  
  # Check for cycles
  cycle = cycle_detection(best_pred)
  
  # Check if G_M is a tree
  if cycle == []:
    return best_pred
  else:
    new_graph, to_cycle_edges, from_cycle_edges = contract(graph, cycle)
    best_pred_new_graph = cle(new_graph)
    
    # Find the node x that links in the best path to the cycle
    cycle_index = max(list(graph.keys())) + 1 
    x = best_pred_new_graph[cycle_index][0]
    c = to_cycle_edges[x] # member of the cycle that x links to
    pos_c = cycle.index(c)# position of c in cycle
    if pos_c == 0:
      pos_c = len(cycle)
    pred_c =  cycle[pos_c - 1] # predecessor of c in cycle
    
    # Create a new best_pred list fron best_pred_new_graph
    new_best_pred = {}
    vertices = list(best_pred_new_graph.keys())
    for vtx in vertices:
      if vtx == cycle_index:
        # Add edge x->c
        new_best_pred[c]= (x, graph[x][c])
        # Add edges within the cycle, except for pred_c-> c
        for i in range(1, len(cycle)):
          if i != pos_c:
            new_best_pred[cycle[i]] = best_pred[cycle[i]]
      else:
        if best_pred_new_graph[vtx][0] != cycle_index:
          new_best_pred[vtx] = best_pred_new_graph[vtx]
        else:
          new_best_pred[vtx] = (from_cycle_edges[vtx], best_pred_new_graph[vtx][1])
  
  return new_best_pred

def example_graph_1():
  '''This function return the graph provided in the slides week 3, where
  root = node_0, John = node_1, saw = node_2, Mary = node_3'''
  
  nbhs_0 = {1: 9, 2: 10, 3: 9}
  nbhs_1 = {2: 20, 3: 3}
  nbhs_2 = {1: 30, 3: 30}
  nbhs_3 = {1: 11, 2: 0}
  
  return {0: nbhs_0, 1: nbhs_1, 2: nbhs_2, 3: nbhs_3}

def example_graph_2():
  '''This function return the graph provided in figure 14.14 from book, where
  root = node_0, Book = node_1, that = node_2, flight = node_3'''
  
  nbhs_0 = {1: 12, 2: 4, 3: 4}
  nbhs_1 = {2: 5, 3: 7}
  nbhs_2 = {1: 6, 3: 8}
  nbhs_3 = {1: 5, 2: 7}
  
  return {0: nbhs_0, 1: nbhs_1, 2: nbhs_2, 3: nbhs_3}