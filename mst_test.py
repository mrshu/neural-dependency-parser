import mst
import mst_Tim_Dozat as mst_test
from random import randint
import numpy as np

def negative_graph(graph):
  '''This puts the negative weight in graph as the cost in the new graph'''
  
  new_graph = {}
  for vtx, nbhs in graph.items():
    new_graph[vtx] = {}
    for nbh, weight in nbhs.items():
      new_graph[vtx][nbh] = -weight
  
  return new_graph
  
def random_graph_gen(n):
  '''This function generates a graph rooted at 0, with random positive weighths'''
  
  graph = {}
  for vtx in range(n):
    graph[vtx] = {}
    for nbh in range(1, n):
      if nbh != vtx:
        graph[vtx][nbh] = randint(1, 10)
  
  return graph

def cycle_check(root, graph, vertices):
  '''This finds out if there is a cycle in the graph - BFS'''
  
  visited = {vtx: 0 for vtx in vertices}
  non_leaves = {vtx for vtx, nbhs in graph.items()}
  queue = [root]
  
  while queue != []:
    vtx = queue[0]
    visited[vtx] = 1
    
    if vtx in non_leaves:
      nbhs = graph[vtx]
    
      for nbh, weight in nbhs.items():
        if visited[nbh] == 1:
          return 0
        queue.append(nbh)
    
    queue.pop(0)
  
  # Now check that every vertex was visited (only one connected component)
  for vtx in vertices:
    if visited[vtx] == 0:
      return 0
  
  return 1
  
def graph2scores (graph):
  '''Takes a graph as input and returns the reversed scores'''
  
  n = len(graph)
  score = np.full([n, n], 0.)
  
  for vtx, nbhs in graph.items():
    for nbh, weight in nbhs.items():
      score[nbh][vtx] = weight
  
  def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    x = np.exp(x)
    return x / np.sum(x, axis=1, keepdims=True)
  probs = softmax(score)
  probs *= 1-np.eye(len(probs)).astype(np.float32)
  probs[0] = 0
  probs[0,0] = 1
  probs /= np.sum(probs, axis=1, keepdims=True)
  
  '''# Now make it a probability matrix (i.e. sum of outgoing scores should be 1)
  # (i.e. sum of columns should be 1)
  for vtx in range(n):
    sum_col = 0
    for nbh in range(1, n):
      sum_col += score[nbh][vtx]
    for nbh in range(1, n):
      score[nbh][vtx] /= sum_col'''
  
  return probs

def scores2graph(scores):
  '''Takes a score matrix and return the corresponding graph'''
  
  l = len(scores)
  graph = {}
  for vtx in range(l):
    graph[vtx]= {}
    for nbh in range(1, l):
      graph[vtx][nbh] = scores[nbh][vtx]
  
  return graph

def test(no_tests, size_graph):
  '''Test mst against mst_Martin_Louis_Bright for no_tests many test of size
  size_graph'''
  
  # This variables chek the number of tests on which the comparation algorithm 
  # did return a tree, and how many times it had the same weight
  pass_test = 0
  same_weight = 0
  count = 0
  
  for t in range(no_tests):
    graph = random_graph_gen(size_graph)
    score = graph2scores(graph)
    
    test_tree = mst.mst(graph)
    comp_tree = mst_test.chu_liu_edmonds(score)
    
    # compare the two trees by comarping the resulting weight
    test_sum = 0
    comp_sum = 0
    
    for vtx, nbhs  in test_tree.items():
      for nbh, weight in nbhs.items():
        test_sum += graph[vtx][nbh]
    
    for nbh, vtx in enumerate(comp_tree):
      if nbh != 0 and nbh!=vtx:
        comp_sum += graph[vtx][nbh]
    
    if test_sum != comp_sum:
      #count += 1
      return ('problem found:', graph, test_sum, comp_sum, test_tree, comp_tree, score)
    
    # Below is the comparation of the exact trees
    '''
    for vtx, nbhs in comp_tree.items():
      test_nbh = list(test_tree[vtx].keys())
      comp_nbh = list(comp_tree[vtx].keys())
      if test_nbh != comp_nbh:
        return ('problem found:', graph, test_tree, comp_tree)'''
  
  return ('All fine!', count)
    
    