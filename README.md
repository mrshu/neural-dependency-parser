# neural-dependency-parser
An impelemntation of a Neural Dependency Parser

MST:
The Maximum Spanning Tree Algorithm was done by Chu-Liu-Edmonds
  - This algorithm can be found in mst.py
  - To access it call the function mst:
      - The input should be a directed graph such that:
          - The root is 0
          - All other nodes are labeled 1 -> n
          - The graph is provided as a dictionary with keys being the nodes and the entries for each key being in turn a dictionary
          - The dictionary corresponding to node i should have keys for each j between 1 and n, and distinct from i, and for each such j, the entry is the weight for the arc from node i to j
      - The output is the Maximum Spanning Tree:
          - It is given as a dictionary where the entries are nodes
          - For each node i, there corresponding value is a new dictionary that gives the successors of i in the maximum spanning tree, and for each of them the corresponding weight

