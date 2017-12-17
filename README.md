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
  - Call mst_one_out_root(graph) to have the  maximum spanning tree such that there is only one outgoing edge from node 0 (root)
  - The file mst_Tim_Dozat contains the original file of Tim Dozat (https://github.com/tdozat/UnstableParser/blob/master/parser/misc/mst.py), with correction added for the implementation of Chu-Liu-Edmonds, in the function chu_liu_edmonds
  - The file mst_test.py contains the automated testing procedure of ourimplementation of MST versus the corrected version of that of Tim Dozat
          - Call the function test, with two parameters namely the number of tests, no_tests, and the size of the testing graphs, size_graph, to find if the two algorithms agree on the maximum weight of a spanning tree
          - The procedure should return 'All fine!' if now diference wes found for no_tests many random graphs of sizes size_graph
          - If differences are found the alogrithm returns 'problem found:' followed by the details of the example to caused the fail
          - We've run this function with 1000 graphs of size 70, and no differences were found

