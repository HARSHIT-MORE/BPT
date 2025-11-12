import random as rd
import copy
import math as m
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

def generate_graph(n,m):
  graph=[]
  graph=[[0 for j in range(n)] for i in range(n)]
  m_Count=0
  while m_Count!=m:
    for i in range(n):
      for j in range(n):
        if(m_Count==m):
          break
        if(i==j):
          continue
        k=rd.randint(0,1)
        if(graph[i][j]!=1 and k==1):
          graph[i][j]=1
          graph[j][i]=1
          m_Count=m_Count+1
  return graph

def check_vertexCover(graph,n,vc):
  graphCopy=copy.deepcopy(graph)
  for i in range(n):
    if int(vc[i])==1:
      for j in range(n):
        if(graphCopy[i][j]==1):
          graphCopy[i][j]=0
          graphCopy[j][i]=0
  for i in range(n):
    for j in range(n):
      if graphCopy[i][j]==1:
        return False
  return True

def countOnes(vc):
  c=0
  for i in range(n):
    if(int(vc[i])==1):
      c=c+1
  return c

def printSolution(vc):
  solution=[]
  for i in range(n):
    if(int(vc[i])==1):
      solution.append(i)
  print(solution)
  print("Size of Vertex Cover: ",len(solution))
  return solution

def vertexCover(graph,n):
  minCover=n
  solution="1"*n
  for i in range(int(m.pow(2,n))):
    vc="0"*int(n-len((bin(i)+"")[2::])) + (bin(i)+"")[2::]
    check=check_vertexCover(graph,n,vc)
    if(check and minCover>countOnes(vc)):
      minCover=countOnes(vc)
      solution=vc

  solution=printSolution(solution)
  return solution

def matchingGraph(G):
  E = copy.deepcopy(G[1])
  print(E)
  seen = []
  matc = []
  senI = []
  i = rd.randint(0, len(E)-1)
  senI.append(i)
  while(len(seen) < len(E)):
    while(i in senI):
      i = rd.randint(0, len(E)-1)
    senI.append(i)
    if(E[i] not in seen):
      matc.append(E[i])
      for j in range(len(E)):
        if((E[i][0] in E[j] or E[i][1] in E[j]) and E[j] not in seen):
          seen.append(E[j])
  return matc


def vertexCoverApprox(matching):
  vc = []
  for e in matching:
    vc.append(e[0])
    vc.append(e[1])
  return vc

def matToSet(G):
  V = [x for x in range(len(G))]
  E = []
  for i in range(len(G)):
    for j in range(len(G[i])):
      if(i <= j and G[i][j] == 1):
        E.append((i, j))
  return (V, E)

def plot_graph(G, highlight_nodes=None):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='blue', node_size=700)
    if highlight_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=highlight_nodes, node_color='red', node_size=900)
    plt.show()

def plot_matching(matching):
  G = nx.Graph()
  G.add_edges_from(matching)
  pos = nx.planar_layout(G)
  nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='blue', node_size=700)
  plt.title("Matching")
  plt.show()


n = 10
edges = [10, 20, 30, 40]
graphs = []
try:
  f = open("Input.txt", "r+")
except:
  f = open("Input.txt", "w+")
lines = f.readlines()
if(len(lines)!=0):
  graph = []
  lno = 1
  while(lno < len(lines)):
    graph.append(lines[lno].split())
    for j in range(len(graph[-1])):
      graph[-1][j] = int(graph[-1][j])
    if(lno%n == 0):
      graphs.append(graph)
      graph = []
    lno+=1
else:
  f.write("INPUT GRAPH\n")
  for m in edges:
    graph = generate_graph(n, m)
    for i in graph:
      for j in i:
        f.write(str(j)+" ")
      f.write("\n")

f.close()

node=[10]*4
edge=[]
vertex_c=[]
vertex_a=[]
factor = []
execution_time=[]
for graph in graphs:
  graph1 = copy.deepcopy(graph)
  vc=vertexCover(graph, n)
  vertex_c.append(vc)
  graph = matToSet(graph)
  print("\nNo. of Nodes = ", n)
  edgeno=len(graph[1])
  print("\nNo. of edges = ", edgeno)
  edge.append(edgeno)


  print("\nVertex Cover Approximation: ")
  start=time.time()
  matching = matchingGraph(graph)
  vcApprox=vertexCoverApprox(matching)
  end=time.time()-start
  print(vcApprox)
  vertex_a.append(vcApprox)
  execution_time.append(end)
  print("execution time: ",end)
  graph_np = np.array(graph1)
  factor.append(len(vcApprox)/len(vc))

  G = nx.from_numpy_array(graph_np)
  plot_graph(G,vcApprox)
  print("\n\n")
  plot_matching(matching)


df=pd.DataFrame({
  "No. of Node":node,
  "No. of Edge":edge,
  "Vertex Cover":vertex_c,
  "Vertex Cover Approx.":vertex_a,
  "Factor":factor,
  "Execution time":execution_time

  })
print(df)
df.to_csv("output2.csv", index=False)
