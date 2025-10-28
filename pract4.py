import time
import random 
import math
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import copy
from scipy.optimize import linprog

def check_vertexCover(graph,vc):
    n = len(graph)
    for i in range(n):
        for j in range(i+1, n):
            if graph[i][j] == 1:  
                if i not in vc and j not in vc:
                    return False
    return True

def brute_force(graph):
  n=len(graph)
  node=list(range(n))
  for i in range(1,n+1):
    for subset in itertools.combinations(node, i): #creating subset 
      if check_vertexCover(graph,subset):
        return list(subset)
  return []

def matching(edges):
  E=set(edges)
  vc=set()
  while E:
    (u,v)=E.pop()
    vc.update([u,v])
    E={e for e in E if u not in e and v not in e}
  return vc

def lp_solver(edges, n=None, threshold=0.5):
    if n is None:
        n = max(max(u, v) for u, v in edges) + 1 #max vertice index

    c = np.ones(n)

    # Linprog-->  Constraints: for each edge (u,v):  -x_u - x_v <= -1  (x_u + x_v >= 1)
    A, b = [], []
    for (u, v) in edges:
        row = np.zeros(n)
        row[u] = -1
        row[v] = -1
        A.append(row)
        b.append(-1)

    bounds = [(0, 1)] * n
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

    # fractional LP solution
    x = result.x
    vc = {(i,xv) for i, xv in enumerate(x) if xv >= threshold}
    return vc

def gen_graph(n,m):
    try:
        m < ((n)*(n-1))/2
    except:
        print("Invalid number of edges")
        return None

    graph = np.zeros((n, n), dtype=int)
    count =m

    while(count>0):
        for i in range(n):
            for j in range(n):
                if i!=j and count>0 and graph[i][j]!=1:
                    temp = int(random.randint(0,1))
                    if temp==1:
                        graph[i][j] = temp
                        graph[j][i] = temp
                        count-=1

    return graph.tolist()

for n in [10,20]:
    results = []
    for m in range(10, int((n*(n-1))/2),10):
        graph = gen_graph(n, m)

        # Convert adjacency matrix to list of edges
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if graph[i][j] == 1:
                    edges.append((i, j))

        mvc = matching(edges)
        bf = brute_force(graph)
        lp = lp_solver(edges)

        results.append({
            "edges": m,
            "Greedy VC Size": len(mvc),
            "Size of LP Solver": len(lp),
            "Min VC Size": len(bf),
            "Greedy Approx": len(mvc)/len(bf),
            "LP Approx": len(lp)/len(bf),
        })

    #saving results
    df = pd.DataFrame(results)
    df.to_csv("Comparsion: VC_appx_and_lp_solver.csv")
    print(df)
    


    #visualization------------------

    plt.figure(figsize=(8,5))
    plt.plot(df["edges"], df["Greedy Approx"], marker='o', label="Greedy 2-Approx Factor")
    plt.plot(df["edges"], df["LP Approx"], marker='s', label="LP Solver")
    plt.axhline(1, color='gray', linestyle='--', label="Optimal (Brute Force)")
    plt.xlabel("Number of edges (m)")
    plt.ylabel("Approximation Factor")
    plt.title(f"Approximation Factor Comparison (n={n})")
        

    plt.legend()
    plt.grid(True)
    plt.show()


