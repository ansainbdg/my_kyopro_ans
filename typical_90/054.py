import sys
from collections import defaultdict, Counter, deque
from itertools import permutations, combinations, product, combinations_with_replacement, groupby, accumulate
import operator
from math import sqrt, gcd, factorial
from copy import deepcopy
# from math import isqrt, prod,comb  # python3.8用(notpypy)
#from bisect import bisect_left,bisect_right
#from functools import lru_cache,reduce
#from heapq import heappush,heappop,heapify,heappushpop,heapreplace
#import numpy as np
#import networkx as nx
#from networkx.utils import UnionFind
#from numba import njit, b1, i1, i4, i8, f8
#from scipy.sparse import csr_matrix
#from scipy.sparse.csgraph import shortest_path, floyd_warshall, dijkstra, bellman_ford, johnson, NegativeCycleError
# numba例 @njit(i1(i4[:], i8[:, :]),cache=True) 引数i4配列、i8 2次元配列,戻り値i1
def input(): return sys.stdin.readline().rstrip()
def divceil(n, k): return 1+(n-1)//k  # n/kの切り上げを返す
def yn(hantei, yes='Yes', no='No'): print(yes if hantei else no)


def main():
    mod = 10**9+7
    mod2 = 998244353
    n, m = map(int, input().split())
    rs = []
    writes = [[] for i in range(n)]
    for i in range(m):
        k = input()
        r = list(map(lambda x: int(x)-1, input().split()))
        rs.append(r)
        for rr in r:
            writes[rr].append(i)
    
    dist = [10**10]*n
    d = deque()
    d.append(0)
    dist[0] = 0
    while d:
        v = d.popleft()
        for w in writes[v]:
            if rs[w]:
                for mem in rs[w]:
                    if dist[mem] > dist[v]+1:
                        dist[mem]=dist[v]+1
                        d.append(mem)
                rs[w]=[]
    
    for i in range(n):
        if dist[i]==10**10:
            print(-1)
        else:
            print(dist[i])


if __name__ == '__main__':
    main()
