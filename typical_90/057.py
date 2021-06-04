import sys
from collections import defaultdict, Counter, deque
from itertools import permutations, combinations, product, combinations_with_replacement, groupby, accumulate
import operator
from math import sqrt, gcd, factorial
# from math import isqrt, prod,comb  # python3.8用(notpypy)
#from bisect import bisect_left,bisect_right
#from functools import lru_cache,reduce
#from heapq import heappush,heappop,heapify,heappushpop,heapreplace
import numpy as np
#import networkx as nx
#from networkx.utils import UnionFind
#from numba import njit, b1, i1, i4, i8, f8
#from scipy.sparse import csr_matrix
#from scipy.sparse.csgraph import shortest_path, floyd_warshall, dijkstra, bellman_ford, johnson, NegativeCycleError
# numba例 @njit(i1(i4[:], i8[:, :]),cache=True) 引数i4配列、i8 2次元配列,戻り値i1
def input(): return sys.stdin.readline().rstrip()
def divceil(n, k): return 1+(n-1)//k  # n/kの切り上げを返す
def yn(hantei, yes='Yes', no='No'): print(yes if hantei else no)


def hakidashi2(A):
    x,y=A.shape
    ans = []
    for i in range(y):
        if A[:,i].max() != 0:
            A=A[np.argsort(A[:,i])[::-1]]
            ans.append(np.copy(A[0]))
            #print(A)
            A[np.where(A[:,i]>0)[0]]^=A[0]
    return np.stack(ans)

def hakidashi_hantei2(ans, K):
    for aa in ans:
        if K[np.where(aa==1)[0][0]]!=0:
            K ^= aa
    return np.all(K == np.zeros_like(K))



def main():
    mod = 10**9+7
    mod2 = 998244353
    m,n=map(int, input().split())
    panels=np.zeros((m,n),dtype=np.int64)
    for i in range(m):
        t=int(input())
        A=list(map(int, input().split()))
        for AA in A:
            panels[i][AA-1]+=1

    ans=hakidashi2(panels)
    S=np.array(list(map(int, input().split())),dtype=np.int64)
    if hakidashi_hantei2(ans,S):
        print(pow(2,len(panels)-len(ans),mod2))
    else:
        print(0)


if __name__ == '__main__':
    main()

