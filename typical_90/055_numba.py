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
import numpy as np
#import networkx as nx
#from networkx.utils import UnionFind
from numba import njit, b1, i1, i4, i8, f8
#from scipy.sparse import csr_matrix
#from scipy.sparse.csgraph import shortest_path, floyd_warshall, dijkstra, bellman_ford, johnson, NegativeCycleError
# numba例 @njit(i1(i4[:], i8[:, :]),cache=True) 引数i4配列、i8 2次元配列,戻り値i1
def input(): return sys.stdin.readline().rstrip()
def divceil(n, k): return 1+(n-1)//k  # n/kの切り上げを返す
def yn(hantei, yes='Yes', no='No'): print(yes if hantei else no)

@njit(i8(i8,i8,i8,i8[:]),cache=True)
def solve(n,p,q,A):
    ans = 0
    for a in range(n):
        r = A[a]
        for b in range(a+1, n):
            rr = r*A[b] % p
            for c in range(b+1, n):
                rrr = (rr*A[c]) % p
                for d in range(c+1, n):
                    rrrr = (rrr*A[d]) % p
                    for e in range(d+1, n):
                        if (rrrr*A[e]) % p==q:
                            ans+=1
    return ans


def main():
    mod = 10**9+7
    mod2 = 998244353
    n, p, q = map(int, input().split())
    A = np.array(list(map(int, input().split())),dtype=np.int64)
    print(solve(n,p,q,A))


if __name__ == '__main__':
    main()
