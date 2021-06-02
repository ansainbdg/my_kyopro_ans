import sys
from collections import defaultdict, Counter, deque
from itertools import permutations, combinations, product, combinations_with_replacement, groupby, accumulate
import operator
from math import sqrt, gcd, factorial
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
    n, S = map(int, input().split())
    ab = [list(map(int, input().split())) for i in range(n)]
    dpa = [1 << ab[0][0]]
    dpb = [1 << ab[0][1]]
    mask = (1 << S+4)-1
    for a, b in ab[1:]:
        dp = (dpa[-1] | dpb[-1]) & mask
        dpa.append(dp << a)
        dpb.append(dp << b)
    if (dpa[-1] | dpb[-1]) & (1 << S):
        ans=''
        while dpa:
            a,b=ab.pop()
            if dpa.pop() & (1 << S):
                S-=a
                ans+='A'
            else:
                S-=b
                ans+='B'
        print(ans[::-1])
    else:
        print('Impossible')



if __name__ == '__main__':
    main()
