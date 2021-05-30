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


class SparceTable:
    def __init__(self, A, op=min, e=10**10):
        self.n = len(A)
        self.op = op
        self.e = e
        self.table = [None]*(len(A).bit_length())  # [i, i+2^k)のop
        self.table[0] = A
        pre_table = A
        k = 0
        for k in range(len(A).bit_length()-1):
            pre_table = self.table[k]
            self.table[k+1] = [op(pre_table[i], pre_table[i+(1 << k)])
                               for i in range(len(pre_table)-(1 << k))]
            k += 1

    # [l, r)のop
    def query(self, l, r):
        l = min(max(0, l),self.n)
        r = min(max(0, r),self.n)
        if l == r:
            return self.e

        k = (r-l).bit_length()-1
        return self.op(self.table[k][l], self.table[k][r-(1 << k)])


def main():
    mod = 10**9+7
    mod2 = 998244353
    w, n = map(int, input().split())
    c=[-10**15]*(w+1)
    c[0]=0
    dp = SparceTable(c, op=max, e=-10**15)
    for _ in range(n):
        l, r, v = map(int, input().split())
        newdp=[0]*(w+1)
        for i in range(w+1):
            newdp[i]=max(dp.query(i-r,i-l+1)+v,dp.query(i,i+1))
        dp=SparceTable(newdp,op=max,e=-10**15)
        #print(newdp)
    print(max(-1,dp.query(w,w+1)))


if __name__ == '__main__':
    main()