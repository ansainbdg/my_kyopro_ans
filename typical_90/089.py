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


def rankdata(arr):  # 座標圧縮
    unique = list(set(arr))
    unique.sort()
    rank = {e: i+1 for i, e in enumerate(unique)}
    return [rank[i] for i in arr]


class Bit:  # 1-indexed
    def __init__(self, N):
        """
        INPUT
        N [int] -> 全部0で初期化
        N [list] -> そのまま初期化
        """
        if isinstance(N, int):
            self.N = N
            self.depth = N.bit_length()
            self.tree = [0] * (N + 1)
            self.elem = [0] * (N + 1)
        elif isinstance(N, list):
            self.N = len(N)
            self.depth = self.N.bit_length()
            self.tree = [0] + N
            self.elem = [0] + N
            self._init()
        else:
            raise "INVALID INPUT: input must be int or list"

    def _init(self):
        size = self.N
        for i in range(1, self.N):
            if i + (i & -i) > size:
                continue
            self.tree[i + (i & -i)] += self.tree[i]

    def show(self):
        print(*self.elem)

    def add(self, i, x):
        if i == 0:
            raise "BIT is 1_indexed"
        self.elem[i] += x
        while i <= self.N:
            self.tree[i] += x
            i += i & -i

    def sum(self, i):
        res = 0
        while i > 0:
            res += self.tree[i]
            i -= i & -i
        return res

    def lower_bound(self, val):
        if val <= 0:
            return 0
        i = 0
        k = 1 << self.depth
        while k:
            if i + k <= self.N and self.tree[i + k] < val:
                val -= self.tree[i + k]
                i += k
            k >>= 1
        return i + 1

    def __getitem__(self, i):
        return self.sum(i)-self.sum(i-1)


def main():
    mod = 10**9+7
    mod2 = 998244353
    n, k = map(int, input().split())
    A = rankdata(list(map(int, input().split())))
    bit = Bit(n+5)
    check = [0]*n  # check[r]:lからrまでの区間なら連続して選べる
    l = 0
    inversion = 0
    for r, AA in enumerate(A):
        bit.add(AA, 1)
        inversion += r+1-bit.sum(AA)-l
        while inversion > k:
            AAA = A[l]
            inversion -= bit.sum(AAA-1)
            bit.add(AAA, -1)
            l += 1
        check[r] = l
    #print(check)
    dp = [0]*(n+1)  # dp[i] 1indexでiでグループ分けを終えた
    dp[0] = 1
    accdp = [0]*(n+2)
    accdp[1] = 1
    for r, l in enumerate(check):
        dp[r+1] = (accdp[r+1]-accdp[l]) % mod
        accdp[r+2] = dp[r+1]+accdp[r+1]
    print(dp[-1])


if __name__ == '__main__':
    main()
