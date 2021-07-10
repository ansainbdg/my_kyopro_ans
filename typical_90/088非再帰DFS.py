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
    n, q = map(int, input().split())
    A = list(map(int, input().split()))
    zyouken = [[] for i in range(n)]
    for i in range(q):
        x, y = map(lambda x: int(x)-1, input().split())
        zyouken[y].append(x)
    exist = [[] for i in range(8889)]
    stack = []
    stack.append([[1, 0]])
    stack.append([[0, 0]])
    while stack:
        # print(stack)
        tmp = stack[-1]
        if len(tmp) == n:
            sums = sum(t[0]*AA for t, AA in zip(tmp, A))
            if exist[sums]:
                print(sum(t[0] for t in tmp))
                print(*[i+1 for i, t in enumerate(tmp) if t[0]])
                print(sum(t[0] for t in exist[sums]))
                print(*[i+1 for i, t in enumerate(exist[sums]) if t[0]])
                return
            else:
                exist[sums] = tmp
                stack.pop()
                continue
        if tmp[-1][1] == 0:
            tmp[-1][1] = 1
            stack[-1][-1][1] = 1
            stack.append(tmp+[[0, 0]])
        elif tmp[-1][1] == 1:
            tmp[-1][1] = 2
            stack[-1][-1][1] = 2
            for ng in zyouken[len(tmp)]:
                if tmp[ng][0] == 1:
                    stack.pop()
                    break
            else:
                stack.append(tmp+[[1, 0]])
        else:
            stack.pop()


if __name__ == '__main__':
    main()
