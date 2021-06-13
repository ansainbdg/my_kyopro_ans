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


class PrepereFactorial2:  # maxnumまでの階乗を事前計算して、順列、組み合わせ、重複組み合わせを計算するクラス。逆元のテーブルもpow無しで前計算する。maxnumに比べて関数呼び出しが多いならこちら
    def __init__(self, maxnum=3*10**5, mod=10**9+7):
        self.factorial = [1]*(maxnum+1)
        modinv_table = [-1] * (maxnum+1)
        modinv_table[1] = 1
        for i in range(2, maxnum+1):
            self.factorial[i] = (self.factorial[i-1]*i) % mod
            modinv_table[i] = (-modinv_table[mod % i] * (mod // i)) % mod
        self.invfactorial = [1]*(maxnum+1)
        for i in range(1, maxnum+1):
            self.invfactorial[i] = self.invfactorial[i-1]*modinv_table[i] % mod
        self.mod = mod

    def permutation(self, n, r):
        return self.factorial[n]*self.invfactorial[n-r] % self.mod

    def combination(self, n, r):
        if r < 0 or r > n:
            return 0
        return self.permutation(n, r)*self.invfactorial[r] % self.mod

    def combination_with_repetition(self, n, r):
        return self.combination(n+r-1, r)

#https://atcoder.jp/contests/practice2/submissions/16789717
p, g, ig = 998244353, 3, 332748118
W = [pow(g, (p - 1) >> i, p) for i in range(24)]
iW = [pow(ig, (p - 1) >> i, p) for i in range(24)]
 
def fft(k, f):
    for l in range(k, 0, -1):
        d = 1 << l - 1
        U = [1]
        for i in range(d):
            U.append(U[-1] * W[l] % p)
        
        for i in range(1 << k - l):
            for j in range(d):
                s = i * 2 * d + j
                f[s], f[s+d] = (f[s] + f[s+d]) % p, U[j] * (f[s] - f[s+d]) % p
 
def ifft(k, f):
    for l in range(1, k + 1):
        d = 1 << l - 1
        for i in range(1 << k - l):
            u = 1
            for j in range(i * 2 * d, (i * 2 + 1) * d):
                f[j+d] *= u
                f[j], f[j+d] = (f[j] + f[j+d]) % p, (f[j] - f[j+d]) % p
                u = u * iW[l] % p
 
def convolution(a, b):
    n0 = len(a) + len(b) - 1
    k = (n0).bit_length()
    n = 1 << k
    a = a + [0] * (n - len(a))
    b = b + [0] * (n - len(b))
    fft(k, a), fft(k, b)
    for i in range(n):
        a[i] = a[i] * b[i] % p
    ifft(k, a)
    invn = pow(n, p - 2, p)
    for i in range(n0):
        a[i] = a[i] * invn % p
    del a[n0:]
    return a


def main():
    mod = 10**9+7
    mod2 = 998244353
    r, g, b, k = map(int, input().split())
    x, y, z = map(int, input().split())
    pf = PrepereFactorial2(6*10**5+1, mod2)
    rless = k-y
    gless = k-z
    bless = k-x
    """
    問題の条件を以下のように言い換える。

    赤をk-y個未満選ぶ・緑をk-z個未満選ぶ・青をk-x個未満選ぶの3条件を
    全て満たさない。
    
    包除原理により、k個選ぶ全体から、上記条件を1個以上満たす選び方を引いて、条件を2個以上満たす選び方を足して、条件を3個満たす選び方を引けばよい。
    """
    #k個選ぶ全体
    ans = pf.combination(r+g+b, k) 
    rcomb = [0]*rless
    gcomb = [0]*gless
    bcomb = [0]*bless
    #条件を1個以上満たす選び方
    for i in range(rless):
        a = pf.combination(r, i)
        rcomb[i] = a
        ans -= a*pf.combination(g+b, k-i)
    for i in range(gless):
        a = pf.combination(g, i)
        gcomb[i] = a
        ans -= a*pf.combination(r+b, k-i)
    for i in range(bless):
        a = pf.combination(b, i)
        bcomb[i] = a
        ans -= a*pf.combination(r+g, k-i)
    #条件を2個以上満たす選び方
    if rless and gless:
        rgcomb = convolution(rcomb, gcomb)
        for i in range(rless+gless-1):
            ans += rgcomb[i]*pf.combination(b, k-i)
    if rless and bless:
        rbcomb = convolution(rcomb, bcomb)
        for i in range(rless+bless-1):
            ans += rbcomb[i]*pf.combination(g, k-i)
    if gless and bless:
        gbcomb = convolution(gcomb, bcomb)
        for i in range(gless+bless-1):
            ans += gbcomb[i]*pf.combination(r, k-i)
    #条件を3個満たす選び方
    if rless and gless and bless and rless+gless+bless-3 >= k:
        ans -= convolution(rgcomb[:k+1],bcomb[:k+1])[k]
    print(ans % mod2)


if __name__ == '__main__':
    main()



