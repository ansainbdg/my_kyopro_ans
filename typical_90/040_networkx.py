from networkx import*
(n,w),A,*z=[[*map(int,t.split())]for t in open(0)]
g=DiGraph()
d=g.add_edge
while n:
 n-=1;d(1,~n,c=w);d(~n,2,c=A[n])
 for p in z[n][1:]:d(~n,-p,c=9e9)
print(sum(A)-minimum_cut(g,1,2,capacity='c')[0])