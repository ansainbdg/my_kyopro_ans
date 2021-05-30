from numpy import*
from scipy.sparse import*
(n,m),*a=[int_(t.split())for t in open(0)]
a=int_(a).T
a[1]-=1
a=a[:,argsort(a[0])]
a=a[:,unique(a[1:],return_index=1,axis=1)[1]]
print([-1,int(sum(b:=csgraph.minimum_spanning_tree(coo_matrix((a[0],a[1:]),(n+1,n+1))).data))][len(b)==n])