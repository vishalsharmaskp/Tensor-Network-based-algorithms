import numpy as np
from ncon import ncon 
from numpy import linalg as LA

d = 3
D = 30
N = 50

#creating a random tensor network
M = []
for i in range(N):
	M.append(i)
 

for i in range(N):
	if i == 0:
		M[i] = np.random.rand(1, d, D)
	elif i == N-1:
		M[i] = np.random.rand(D, d, 1)
	else:
		M[i] = np.random.rand(D, d, D)


#converting above tensor network to left canonical form
for i in range(N-1):

	A = M[i].copy()
 
	U, S, Vh = LA.svd(A.reshape(np.shape(A)[0]*np.shape(A)[1], np.shape(A)[2]), full_matrices= False)
 
	M[i] = U.reshape(int(np.shape(U)[0]/np.shape(A)[1]), np.shape(A)[1], np.shape(U)[1])
 
	M[i+1] = ncon([np.diag(S), Vh, M[i+1]],[[-1, 1], [1, 2], [2, -2, -3]])

#checking left canonical form 
for i in range(N-1):
	B = ncon([np.conj(M[i]), M[i]], [[1, 2, -1], [1, 2, -2]])
	C = np.identity(np.shape(B)[0])
	diff = LA.norm(C - B)
	print(diff)



 

