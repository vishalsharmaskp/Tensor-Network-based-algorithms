import numpy as np
from ncon import ncon 
from numpy import linalg as LA

 
d = 3
D = 30
N = 50

M = []

#creating a MPS

for i in range(N):
	M.append(i)

for i in range(N):
	if i == 0:
		M[i] = np.random.rand(1, d, D)
		 
	elif i == N-1:
		M[i] = np.random.rand(D, d, 1)

	else:
		M[i] = np.random.rand(D, d, D) 

#converting to a right canonical form

for j in range(N):
	i = N- 1 - j

	A = M[i].copy()	 

	U, S, Vh = LA.svd(A.reshape(np.shape(A)[0], np.shape(A)[1]*np.shape(A)[2]), full_matrices= False)

	M[i] = Vh.reshape(np.shape(Vh)[0], np.shape(A)[1], int(np.shape(Vh)[1]/np.shape(A)[1]))

	if i != 0:

		M[i-1] = ncon([M[i-1], U, np.diag(S)],[[-1, -2, 1], [1, 2], [2, -3]])
 
#checking canonical form 
for j in range(N):
	i = N - 1 -j	 
	B = ncon([np.conj(M[i]), M[i]], [[-1, 1, 2], [-2, 1, 2]])
 	
	C = np.identity(np.shape(B)[0])

	diff = LA.norm(C - B)
#	print(diff)
	print('Done')
