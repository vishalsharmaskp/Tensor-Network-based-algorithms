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
 
def Canonical(M, id, Nkeep):
 	
	for i in range(id):
 
		A = M[i].copy()
 	
		U , S, Vh = LA.svd(A.reshape(np.shape(A)[0]*np.shape(A)[1], np.shape(A)[2]), full_matrices= False)
		
		if Nkeep < np.shape(U)[1]:
			U  = U[:, 0 : Nkeep]
			S  = np.diag(S)[0:Nkeep, 0:Nkeep]
			Vh = Vh[0:Nkeep, :]
			S = np.diag(S)
 	
		M[i] = U.reshape(int(np.shape(U)[0]/np.shape(A)[1]), np.shape(A)[1], np.shape(U)[1])
 
		
		r1 = np.matmul(np.diag(S), Vh)


		if i != id-1:
			M[i+1] = ncon([np.diag(S), Vh, M[i+1]],[[-1, 1], [1, 2], [2, -2, -3]])

		  
	for j in range(N - id):
				
		i = N - 1 - j

		A = M[i].copy()	 

		U , S, Vh = LA.svd(A.reshape(np.shape(A)[0], np.shape(A)[1]*np.shape(A)[2]), full_matrices= False)
		
		if np.shape(Vh)[0] > Nkeep:
			U  = U[:, 0 : Nkeep]
			S  = np.diag(S)[0:Nkeep , 0:Nkeep]
			Vh = Vh[0:Nkeep, :]		
			S  = np.diag(S)
		M[i] = Vh.reshape(np.shape(Vh)[0], np.shape(A)[1], int(np.shape(Vh)[1]/np.shape(A)[1]))

		r2 = np.matmul(U, np.diag(S))

		
		if j != N - id- 1:
			M[i-1] = ncon([M[i-1], U, np.diag(S)],[[-1, -2, 1], [1, 2], [2, -3]])

	if id== 0:
		r1 = 1
	if id ==N:
		r2 = 1	
	 
				 
	return M, r1, r2


id = 20
Nkeep = 25

M, R1, R2 = Canonical(M, id, Nkeep)

print(np.shape(R1))
print(np.shape(R2))
print("check")
for i in range(id):
	print(i)	
	B = ncon([np.conj(M[i]), M[i]], [[1, 2, -1], [1, 2, -2]])
	C = np.identity(np.shape(B)[0])
	diff = LA.norm(C - B)
	print('i', diff)

for j in range(N - id):
	i = N - 1 -j	 
	print(i)
	B = ncon([np.conj(M[i]), M[i]], [[-1, 1, 2], [-2, 1, 2]])
	C = np.identity(np.shape(B)[0])
	diff = LA.norm(C - B)
	print("i", diff)
 
 



 

