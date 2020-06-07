import numpy as np
from ncon import ncon 
from numpy import linalg as LA
import matplotlib.pyplot as plt
 
N = 50
 
#creating tensor network for AKLT Model 
M = []
for i in range(N):
	M.append(i)
 
for i in range(N):
	M[i] = np.zeros((2,3,2)) 

for i in range(N):
	M[i][1, 0, 0] = -np.sqrt(2/3)
	M[i][0, 1, 0] = -1/np.sqrt(3)
	M[i][1, 1, 1] = 1/np.sqrt(3)
	M[i][0, 2, 1] = np.sqrt(2/3)

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



#Calculating Local Magnetization for different Open Boundary Conditions.
# values of m and n determine the up and down boundary of M[0] and M[N-1]

A = M[0].copy()
B = M[N-1].copy()

m = 1
n = 2

mag_analytical = []
if m == n == 1:
	for j in range(1, N+1):
		exact = ((-1/3)**j -(-1/3)**(N-j+ 1))/(0.5*(1+(-1/3)**N))
		mag_analytical.append(exact)	
elif m==n ==2:
	for j in range(1, N+1):
		exact = -((-1/3)**j -(-1/3)**(N-j+ 1))/(0.5*(1+(-1/3)**N))
		mag_analytical.append(exact)	
    
elif m==1 and n==2:
	for j in range(1, N+1):
		exact = ((-1/3)**j +(-1/3)**(N-j+ 1))/(0.5*(1-(-1/3)**N))
		mag_analytical.append(exact)
else:
	for j in range(1, N+1):
		exact = ((-1/3)**j +(-1/3)**(N-j+ 1))/(0.5*(1-(-1/3)**N))
		mag_analytical.append(exact)
		
M[0] = A[m-1:m, :, :]	
M[N-1] = B[:, :, n-1:n]
M1, r1, r2 = Canonical(M, N, 2)

Sz = np.diag([+1, 0, -1])
mag = [] 

for j in range(1, N+1):
	T = np.identity(1)
	 
	for i in range(N): 
		T = ncon ([T, np.conj(M1[i])], [[-1, 1],[1, -2, -3]])
		if i == j-1:
			T = ncon([T, Sz], [[-1, 1, -2], [1, -3]])
			T = T.transpose(0, 2, 1)
	 
		T = ncon([T, M1[i]],[[1, 2, -1],[1, 2, -2]])
	mag.append(T[0, 0])
print("mag=", mag)

#ploting the local magnetizaition vs site for numerical and exat analytical values 
site= []
for k in range(1, N+1):
	site.append(k)
 
plt.plot(site, mag, 'r.')
plt.plot(site, mag_analytical, 'b+')
plt.xlabel('site')
plt.ylabel('Local Magnetization')
plt.show()

		
 



