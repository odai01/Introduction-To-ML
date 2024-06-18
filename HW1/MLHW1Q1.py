import numpy as np 
import matplotlib.pyplot as plt
#   question 1a
N=200000
n=20
pr=0.5
matrix=np.random.choice([0,1],size=(N,n),p=[1-pr,pr])
EmpiricalMeans=np.mean(matrix,axis=1)
#   question 1b+1c
epss=np.linspace(0,1,50)
EmpPr=[]
HoeffdingBounds=[]
count=0
for eps in epss:
    for Xi in EmpiricalMeans:
        if(abs(Xi-0.5)>eps):
            count+=1
    EmpPr.append(count/N)
    count=0
    HoeffdingBounds.append(2*(np.exp(-2*eps*eps*n)))

plt.plot(epss, EmpPr,color='m',label='Empirical Probability')
plt.plot(epss,HoeffdingBounds,color='c',label='HoeffdingBound')
plt.xlabel('epsilon')
plt.ylabel('Empirical Probability\HoeffdingBound')
plt.legend()
plt.show()

