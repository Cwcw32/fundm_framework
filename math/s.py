import numpy as np
from scipy.linalg import solve

print(1)
a = np.array([[-1,1/2], [-2,1]]) #每一个方程对应的系数
print(a)
#b = np.array([[1],[2]])  #常数项系数
#print(b)
c=np.array([[0],[0]])
#x = solve(a, b)
#print(x) #输出解
print(1)
print(c)
y= solve(a,c)
print(y)