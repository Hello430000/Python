# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
#读数据
data = np.loadtxt("magic1.txt",
                usecols=(0,1,2,3,4,5,6,7,8,9),
                unpack=False,
                skiprows=0,
                delimiter=',')
np.set_printoptions(suppress = True)
m, n = data.shape

#计算多元平均向量
Mean=np.mean(data,0)
print("多元平均向量为：")
print(Mean)

print("------------------------------")
#计算协方差内积
data1 = copy.deepcopy(data)
print("数据中心化：")
for i in range(0,n):
    data1[:,i] = data1[:,i] - Mean[i]
print(data1)
data1T = data1.T
Inner = np.dot(data1T,data1)*(1/m)
print("协方差矩阵作为内积：")
print(Inner)

print("------------------------------")
#计算协方差外积
Outer = np.zeros((n,n))
for i in range(0,m):
    Outer = Outer + np.outer(data1T[:,i],data1[i])
Outer = Outer/m
print("协方差矩阵作为外积：")
print(Outer)

print("------------------------------")
#相关性
print("属性1和属性2之间夹角的余弦值为：")
cosine = np.dot(data1[:,0],data1[:,1])/(np.linalg.norm(data1[:,0])*np.linalg.norm(data1[:,1]))
print(cosine)
for i in range(m):
    plt.plot(data[i, 0], data[i, 1], 'ok' )
plt.title("")
plt.xlabel("X1: fLength")
plt.ylabel("X2: fWidth")
plt.show()

print("------------------------------")
#正态分布概率密度函数
u=Mean[0]
o2 = np.var(data[:,0])
o = np.std(data[:,0],ddof=1)
x = np.linspace(u - 4*o, u + 4*o, 100)
y = np.exp(-(x - u) ** 2 /(2* o **2))/(math.sqrt(2*math.pi)*o)
plt.plot(x, y, "b-", linewidth=1)
plt.grid(True)
plt.title("normal distribution(μ="+ str(round(u,3))+",σ^2 ="+str(round(o2,3))+")")
plt.xlabel("X1: fLength")
plt.ylabel("X2: f(x1)")
plt.show()

#方差最值
print("各个属性的方差：")
variance = np.var(data,axis=0)
print(variance)
print("第"+str(int(np.argmax(variance))+1)+"个属性的方差最大，为："+str(np.max(variance)))
print("第"+str(int(np.argmin(variance))+1)+"个属性的方差最小，为："+str(np.min(variance)))

print("------------------------------")
#协方差最值
a=data.T
covariance=np.cov(a)
print("协方差矩阵为：")
print(covariance)
u = np.triu(covariance,1) # 上三角矩阵
m1, n1 = divmod(np.argmax(u), 10)
m2, n2 = divmod(np.argmin(u), 10)
print("第"+str(m1+1)+"个属性和第"+str(n1+1)+"个属性的协方差最大，为："+str(np.max(u)))
print("第"+str(m2+1)+"个属性和第"+str(n2+1)+"个属性的协方差最小，为："+str(np.min(u)))