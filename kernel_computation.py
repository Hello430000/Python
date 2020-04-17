import numpy as np

data = np.loadtxt("iris.txt",
                usecols=(1,2,3,4),
                unpack=False,
                skiprows=1,
                delimiter=' ')
m, n = data.shape

k = np.zeros((m,m))
for i in range(m):
    for j in range(m):
        k[i,j] = (np.dot(data[i],data[j]))**2
print("齐次二次核矩阵：")
print(k)
print("______________________________")
print("------------------------------")

a = np.eye(m) - np.ones((m,m)) / m
k_center = np.dot(np.dot(a,k),a)
print("中心化后的核矩阵：")
print (k_center)
print("______________________________")
print("------------------------------")

b = np.zeros((m,m))
for i in range(m):
    b[i,i] = 1 / (k[i,i]**0.5)
k_normalize = np.dot(np.dot(b,k),b)
print("标准化后的核矩阵：")
print(k_normalize)
print("______________________________")
print("------------------------------")

p_fs = np.zeros((m,10))
for i in range(m):
    for j in range(n):
        p_fs[i,j] = data[i,j]**2
    for i1 in range(n-1):
        for j1 in range(i1+1,n):
            j = j + 1
            p_fs[i,j] = 2**0.5 * data[i,i1] * data[i,j1]
print("变换到特征空间的点：")
print(p_fs)
print("______________________________")
print("------------------------------")

m1,n1 = p_fs.shape
p_center = np.zeros((m1,n1))
for i in range(n1):
    p_center[:,i] = p_fs[:,i] - np.mean(p_fs[:,i])
print("特征空间点的中心化：")
print(p_center)
print("______________________________")
print("------------------------------")

p_normalize = np.zeros((m1,n1))
for i in range(m):
    p_normalize[i] = p_fs[i] / np.linalg.norm(p_fs[i])
print("特征空间点的标准化")
print(p_normalize)
print("______________________________")
print("------------------------------")

#calculate centered kernel through centered fai
k_center1 = np.zeros((m,m))
for i in range(m):
    for j in range(m):
        k_center1[i,j] = np.dot(p_center[i],p_center[j])
print("用中心化特征空间的点计算中心化核矩阵：")
print(k_center1)
print("______________________________")
print("------------------------------")

#calculate normalized kernel through normalized fai
k_normalize1 = np.zeros((m,m))
for i in range(m):
    for j in range(m):
        k_normalize1[i,j] = np.dot(p_normalize[i],p_normalize[j])
print("用标准化特征空间的点计算标准化核矩阵：")
print(k_normalize1)