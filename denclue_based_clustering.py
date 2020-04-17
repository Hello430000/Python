# coding=utf-8
import numpy as np
import math
import copy

def kernelize(x,y,h,degree):
    kernel = np.exp(-(np.linalg.norm(x - y) / h) ** 2 / 2) / ((2. * np.pi) ** (degree/2))
    return kernel

def density(x,data,h,degree):
    m=0
    for i in range(len(data)):
        m = m + kernelize(x,data[i],h,degree)
    d = m / (len(data) * h**degree)
    return d

def den_arrive(x,y):
    a=[]
    b=[]
    d = 0.3
    if len(x)!=1 and len(x)!=0 :
        for i in range(1,len(x)):
            distance = math.sqrt((x[0][0]-x[i][0])**2+(x[0][1]-x[i][1])**2)
            if distance <= d:
                a.append(x[i])
            else:
                b.append(x[i])
        a.append(x[0])
        aa=copy.deepcopy(a)
        bb=copy.deepcopy(b)
        for i in range(len(aa)-1):
            for j in range(len(bb)):
                distance = math.sqrt((aa[i][0]-bb[j][0])**2+(aa[i][1]-bb[j][1])**2)
                if distance <= d:
                    a.append(bb[j])
                    b.remove(bb[j])
    elif len(x)==1:
        a = x
    elif len(x)==0:
        return
    y.append(a)
    den_arrive(b,y)
    return y

#计算X(t+1)
def next_step(d, data, h):
    m, n = data.shape
    xx = np.zeros((1, n))
    W = np.ones((m, 1))
    w = 0
    for i in range(m):
        k = kernelize(d, data[i], h, n)
        k= k * W[i] / (h ** n)
        w = w + k
        xx = xx + (k * data[i])
    xx = xx / w
    a = w / np.sum(W)
    return [xx, a]

def find_attractor(d, D, h, e):
    x1 = np.copy(d)
    b = 0
    while True:
        x0 = np.copy(x1)
        x1, a = next_step(x0, D, h)
        ee = a - b
        b = a
        if ee < e:
            break
    return x1[0]

def Denclue(data,min,e,h):
    m, n = data.shape
    attractor = []        #存放密度吸引子
    point = {}  #到吸引子的点
    for i in range(m):
        den_attractor = find_attractor(data[i],data,h,e)
        Density = density(den_attractor,data,h,n)
        if Density >= min:
            if ",".join('%s' % x for x in den_attractor) in point:
                p = point[",".join('%s' % x for x in den_attractor)]
                p.append(data[i].tolist())
                point[",".join('%s' % x for x in den_attractor)] = p
            else:
                point[",".join('%s' % x for x in den_attractor)] = [data[i].tolist()]
            den_attractor = den_attractor.tolist()
            if den_attractor not in attractor:
                attractor.append(den_attractor)
    A = []  #分类后的密度吸引子
    cluster_num = np.array(den_arrive(attractor,A))
    print("总计分为"+str(len(cluster_num))+"个簇")
    a = 0
    for i in range(len(cluster_num)):
        num = 0
        for j in range(len(cluster_num[i])):
            num = num + len(point[",".join('%s' % x for x in cluster_num[i][j])])
        a = a + num
        print("第"+str(i+1)+"个簇的大小为："+str(num))
    print("--------------------------------")
    Point={} #最终到吸引子的点
    for i in range(len(cluster_num)):
        attractor_s=[]  #在一起的密度吸引子
        for j in range(len(cluster_num[i])):
            c = point[",".join('%s' % x for x in cluster_num[i][j])]
            for k in range(len(c)):
                attractor_s.append(c[k])
        Point[",".join('%s' % x for x in cluster_num[i][0])] = attractor_s

    for i in range(len(cluster_num)):
        print("第"+str(i+1)+"个簇的密度吸引子为：")
        for j in range(len(cluster_num[i])):
            print(cluster_num[i][j],end=" ")
            if (j+1) % 3 ==0:
                print("")
        print("")
        print("该簇中的一组点为：" )

        g = Point[",".join('%s' % x for x in cluster_num[i][0])]
        for j in range(len(g)):
            print(g[j],end=" ")
            if (j+1) % 10 ==0:
                print("")
        print("")
        print("--------------------------------")

    C_point = []  #分类后的点
    for i in range(len(cluster_num)):
        C_point1 = Point[",".join('%s' % id for id in A[i][0])]
        C_point.append(C_point1)
    right_num = 0
    for i in range(len(C_point)):
        num_setosa = 0
        num_versicolor = 0
        num_virginica =0
        for j in range(len(C_point[i])):
            seq=data.tolist().index(C_point[i][j]) + 1
            if data1[seq][2]=="setosa":
                num_setosa += 1
            elif data1[seq][2]=="versicolor":
                num_versicolor += 1
            elif data1[seq][2]=="virginica":
                num_virginica += 1
        right_num = right_num + max(num_setosa,num_versicolor,num_virginica)
    p = right_num / a
    print("聚类的纯度为："+str(p))

data = np.loadtxt("iris.txt",
                   usecols=(1, 2),
                   unpack=False,
                   skiprows=1,
                   delimiter=' ')
data1 = np.loadtxt("iris.txt",
                   dtype=str,
                   usecols=(5),
                   unpack=False,
                   skiprows=1,
                   delimiter=' ')
data1 = np.c_[data,data1]
Denclue(data, 0.3, 0.0001, 0.15) #第二个参数为最小密度，第三个参数为收敛公差，第四个参数为带宽
