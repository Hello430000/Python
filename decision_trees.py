import numpy as np
import pandas as pd
import copy
import math

# 计算某一属性中的最大增益的对应的分割点和增益值
def splitpoint_gain(data, ii, Species):
    data1 = copy.deepcopy(data)
    data1 = data1[data1[:,ii].argsort()]
    splitpoint = []  # 切分点的集合
    species_num = []  #所有分裂点之前的分类情况
    L = len(data1)
    n1 = 0
    n2 = 0
    n3 = 0
    for i in range(L - 1):
        if data1[i][4] == Species[0]:
            n1 += 1
        elif data1[i][4] == Species[1]:
            n2 += 1
        elif data1[i][4] == Species[2]:
            n3 += 1
        if data1[i][ii] != data1[i + 1][ii]:
            point = (data1[i][ii] + data1[i+1][ii]) / 2
            splitpoint.append(point)
            nn = [n1, n2, n3]
            species_num.append(nn)
    num = count(data1,Species)
    H = 0
    for i in range(3):
        if num[i]!=0:
            H = H - (num[i]/ L) * math.log(num[i] / L,2)
    bestpoint = []
    gain = 0
    for i in range(len(splitpoint)):
        species_num1 = species_num[i][0]+species_num[i][1]+species_num[i][2]
        species_num2 = L - species_num1
        H1 = 0
        H2 = 0
        for j in range(3):
            p1 = species_num[i][j] / species_num1
            p2 = (num[j] - species_num[i][j]) / species_num2
            if p1!=0:
                H1 = H1 - p1 * math.log(p1, 2)
            if p2!=0:
                H2 = H2 - p2 * math.log(p2, 2)
        gain1 = H - (species_num1 / L) * H1 - (species_num2 / L) * H2
        if gain1 > gain:
            gain = gain1
            bestpoint = splitpoint[i]
    return bestpoint, gain

# 找所有属性中增益最大的属性
def best_attribute(data,Species):
    best_gain = 0
    attribute_num = 0
    for i in range(4):
        splitpoint, gain = splitpoint_gain(data, i, Species)
        if gain > best_gain:
            best_gain = gain
            attribute_num = i
            best_splitpoint = splitpoint
    return attribute_num,best_splitpoint,best_gain

# 统计数据中三个分类的数量
def count(data,Species):
    num1=num2=num3=0
    for i in range(len(data)):
        if data[i][4] == Species[0]:
            num1 = num1 + 1
        elif data[i][4] == Species[1]:
            num2 = num2 + 1
        elif data[i][4] == Species[2]:
            num3 = num3 + 1
    num=[num1,num2,num3]
    return num

# 按照最佳增益分割数据
def splitdata(data,attribute_num,best_splitpoint):
    front = []
    back = []
    for j in range(len(data)):
        if data[j][attribute_num] <= best_splitpoint:
            front.append(data[j])
        else:
            back.append(data[j])
    left = np.array(front)
    right = np.array(back)
    return left, right

def decisionTree(data, min_size, min_purity):
    Species = ['setosa','versicolor','virginica']
    size = len(data)
    num = count(data,Species)
    purity = max(num)/size
    if size <= min_size or purity >= min_purity:
        c = Species[np.argmax(num)]
        print("该节点是叶子节点 多数属性："+str(c)+",分类纯度："+str(purity)+",节点大小："+str(size))
        print("---")
        return
    attribute_num, best_splitpoint, best_gain = best_attribute(data,Species)
    attribute = ['Sepal_Length', 'Sepal_Width', 'Petal_Length','Petal_Width']
    print("节点的分裂点："+str(attribute[attribute_num])+" <= " + str(best_splitpoint)+"  信息增益："+str(best_gain))
    data_left, data_right = splitdata(data, attribute_num, best_splitpoint)
    print("节点 "+str(attribute[attribute_num])+" <= " + str(best_splitpoint)+" 的左子树 Ture"+" ：")
    decisionTree(data_left, min_size, min_purity)
    print("节点 "+str(attribute[attribute_num])+" <= " + str(best_splitpoint)+" 的右子树 False"+" ：")
    decisionTree(data_right, min_size, min_purity)

data = pd.read_csv('iris.txt',
                header=None,
                skiprows=1,
                sep=" ",
                usecols=[1,2,3,4,5])
data = np.array(data)
decisionTree(data,5,0.95)
