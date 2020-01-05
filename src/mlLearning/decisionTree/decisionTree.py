import numpy as np
from math import log
import operator
import pickle
'''
创建数据集
年龄：0代表青年，1代表中年，2代表老年；
有工作：0代表否，1代表是；
有自己的房子：0代表否，1代表是；
信贷情况：0代表一般，1代表好，2代表非常好；
类别(是否给贷款)：no代表否，yes代表是。

'''
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']   #属性标签
    return dataSet, labels

# 经验熵
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCount={}
    for featVec in dataSet:
        currentLabel=featVec[-1]  #'yes' ot 'no'
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel]=0
        labelCount[currentLabel]+=1;
    shannonEnt=0.0
    for key in labelCount:
        prob=float(labelCount[key]/numEntries)
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

#计算信息增益
'''
 条件经验熵
'''

'''
Parameters:
    dataSet - 待划分的数据集
    axis - 划分数据集的特征
    value - 需要返回的特征的值
'''
# 按照给定特征与给定的特征值划分数据集以便于计算条件经验熵
# condtion:dataSet[i:len(dataSet)][axis]==value
def splitDataSet(dataSet, axis, value):
     retData=[]
     for featvec in dataSet:
         if featvec[axis]==value:
             reducedFeatVec=featvec[:axis]  #去除index==axis的特征= 拿index==axis之前的特征
             reducedFeatVec.extend(featvec[axis + 1:])  #拼接index==axis+1 及其之后的特征
             retData.append(reducedFeatVec)
     return retData

'''
 选择最优特征
 Returns:
    bestFeature - 信息增益最大的(最优)特征的索引值
'''
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  #特征数量
    baseEntropy = calcShannonEnt(dataSet)  #数据集的基本经验熵
    bestInfoGain = 0.0            #信息增益
    bestFeature = -1              #最优特征的索引值
    for i in range(numFeatures):
     featList = [example[i] for example in dataSet]  #第i个所有特征
     uniqueVals = set(featList)
     newEntropy = 0.0
     for value in uniqueVals:
         subDataSet=splitDataSet(dataSet,i,value)  #subDataSet为以condition：dataSet[i:len(dataSet)][axis]==value 划分后的子集
         prob=len(subDataSet)/float(len(dataSet))
         newEntropy+=prob*calcShannonEnt(subDataSet) #calcShannonEnt(subDataSet) 计算在满足condition的前提下的经验熵
     infoGain=baseEntropy-newEntropy     # 计算每一个特征的信息增益
     print("第%d个特征的增益为%.3f" % (i, infoGain))
     if (infoGain > bestInfoGain):
         bestInfoGain = infoGain  # 更新信息增益，找到最大的信息增益
         bestFeature = i
    return bestFeature

'''
函数说明:统计classList中出现此处最多的元素(类标签)
'''
def majorityCnt(classList):
    classCount = {}
    for vote in classList:                                        #统计classList中每个元素出现的次数
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)        #根据字典的值降序排序
    return sortedClassCount[0][0]

'''
函数说明:创建决策树
 Parameters:
    dataSet - 训练数据集
    labels - 分类属性标签
    featLabels - 存储选择的最优特征标签

dic 存储决策树
'''
def createTree(dataSet, labels, featLabels):
    classList = [example[-1] for example in dataSet]            #取分类标签(是否放贷:yes or no)
    if classList.count(classList[0]) == len(classList):         #如果类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1 or len(labels) == 0:                 #遍历完所有特征时返回出现次数最多的类标签，特征不够用
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)                #选择最优特征的index
    bestFeatLabel = labels[bestFeat]                            #最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}                                  #根据最优特征的标签生成树
    del(labels[bestFeat])                                        #删除已经使用特征标签
    featValues = [example[bestFeat] for example in dataSet]      #得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)                                #去掉重复的属性值
    for value in uniqueVals:                                   #遍历特征，创建决策树。
        subLabels = labels[:]   #下级
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, featLabels)
    return myTree

#分类
def classify(inputTree, featLabels, testVec):
	firstStr = next(iter(inputTree))			#获取决策树结点
	secondDict = inputTree[firstStr]		    #下一个字典
	featIndex = featLabels.index(firstStr)
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict':     #如果是dict 继续向下递归
				classLabel = classify(secondDict[key], featLabels, testVec)
			else: classLabel = secondDict[key]
	return classLabel

def storeTree(inputTree, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)
if __name__ == '__main__':
    dataSet, labels = createDataSet()
    featLabels = []
    myTree = createTree(dataSet, labels, featLabels)
    storeTree(myTree, 'classifierStorage.txt')
