import numpy as np
import operator
def createDataSet():
    group=np.array([
        [1,101],[5,89],
        [108,5],[115,8]
    ])
    labels=['爱情片','爱情片','动作片','动作片']
    return group,labels
'''
:param inx 测试数据集 data训练数据集
:connotation 二维KNN
'''
def KNN0(inx,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=np.tile(inx,(dataSetSize,1))-dataSet ## x-x(j)
    '''
    np.tile 堆砌函数
    :param 
    inx 目标矩阵 
    (列，行) 堆砌次数
    '''
    sqDiffMat=diffMat**2; # 求平方
    sqDistances=sqDiffMat.sum(axis=1) #行相加
    distances=sqDistances**0.5 #开方
    sortedDistancesIndices=distances.argsort() #排序后返回索引值
    classCount={} #字典
    for i in range(k):
        voteIable=labels[sortedDistancesIndices[i]]
        classCount[voteIable]=classCount.get(voteIable,0)+1

    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    # key=operator.itemgetter(1)根据字典的值进行排序
    return sortedClassCount[0][0]

if __name__ == '__main__':
    group, labels=createDataSet()
    test=[60,20]
    test_class=KNN0(test,group,labels,3);
    print(test_class);
