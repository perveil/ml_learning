import  numpy as np
import operator

#数据处理三维数据
def fileRead(filename):
    fr=open(filename)
    arrayLines=fr.readlines()
    numberOflines=len(arrayLines)
    returnMat=np.zeros((numberOflines,3)) #创建默认矩阵
    classLabelVector=[]
    index=0
    for line in arrayLines:
        line=line.strip()
        listFromLine=line.split('\t') #依据空格切割数据返回数组
        returnMat[index,:]=listFromLine[0:3]
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index+=1
    return returnMat, classLabelVector
#数据归一化处理
#newValue = (oldValue - min) / (max - min)
''' 
 防止数值较大的数据对于计算结果的不公平影响
'''
def autoNorm(dataSet):
    minVals=dataSet.min(0)  #返回每一列的最小值
    maxVals=dataSet.max(0)  #返回每一个列的最大值
    ranges=maxVals-minVals  #(max - min)
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1)) # np.tile 之后变为 m*3的矩阵
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def KNN0(inx,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=np.tile(inx,(dataSetSize,1))-dataSet  ## x-x(j)
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
    # key=operator.itemgetter(1)根据字典的值item进行排序
    return sortedClassCount[0][0]

def Test():
    filename = "./datingTestSet.txt"
    #打开并处理数据
    datingDataMat, datingLabels = fileRead(filename)
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    hoRatio = 0.10
    m = normDataSet.shape[0]
    # 百分之十的测试数据的个数
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = KNN0(normDataSet[i,:], normDataSet[numTestVecs:m,:],
            datingLabels[numTestVecs:m], 4)
        print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率:%f%%" % (errorCount / float(numTestVecs) * 100))

if __name__ == '__main__':
    Test();