import numpy as np
from functools import reduce

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                #切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList, classVec
'''
  生成词汇表，目的是将词条向量化，一个单词在词汇表中出现过一次，那么就在相应位置记作1，如果没有出现就在相应位置记作0
'''
def createVocabList(dataSet):
    vocabSet = set([])                  #创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document)      #取并集
    return list(vocabSet)
'''
 将每一个词条转换成词条向量
 Parameters:
    vocabList - 词汇表
    inputSet - 切分的词条列表
'''
def setOfWords2Vec(vocabList, inputSet):
    returnVec=[0]*len(vocabList)  # 生成长度为len(vocabList) 的一维数组
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

'''
训练模型所需的各个条件概率
Parameters:
    trainMatrix - 训练文档矩阵
    trainCategory - 训练类别标签向量 0/1 1 表示侮辱性的
问题：
   1. 0概率问题 <-拉普拉斯平滑
   2.  条件概率连乘下溢出 
'''
def trainNB(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)   #计算训练的文档数目
    numWords=len(trainMatrix[0])    #计算每篇文档的词条数
    pAbusive = sum(trainCategory) / float(numTrainDocs)  #文档属于侮辱类的概率
    p0Num = np.ones(numWords)   #统计属于非侮辱类的条件概率所需的数据，非侮辱类词条中一共出现的各个词汇的次数 使用ones目的是解决0概率问题
    p1Num = np.ones(numWords)   #统计属于侮辱类的条件概率所需的数据，侮辱类词条中一共出现的各个词汇的次数
    p0Denom = 2.0  #非侮辱性词条中里边的词汇数
    p1Denom = 2.0  #侮辱性词条里边的所有词汇数
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:                 #统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:                                     #统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect =np.log(p1Num/p1Denom)                #取自然对数
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive
'''
 分类器，通过训练出来的朴素贝叶斯分类器的所有条件概率计算后验概率
 Parameters:
    vec2Classify - 待分类的词条数组，只存在0/1的词汇数组 vec2Classify * p1Vec 等于各个词汇出现的次数*训练所得到的条件概率
    p0Vec - 侮辱类的条件概率数组
    p1Vec -非侮辱类的条件概率数组
    pClass1 - 文档属于侮辱类的概率
Returns:
    0 - 属于非侮辱类
    1 - 属于侮辱类
'''
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify*p1Vec)+np.log(pClass1)            #
    p0 = sum(vec2Classify*p0Vec)+np.log(1-pClass1)
    print('p0:',p0)
    print('p1:',p1)
    if p1 > p0:
        return 1
    else:
        return 0

def TestNB():
    listOPosts, listClasses = loadDataSet()   #训练样本
    myVocabList = createVocabList(listOPosts) #词汇表
    trainMat = []  #训练矩阵，每一个词条中每一个单词出现的次数
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB(np.array(trainMat), np.array(listClasses)) # 得出条件概率

    #测试1
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry)) #获得测试所得的词条向量
    result=classifyNB(thisDoc,p0V,p1V,pAb)
    print(result)
if __name__ == '__main__':
    TestNB();
