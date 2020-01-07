#Administrator
#2020/1/7
#19:48
import re #python 正则匹配

def textParse(bigString):
    listOfTokens=re.split(r'\W+',bigString)   #使用字母切割字符串
    return [ tok.lower() for tok in listOfTokens if len(tok)>2]  #排除字符长度小于2的词汇

def createVocabList(dataSet):
    vocabSet = set([])                  #创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document)      #取并集
    return list(vocabSet)

'''
returns:
    vocabList - 词汇表
    docList - 词条矩阵
    classList- 类别标记
'''

def init():
    docList = []
    classList = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i, 'r').read())
        docList.append(wordList)
        classList.append(1)                                                 #标记垃圾邮件，1表示垃圾文件
        wordList = textParse(open('email/ham/%d.txt' % i, 'r').read())      #读取每个非垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)
    return vocabList,docList,classList

if __name__ == '__main__':
    vocabList, docList, classList=init()
    print(vocabList)