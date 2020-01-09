#Administrator - 当前用户名;
#2020/1/7 - 当前系统日期;
#20:16 - 当前系统时间;

import jieba  #中文分词工具
import os
import random
'''
 folder_path:'./SogouC/Sample'
 数据预处理
'''
def TextProcessing(folder_path,test_size = 0.2):
    folder_list=os.listdir(folder_path)  #folder_path 下的子目录
    data_list=[]
    class_list=[]
    for folder in folder_list:
        new_folder_path=os.path.join(folder_path,folder)
        files=os.listdir(new_folder_path)
        j=1
        for file in files:
            if j>100:
                break;
            with open(os.path.join(new_folder_path,file),'r', encoding = 'utf-8') as f:
                raw=f.read()

            word_cut=jieba.cut(raw,cut_all=False)
            word_list=list(word_cut)

            data_list.append(word_list)
            class_list.append(folder)
            j+=1
    data_class_list=list(zip(data_list,class_list))   #标签和内容一一对应
    random.shuffle(data_class_list)  #乱序data_class_list
    #划分训练集、测试集
    index=int(len(data_class_list)*test_size)
    train_list=data_class_list[index:]
    test_list=data_class_list[:index]
    #训练集、测试集解压缩
    train_data_list, train_class_list = zip(*train_list)        #训练集解压缩
    test_data_list, test_class_list = zip(*test_list)            #测试集解压缩

    all_word_dic={}  #词汇字典
    for word_list in train_data_list:
        for word in word_list:
            if word in all_word_dic.keys():
                all_word_dic[word]+=1
            else:
                all_word_dic[word]=1
    # 根据键的值倒序排序->去除
    all_words_tuple_list=sorted(all_word_dic.items(),key = lambda f:f[1],reverse=True)
    all_words_list, all_words_nums = zip(*all_words_tuple_list)  # 解压缩词汇表
    all_words_list=list(all_words_list)
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list

'''
  函数说明:读取文件里的内容，并去重
  word_file:./stopwords_cn.txt
'''
def MakeWordsSet(words_file):
    words_set = set()                                            #创建set集合
    with open(words_file, 'r', encoding = 'utf-8') as f:        #打开文件
        for line in f.readlines():                                #一行一行读取
            word = line.strip()                                    #去回车
            if len(word) > 0:                                    #有文本，则添加到words_set中
                words_set.add(word)
    return words_set
'''
 1.生成新闻词汇向量
 2.去除词频最高的前100个词
 3.去除没有分类特征的词或字
'''
def  words_dict(all_words_list, deleteN, stopwords_set = set()):
    feature_words = []
    n = 1
    for t in range(deleteN,len(all_words_list),1):
        if n > 1200:  # feature_words的维度为1000
            break
        #排除数字、无分类特征的词或字、长度小于1或者大于5的字或词
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
        n+=1
    return feature_words

'''
  数据处理函数
  返回
'''
def loadData():
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list=TextProcessing("./SogouC/Sample")
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)
    feature_words = words_dict(all_words_list, 100, stopwords_set)
    return train_data_list, test_data_list, train_class_list, test_class_list,feature_words

if __name__ == '__main__':
    loadData()