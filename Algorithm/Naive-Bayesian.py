'''
条件概率公式：P(A|B) = P(A∩B)/P(B) = [P(B|A)*P(A)]/P(B)
全概率公式：
P(B) = P(B|A)*P(A)+P(B|A')*P(A')
P(A|B) = P(B|A)*P(A)/[P(B|A)*P(A)+P(B|A')*P(A')]
贝叶斯推断：
P(A|B) = P(A)*[P(B|A)/P(B)]
朴素贝叶斯：对条件个概率分布做了条件独立性的假设
后验概率　＝　先验概率 ｘ 调整因子
P(A|Xn) = P(A)*[P(Xn|A)/P(Xn)] = P(A)*[P(X1*X2*..*Xn|A)/P(X1*X2*..*Xn)] = P(A)*[P(X1|A)*P(X2|A)*..*P(Xn|A)/P(X1)*P(X2)*..*P(Xn)]
拉普拉斯平滑(Laplace Smoothing)又被称为加1平滑，是比较常用的平滑方法，它就是为了解决0概率问题。
'''
import numpy as np
from functools import reduce


# 创建样本
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


# 创建词条向量
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


# 创建词汇表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)       # 文档属于侮辱类的概率
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom, p1Denom = 2.0, 2.0                             # 分母初始化为2,拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)                            # 取对数，防止下溢出
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive


# 朴素贝叶斯分类器分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    print(p1Vec)
    print(vec2Classify * p1Vec)
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)  # logab = loga + logb
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    print('p0:', p0)
    print('p1:', p1)
    if p1 > p0:
        return 1
    else:
        return 0


# 测试朴素贝叶斯分类器
def testingNB():
    listOPosts, listClasses = loadDataSet()									# 创建实验样本
    myVocabList = createVocabList(listOPosts)								# 创建词汇表
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))				# 将实验样本向量化
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))		# 训练朴素贝叶斯分类器
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')
    else:
        print(testEntry, '属于非侮辱类')
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')
    else:
        print(testEntry, '属于非侮辱类')


if __name__ == '__main__':
    testingNB()
