
# from readfile import readfiletree
from numpy import *
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)   # 计算数据中的实例总数
    labelCounts = {}       # 创建字典，保存各类标签的数量
    for featVec in dataSet: # 计算各个类的数量
        currentLabel = int(featVec[-1])
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    ent = 0.0
    # print(labelCounts)
    for key in labelCounts:
        prob = labelCounts[key]/numEntries
        ent -= prob * math.log(prob,2)
    # print(ent)
    return ent
def splitDataSet_c(dataSet, axis, value, LorR='L'):# 划分数据集, axis:按第几个特征划分, value:划分特征的值, LorR: value值左侧（小于）或右侧（大于）的数据集
    retDataSet = []
    featVec = []
    if LorR == 'L':
        for featVec in dataSet:
            if float(featVec[axis]) < value:
                retDataSet.append(featVec)
    else:
        for featVec in dataSet:
            if float(featVec[axis]) > value:
                retDataSet.append(featVec)
    # print(retDataSet)
    return retDataSet


def chooseBestFeatureToSplit_c(dataSet, labelProperty):# 选择最好的数据集划分方
    numFeatures = len(labelProperty)  # 特征数
    baseEntropy = calcShannonEnt(dataSet)  # 计算根节点的信息熵
    bestInfoGain = 0.0
    bestFeature = -1
    bestPartValue = None  # 连续的特征值，最佳划分值
    for i in range(numFeatures):  # 对每个特征循环
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 该特征包含的所有值
        newEntropy = 0.0;
        bestPartValuei = None
        if labelProperty[i] == 0:  # 对离散的特征
            for value in uniqueVals:  # 对每个特征值，划分数据集, 计算各子集的信息熵
                subDataSet = splitDataSet_c(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob * calcShannonEnt(subDataSet)
        else:  # 对连续的特征
            sortedUniqueVals = list(uniqueVals)  # 对特征值排序
            sortedUniqueVals.sort()
            listPartition = []
            minEntropy =float("inf")
            for j in range(len(sortedUniqueVals) - 1):  # 计算划分点
                partValue = (float(sortedUniqueVals[j]) + float(
                    sortedUniqueVals[j + 1])) / 2
                # 对每个划分点，计算信息熵
                dataSetLeft = splitDataSet_c(dataSet, i, partValue, 'L')
                dataSetRight = splitDataSet_c(dataSet, i, partValue, 'R')
                probLeft = len(dataSetLeft) / float(len(dataSet))
                probRight = len(dataSetRight) / float(len(dataSet))
                Entropy = probLeft * calcShannonEnt(
                    dataSetLeft) + probRight * calcShannonEnt(dataSetRight)
                if Entropy < minEntropy:  # 取最小的信息熵
                    minEntropy = Entropy
                    bestPartValuei = partValue
            newEntropy = minEntropy
        infoGain = baseEntropy - newEntropy  # 计算信息增益
        if infoGain > bestInfoGain:  # 取最大的信息增益对应的特征
            bestInfoGain = infoGain
            bestFeature = i
            bestPartValue = bestPartValuei
    # print(bestFeature,bestPartValue)
    return bestFeature, bestPartValue                                
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=lambda itemLabel:itemLabel[1], reverse=True)
    return sortedClassCount[0][0]
def createTree_c(dataSet, labels, labelProperty):# 创建树, 样本集 特征 特征属性（0 离散， 1 连续）
    # print dataSet, labels, labelProperty
    classList = [example[-1] for example in dataSet]  # 类别向量
    if classList.count(classList[0]) == len(classList):  # 如果只有一个类别，返回
        return classList[0]
    if len(dataSet[0]) == 1:  # 如果所有特征都被遍历完了，返回出现次数最多的类别
        return majorityCnt(classList)
    bestFeat, bestPartValue = chooseBestFeatureToSplit_c(dataSet, labelProperty)  # 最优分类特征的索引
    if bestFeat == -1:  # 如果无法选出最优分类特征，返回出现次数最多的类别
        return majorityCnt(classList)
    if labelProperty[bestFeat] == 0:  # 对离散的特征
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel: {}}
        labelsNew = copy.copy(labels)
        labelPropertyNew = copy.copy(labelProperty)
        del (labelsNew[bestFeat])  # 已经选择的特征不再参与分类
        del (labelPropertyNew[bestFeat])
        featValues = [example[bestFeat] for example in dataSet]
        uniqueValue = set(featValues)  # 该特征包含的所有值
        for value in uniqueValue:  # 对每个特征值，递归构建树
            subLabels = labelsNew[:]
            subLabelProperty = labelPropertyNew[:]
            myTree[bestFeatLabel][value] = createTree_c(
                splitDataSet_c(dataSet, bestFeat, value), subLabels,
                subLabelProperty)
    else:  # 对连续的特征，不删除该特征，分别构建左子树和右子树
        bestFeatLabel = str(labels[bestFeat]) + '<' + str(bestPartValue)
        myTree = {bestFeatLabel: {}}
        subLabels = labels[:]
        subLabelProperty = labelProperty[:]
        # 构建左子树
        valueLeft = '是'
        myTree[bestFeatLabel][valueLeft] = createTree_c(
            splitDataSet_c(dataSet, bestFeat, bestPartValue, 'L'), subLabels,
            subLabelProperty)
        # 构建右子树
        valueRight = '否'
        myTree[bestFeatLabel][valueRight] = createTree_c(
            splitDataSet_c(dataSet, bestFeat, bestPartValue, 'R'), subLabels,
            subLabelProperty)
    # print(myTree)
    return myTree
def classify_c(inputTree, featLabels, featLabelProperties, testVec):
    firstStr = list(inputTree.keys())[0]  # 根节点
    firstLabel = firstStr
    # print(firstLabel)
    # print("---")

    lessIndex = str(firstStr).find('<')
    if lessIndex > -1:  # 如果是连续型的特征
        firstLabel = str(firstStr)[:lessIndex]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstLabel)  # 根节点对应的特征
    classLabel = None
    for key in secondDict.keys():  # 对每个分支循环
        if featLabelProperties[featIndex] == 0:  # 离散的特征
            if testVec[featIndex] == key:  # 测试样本进入某个分支
                if type(secondDict[key]).__name__ == 'dict':  # 该分支不是叶子节点，递归
                    classLabel = classify_c(secondDict[key], featLabels,
                                           featLabelProperties, testVec)
                else:  # 如果是叶子， 返回结果
                    classLabel = secondDict[key]
        else:
            partValue = float(str(firstStr)[lessIndex + 1:])
            if testVec[featIndex] < partValue:  # 进入左子树
                if type(secondDict['是']).__name__ == 'dict':  # 该分支不是叶子节点，递归
                    classLabel = classify_c(secondDict['是'], featLabels,
                                           featLabelProperties, testVec)
                else:  # 如果是叶子， 返回结果
                    classLabel = secondDict['是']
            else:
                if type(secondDict['否']).__name__ == 'dict':  # 该分支不是叶子节点，递归
                    classLabel = classify_c(secondDict['否'], featLabels,
                                           featLabelProperties, testVec)
                else:  # 如果是叶子， 返回结果
                    classLabel = secondDict['否']
 
    return classLabel
# def storeTree(inputTree,filename):
#     import pickle
#     fw = open(filename,'wb')
#     pickle.dump(inputTree,fw)
#     fw.close()
# def grabTree(filename):
#     import pickle
#     fr = open(filename, 'rb')
#     return pickle.load(fr)
# def generatetree():
#     data,attributes,label = readfiletree()
#     mytree = createTree_c(data,attributes,label)
#     storeTree(mytree,'tree.dat')




def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,4))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[:]
        # classLabelVector.append(int(listFromLine[-1]))
        index += 1
    classLabelVector=['飞机里程','游戏时间','冰激淋消耗量']
    return returnMat,classLabelVector






if __name__=='__main__':
    dataSet,labels=file2matrix('datingTestSet2.txt')
    # print(dataSet)
    calcShannonEnt(dataSet)
    # print(labels)
    labelProperty=[1,1,1]
    chooseBestFeatureToSplit_c(dataSet,labelProperty)
    tree=createTree_c(dataSet,labels,labelProperty)
    print(tree)
    # 取30%的数据进行验证，统计正确率]
    test=[]
    for data in dataSet[:30]:
        test.append(data[:-1])
    print(test)
    for data in test:
        print(classify_c(tree, labels, labelProperty, data))

    # print(classify_c(tree,labels,labelProperty,[77372,15.299570,0.331351]))



