

from numpy import *

def classify0(inX, dataSet, labels, k):
    dataSetSize=dataSet.shape[0] 
    diffMat = tile(inX, (dataSetSize, 1))-dataSet 
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1) 
    distances = sqDistance ** 0.5        
    sortedDistIndicies = distances.argsort()   
    classCount={}    
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
    classCount[voteIlabel] = classCount.get(voteIlabel,0) +1       
    sortedClassCount=sorted(classCount.items(),key=lambda itemLabel:itemLabel[1],reverse=True) 
    return sortedClassCount[0][0]             
    
def createDataSet(data):
    numberOfLines = len(data)   
    returnMat = empty((numberOfLines,3)) 
    classLabelVector = []  
    for index in range(numberOfLines):
        temp = data[index][0:3]
        returnMat[index,:] = array(temp)
        classLabelVector.append(int(data[index][-1]))
    return returnMat, classLabelVector
        
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]               
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   
    return normDataSet, ranges, minVals

def clearify_knn(traindataset,checkdataset,k):
    matrix, label=createDataSet(traindataset)
    normDataSet, ranges, minVals=autoNorm(matrix)
    smatrix, slabel=createDataSet(traindataset)
    outcome=[]
    for item in smatrix:
        result = classify0(item, matrix, label,k)
        outcome.append(result)
    return outcome

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print ("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print (errorCount)


if __name__=='__main__':
    datingClassTest()