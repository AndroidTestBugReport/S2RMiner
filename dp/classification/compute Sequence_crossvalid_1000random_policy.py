from itertools import permutations 
from statistics import mean 
import numpy as np
from scipy.sparse import vstack
from scipy.sparse import hstack
from scipy import sparse
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold
import random



def computeOptimal(dataSet,indexList):
    runTimeAll=sparse.csr_matrix(np.array([]))
    for i in range(0,len(dataSet)):
        runTimeOneReport=[]
        #runTimeOneReport=sparse.csr_matrix(np.array([]))
        print(i)
        #cou=0
        for oneIndexSeq in indexList:
            #cou+=1
            #print(cou)
            
            time=0
            for item in oneIndexSeq:
                if "N" in dataSet[i][item]:
                    time=time+int(dataSet[i][item].replace("N",""))
                else:
                    time=time+int(dataSet[i][item])
                    runTimeOneReport.append(time)
                    #runTimeOneReport=hstack([runTimeOneReport,time])
                    
                    break
        #print(len(runTimeOneReport))
        
        
        if i==0:
            #runTimeAll=runTimeOneReport
            runTimeAll=sparse.csr_matrix(np.array(runTimeOneReport))
            #runTimeAll=np.array([runTimeOneReport])
        else:
            #runTimeAll=vstack((runTimeAll,sparse.csr_matrix(np.array(runTimeOneReport))))
            #runTimeAll.sum(0)
            runTimeAll=runTimeAll+sparse.csr_matrix(np.array(runTimeOneReport))
    
    
    print(np.shape(runTimeAll))

    #MinIndex=0
    #MinValue=1000000
    
    resultList=(runTimeAll/len(dataSet)).toarray()
    #resultList=runTimeAll.mean(0)[0]
    
    #minIndexList=np.where(resultList == resultList.min())[0]
    meanResult=np.min(resultList)
    
    
    returnList=[]
    for index in range(0,len(resultList[0])):
        if resultList[0][index]==meanResult:
            returnList.append(indexList[index])
    
    #minIndex=np.argmin(resultList)
    print(meanResult)
    #print(indexList[minIndexList])
    
    
    #print("aa")
    #print(resultList[minIndexList[0]])
    
    #returnList=[]
    '''
    for item in minIndexList:
        returnList.append(indexList[item])
    '''
    return returnList



def computTime(testDataSet,inputIndexListRandom):
    
    
    inputTime=0
    for trainIndexList in inputIndexListRandom:
    
        for i in range(0,len(testDataSet)):
            
            
            for item in trainIndexList:
                if "N" in testDataSet[i][item]:
                    inputTime=inputTime+int(testDataSet[i][item].replace("N",""))
                else:
                    inputTime=inputTime+int(testDataSet[i][item])                
                    break
    inputTime=inputTime/len(inputIndexListRandom)
    
    return  inputTime/len(testDataSet)



def computeFoldSet(dataSetWhole,trainIndex):
    
    trainDataSet=[]
    for index in trainIndex:
        trainDataSet.append(dataSetWhole[index])
    
    
    return trainDataSet
    
    
#b=sparse.csr_matrix(a)



randomNum=10000
dataSetWhole=[ ]
file = open("./"+"dataset-result", 'r')
contents = file.readlines()

oneBugreport=[]
for index,item in enumerate(contents):
    
    if (index+1)%11==0:
        oneBugreport.append(item.strip())
        dataSetWhole.append(oneBugreport)
        oneBugreport=[]
    else:
        oneBugreport.append(item.strip())
        
        
#print(dataSet[15])

#indexList=range(0,11)
indexList=range(0,11)
perm = permutations(indexList) 
indexList=list(perm)
print(len(indexList))


#for i in range(0,len(dataSet)):
dataSetWhole=dataSetWhole

#################

y=np.array(np.zeros(len(dataSetWhole)))


random.shuffle(dataSetWhole)
random.shuffle(dataSetWhole)
random.shuffle(dataSetWhole)

folder = KFold(n_splits=3,random_state=0,shuffle=False)
#folder = StratifiedKFold(n_splits=4,random_state=0,shuffle=False)


for trainIndex, testIndex in folder.split(dataSetWhole,y):
    
    
    #trainDataSet=dataSetWhole[trainIndex]
    
    print("train")
    trainDataSet=computeFoldSet(dataSetWhole,trainIndex)
    trainIndexListRandom=computeOptimal(trainDataSet,indexList)
    
    if len(trainIndexListRandom)>randomNum:
        trainIndexListRandom=random.sample(trainIndexListRandom,randomNum)
    
    
    
    
    #print(trainIndexList)
    
    print("test")
    testDataSet=computeFoldSet(dataSetWhole,testIndex)
    testIndexListRandom=computeOptimal(testDataSet,indexList)
    
    if len(testIndexListRandom)>randomNum:
        testIndexListRandom=random.sample(testIndexListRandom,randomNum)
    
    
    trainOnTrain=computTime(trainDataSet,trainIndexListRandom)
    print("trainOntrain:"+str(trainOnTrain))
    
    
    #print(testIndexList)
    testOptimalTime=computTime(testDataSet,testIndexListRandom)
    trainedTime=computTime(testDataSet,trainIndexListRandom)
    
    
    
    #trainedTime,testOptimalTime=computTimeDifferent(testDataSet,trainIndexListRandom,testIndexListRandom)
    trainPercentage=float(trainedTime)/testOptimalTime-1####trained=(1+x%)opt   how many percentage increased
    
    
    
    randomListRandom=random.sample(indexList,randomNum)
    randomTime=computTime(testDataSet,randomListRandom)
    
    
    #randomTime,testOptimalTime=computTimeDifferent(testDataSet,randomListRandom,testIndexListRandom)
    randomPercentage=float(randomTime)/testOptimalTime-1####trained=(1+x%)opt   how many percentage increased
    
    print("random")
    print(randomListRandom[0])
    
    
    
    
    outputFile = open("policy-cross-result/4fold-random.txt", 'a')
    outputFile.write("\n")
    outputFile.write("trained\n")
    outputFile.write("trainIndexList:  "+str(trainIndexListRandom[0]) +"\n")
    outputFile.write("trainLen:  "+str(len(trainIndexListRandom)) +"\n")
    outputFile.write("testIndexList:  "+str(testIndexListRandom[0]) +"\n")
    outputFile.write("testLen:  "+str(len(testIndexListRandom)) +"\n")
    outputFile.write("trainedTime:  "+str(trainedTime) +"\n")
    outputFile.write("testOptimalTime:  "+str(testOptimalTime) +"\n")
    outputFile.write("trainPercentage:  "+str(trainPercentage) +"\n")
    outputFile.write("randomTime:  "+str(randomTime) +"\n")
    outputFile.write("randomIndexList:"+str(randomListRandom[0])+"\n")
    outputFile.write("randomPercentage:  "+str(randomPercentage) +"\n")
    
    outputFile.close()
    
    
    
    
    

    
    

##673.875
#(2, 10, 0, 1, 3, 9, 4, 6, 5, 7, 8)




#print(len(list(perm)))